"""
vLLM Student Reward Service

使用独立部署的 vLLM 服务计算 Student Reward
- 认知奖励：使用 prompt_logprobs 计算 log P(Answer | Question, CoT)
- FSF 奖励：使用 vLLM 生成 FSF 分析结果

架构：
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0-2: Policy Model + Rollout (训练进程)                       │
└─────────────────────────────────────────────────────────────────┘
           │ HTTP API
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ GPU 3: vLLM Server (独立进程)                                    │
│   - Student Model (Qwen2.5-3B)                                  │
│   - FSF LoRA Adapter (可选)                                     │
└─────────────────────────────────────────────────────────────────┘

启动 vLLM 服务：
    # 基础模型
    python -m vllm.entrypoints.openai.api_server \
        --model ./models/Qwen2.5-3B-Instruct \
        --port 8100 \
        --gpu-memory-utilization 0.9

    # 带 LoRA（如果需要 FSF）
    python -m vllm.entrypoints.openai.api_server \
        --model ./models/Qwen2.5-3B-Instruct \
        --enable-lora \
        --lora-modules fsf_lora=./outputs/fsf_lora \
        --port 8100

使用方法：
    配置 custom_reward_function:
        path: src/reward/vllm_student_reward.py
        name: compute_vllm_student_reward
        reward_kwargs:
            vllm_base_url: http://localhost:8100/v1
            use_fsf_lora: true
"""

import asyncio
import json
import re
from typing import Any, Optional, List, Dict
import torch
import httpx

from verl import DataProto


class VLLMStudentRewardClient:
    """
    vLLM Student Reward 客户端

    通过 HTTP API 调用 vLLM 服务计算奖励
    """

    FSF_PROMPT_TEMPLATE = """<|im_start|>system
你是一个推理过程分析专家。请分析推理过程中的"失败分支"占比。
失败分支定义：尝试了某种方法但最终放弃的分支。
请输出 JSON 格式：{{"total_steps": N, "failed_steps": M, "fsf": 0.0到1.0的数值}}<|im_end|>
<|im_start|>user
分析以下推理过程的失败分支占比：

{reasoning}

只输出 JSON，不要其他文字。<|im_end|>
<|im_start|>assistant
"""

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8100/v1",
        model_name: str = "Qwen2.5-3B-Instruct",
        fsf_lora_name: Optional[str] = "fsf_lora",  # LoRA adapter 名称
        timeout: float = 60.0,
        max_concurrent: int = 32,
    ):
        """
        初始化 vLLM 客户端

        Args:
            vllm_base_url: vLLM 服务地址
            model_name: 基础模型名称
            fsf_lora_name: FSF LoRA adapter 名称（如果 vLLM 启用了 --enable-lora）
            timeout: 请求超时时间
            max_concurrent: 最大并发数
        """
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self.fsf_lora_name = fsf_lora_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 用于 prompt_logprobs 计算
        self.completions_url = f"{self.vllm_base_url}/completions"
        # 用于 FSF 生成（chat 接口）
        self.chat_url = f"{self.vllm_base_url}/chat/completions"

    # ==================== 认知奖励 ====================

    async def _compute_single_log_prob_async(
        self,
        question: str,
        rewrite_cot: str,
        answer: str,
        client: httpx.AsyncClient,
    ) -> float:
        """
        异步计算单个样本的 log P(Answer | Question, CoT)

        使用 vLLM 的 prompt_logprobs 功能
        """
        # 构造 prefix 和完整文本
        prefix = f"Question: {question}\n\nReasoning: {rewrite_cot}\n\nAnswer: "
        full_text = f"{prefix}{answer}"

        async with self.semaphore:
            try:
                response = await client.post(
                    self.completions_url,
                    json={
                        "model": self.model_name,
                        "prompt": full_text,
                        "max_tokens": 1,  # 不需要生成
                        "prompt_logprobs": 1,  # 返回 prompt 的 logprobs
                        "echo": True,  # 返回完整序列（包含 prompt）
                        "temperature": 0,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                # 解析 logprobs
                choice = result["choices"][0]

                if "logprobs" in choice and choice["logprobs"]:
                    logprobs_data = choice["logprobs"]

                    # 获取 tokens 和对应的 logprobs
                    tokens = logprobs_data.get("tokens", [])
                    token_logprobs = logprobs_data.get("token_logprobs", [])

                    if not tokens or not token_logprobs:
                        return -100.0

                    # 找到 Answer 部分的起始位置
                    # 方法：通过累积 tokens 找到 prefix 结束的位置
                    prefix_tokens_count = self._estimate_prefix_tokens(
                        tokens, prefix
                    )

                    # 提取 Answer 部分的 logprobs 并求和
                    answer_logprobs = token_logprobs[prefix_tokens_count:]

                    # 过滤掉 None（第一个 token 通常是 None）
                    answer_logprobs = [lp for lp in answer_logprobs if lp is not None]

                    if answer_logprobs:
                        return sum(answer_logprobs)

                return -100.0

            except Exception as e:
                print(f"[VLLMStudentReward] 计算 log_prob 失败: {e}")
                return -100.0

    def _estimate_prefix_tokens(self, tokens: List[str], prefix: str) -> int:
        """
        估算 prefix 对应的 token 数量

        通过累积 tokens 直到匹配 prefix 长度
        """
        accumulated = ""
        for i, token in enumerate(tokens):
            accumulated += token
            # 当累积长度接近或超过 prefix 时，认为 prefix 结束
            if len(accumulated) >= len(prefix) - 10:  # 允许一些误差
                return i + 1
        return len(tokens) // 2  # fallback

    async def compute_cognitive_reward_batch_async(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """异步批量计算认知奖励"""
        async with httpx.AsyncClient() as client:
            tasks = [
                self._compute_single_log_prob_async(q, cot, ans, client)
                for q, cot, ans in zip(questions, rewrite_cots, answers)
            ]
            return await asyncio.gather(*tasks)

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """同步批量计算认知奖励"""
        return asyncio.run(
            self.compute_cognitive_reward_batch_async(questions, rewrite_cots, answers)
        )

    # ==================== FSF 奖励 ====================

    def _parse_fsf(self, text: str) -> float:
        """解析 FSF 输出"""
        try:
            match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if match:
                return float(match.group(1))
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return min(float(match.group(1)), 1.0)
        except:
            pass
        return 0.5

    async def _compute_single_fsf_async(
        self,
        reasoning: str,
        client: httpx.AsyncClient,
    ) -> float:
        """异步计算单个 FSF"""
        prompt = self.FSF_PROMPT_TEMPLATE.format(reasoning=reasoning[:2000])

        # 使用 FSF LoRA（如果配置了）
        model = self.fsf_lora_name if self.fsf_lora_name else self.model_name

        async with self.semaphore:
            try:
                response = await client.post(
                    self.chat_url,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                        "temperature": 0.1,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                text = result["choices"][0]["message"]["content"]
                return self._parse_fsf(text)

            except Exception as e:
                print(f"[VLLMStudentReward] FSF 计算失败: {e}")
                return 0.5

    async def compute_fsf_batch_async(self, reasonings: List[str]) -> List[float]:
        """异步批量计算 FSF"""
        async with httpx.AsyncClient() as client:
            tasks = [self._compute_single_fsf_async(r, client) for r in reasonings]
            return await asyncio.gather(*tasks)

    def compute_fsf_batch(self, reasonings: List[str]) -> List[float]:
        """同步批量计算 FSF"""
        return asyncio.run(self.compute_fsf_batch_async(reasonings))

    # ==================== 组合奖励 ====================

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> Dict[str, Any]:
        """
        计算组合奖励

        Returns:
            {
                "cognitive_rewards": List[float],
                "logic_rewards": List[float],
                "total_rewards": List[float],
            }
        """
        # 认知奖励
        r_cog = self.compute_cognitive_reward_batch(questions, rewrite_cots, answers)

        # 逻辑奖励
        if use_logic_reward and original_cots and len(original_cots[0]) > 0:
            original_fsfs = self.compute_fsf_batch(original_cots)
            rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)
            r_logic = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]
        else:
            r_logic = [0.0] * len(questions)

        # 组合
        total = [c + lambda_logic * l for c, l in zip(r_cog, r_logic)]

        return {
            "cognitive_rewards": r_cog,
            "logic_rewards": r_logic,
            "total_rewards": total,
        }

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vllm_base_url}/models",
                    timeout=5.0,
                )
                return response.status_code == 200
        except:
            return False


# ==================== 全局单例 ====================

_vllm_client: Optional[VLLMStudentRewardClient] = None


def get_vllm_client(
    vllm_base_url: str = "http://localhost:8100/v1",
    **kwargs
) -> VLLMStudentRewardClient:
    """获取 vLLM 客户端单例"""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMStudentRewardClient(vllm_base_url=vllm_base_url, **kwargs)
    return _vllm_client


# ==================== VERL 接口 ====================

def compute_vllm_student_reward(
    data: DataProto,
    tokenizer: Any,
    vllm_base_url: str = "http://localhost:8100/v1",
    model_name: str = "Qwen2.5-3B-Instruct",
    fsf_lora_name: Optional[str] = "fsf_lora",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    VERL 奖励函数接口（使用 vLLM 服务）

    Args:
        data: Verl DataProto
        tokenizer: 分词器
        vllm_base_url: vLLM 服务地址
        model_name: 基础模型名称
        fsf_lora_name: FSF LoRA 名称（需要 vLLM 启用 --enable-lora）
        lambda_logic: 逻辑奖励权重
        use_logic_reward: 是否使用逻辑奖励

    Returns:
        {"reward_tensor": Tensor, "reward_extra_info": dict}
    """
    batch_size = data.batch.batch_size[0]

    # 解码
    prompts = tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
    rewrite_cots = tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    # 提取信息
    answers, original_cots, questions = [], [], []

    for i in range(batch_size):
        reward_info = data.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_info, list):
            reward_info = reward_info[i] if i < len(reward_info) else {}

        answers.append(reward_info.get("ground_truth", ""))

        orig_cot = ""
        if "original_cot" in data.non_tensor_batch:
            orig_cots_list = data.non_tensor_batch["original_cot"]
            if isinstance(orig_cots_list, list) and i < len(orig_cots_list):
                orig_cot = orig_cots_list[i]
        original_cots.append(orig_cot)

        questions.append(prompts[i])

    # 获取 vLLM 客户端
    client = get_vllm_client(
        vllm_base_url=vllm_base_url,
        model_name=model_name,
        fsf_lora_name=fsf_lora_name if use_logic_reward else None,
    )

    # 计算奖励
    result = client.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
    )

    # 转为 Tensor
    total_rewards = torch.tensor(
        result["total_rewards"],
        device="cuda",
        dtype=torch.bfloat16
    )
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0).clamp(-100, 100)

    # 转为 token-level
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size, response_length,
        device=total_rewards.device,
        dtype=total_rewards.dtype
    )

    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full((batch_size,), response_length - 1, dtype=torch.long)

    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": result,
    }
