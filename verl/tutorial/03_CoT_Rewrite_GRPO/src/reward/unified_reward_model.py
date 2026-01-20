"""
统一奖励模型 (Unified Reward Model)

将 Student Model 和 FSF Analyzer 合并为一个 Reward Model Worker
两个任务共享同一个基座模型，通过不同的 prompt/adapter 实现不同功能

方案优势：
1. 减少显存占用 (共享基座)
2. 消除网络延迟 (本地推理)
3. 统一管理，简化部署
"""

from typing import Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

from verl import DataProto


class UnifiedRewardModel:
    """
    统一奖励模型

    使用同一个基座模型完成两个任务：
    1. 认知奖励: 计算 log P(Answer | Question, CoT)
    2. 逻辑奖励: 分析 FSF (Failed Step Fraction)

    显存优化策略：
    - 共享基座模型权重
    - 可选：使用不同的 LoRA adapter 切换任务
    """

    # FSF 分析 prompt (让模型做结构化输出)
    FSF_PROMPT_TEMPLATE = """<|im_start|>system
你是一个推理过程分析专家。请分析推理过程中的"失败分支"占比。
失败分支定义：尝试了某种方法但最终放弃的分支。
请输出 JSON 格式：{{"fsf": 0.0到1.0的数值}}<|im_end|>
<|im_start|>user
分析以下推理过程的失败分支占比：

{reasoning}

只输出 JSON，不要其他文字。<|im_end|>
<|im_start|>assistant
"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        use_lora_for_fsf: bool = False,
        fsf_lora_path: Optional[str] = None,
    ):
        """
        初始化统一奖励模型

        Args:
            model_path: 基座模型路径 (如 Qwen2.5-3B-Instruct)
            device: 设备
            dtype: 数据类型
            max_length: 最大序列长度
            use_lora_for_fsf: 是否为 FSF 任务使用单独的 LoRA
            fsf_lora_path: FSF LoRA adapter 路径
        """
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.use_lora_for_fsf = use_lora_for_fsf

        print(f"[UnifiedRewardModel] 加载基座模型: {model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 可选：加载 FSF LoRA adapter
        if use_lora_for_fsf and fsf_lora_path:
            from peft import PeftModel
            print(f"[UnifiedRewardModel] 加载 FSF LoRA: {fsf_lora_path}")
            self.fsf_model = PeftModel.from_pretrained(self.model, fsf_lora_path)
            self.fsf_model.eval()
        else:
            self.fsf_model = None

        print(f"[UnifiedRewardModel] 加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==================== 认知奖励 (R_cog) ====================

    def compute_cognitive_reward(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算认知奖励: log P(Answer | Question, CoT)

        Args:
            questions: 问题列表
            rewrite_cots: 改写后的 CoT 列表
            answers: 正确答案列表

        Returns:
            log_probs: shape (batch_size,)
        """
        batch_size = len(questions)

        # 构造输入
        full_texts = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: {ans}"
            for q, cot, ans in zip(questions, rewrite_cots, answers)
        ]

        prefixes = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: "
            for q, cot in zip(questions, rewrite_cots)
        ]

        # Tokenize
        prefix_enc = self.tokenizer(
            prefixes, padding=True, truncation=True,
            max_length=self.max_length - 100, return_tensors="pt"
        ).to(self.device)

        full_enc = self.tokenizer(
            full_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        prefix_lengths = prefix_enc.attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_enc.input_ids,
                attention_mask=full_enc.attention_mask,
            )
            logits = outputs.logits

        # 计算 answer 部分的 log prob
        log_probs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            start_idx = prefix_lengths[i].item()
            end_idx = full_enc.attention_mask[i].sum().item()

            if start_idx >= end_idx:
                log_probs[i] = -100.0
                continue

            ans_logits = logits[i, start_idx-1:end_idx-1, :]
            ans_labels = full_enc.input_ids[i, start_idx:end_idx]

            ans_log_probs = F.log_softmax(ans_logits, dim=-1)
            token_log_probs = ans_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
            log_probs[i] = token_log_probs.sum()

        return log_probs

    # ==================== 逻辑奖励 (R_logic) ====================

    def _parse_fsf(self, text: str) -> float:
        """解析 FSF 输出"""
        try:
            # 尝试提取 JSON
            match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if match:
                return float(match.group(1))

            # 尝试直接提取数字
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                val = float(match.group(1))
                return min(val, 1.0)  # 确保 <= 1.0

        except (ValueError, AttributeError):
            pass

        return 0.5  # 默认值

    def compute_fsf(self, reasonings: list[str]) -> list[float]:
        """
        计算 FSF (Failed Step Fraction)

        使用模型本身作为 Judge，分析推理过程中的失败分支占比

        Args:
            reasonings: 推理过程列表

        Returns:
            fsf_values: FSF 值列表 (0.0 - 1.0)
        """
        # 选择使用哪个模型
        model = self.fsf_model if self.fsf_model else self.model

        fsf_values = []

        # 可以批处理，但为了简单先逐个处理
        for reasoning in reasonings:
            prompt = self.FSF_PROMPT_TEMPLATE.format(reasoning=reasoning[:2000])  # 截断过长文本

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 100
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            fsf = self._parse_fsf(response)
            fsf_values.append(fsf)

        return fsf_values

    def compute_logic_reward(
        self,
        original_cots: list[str],
        rewrite_cots: list[str],
    ) -> torch.Tensor:
        """
        计算逻辑奖励

        R_logic = 1 if FSF(rewrite) < FSF(original) else 0

        Args:
            original_cots: 原始 CoT 列表
            rewrite_cots: 改写后 CoT 列表

        Returns:
            rewards: shape (batch_size,)
        """
        original_fsfs = self.compute_fsf(original_cots)
        rewrite_fsfs = self.compute_fsf(rewrite_cots)

        rewards = []
        for orig_fsf, rewrite_fsf in zip(original_fsfs, rewrite_fsfs):
            rewards.append(1.0 if rewrite_fsf < orig_fsf else 0.0)

        return torch.tensor(rewards, device=self.device, dtype=self.dtype)

    # ==================== 组合奖励 ====================

    def compute_combined_reward(
        self,
        questions: list[str],
        original_cots: list[str],
        rewrite_cots: list[str],
        answers: list[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算组合奖励: R = R_cog + λ * R_logic

        Args:
            questions: 问题列表
            original_cots: 原始 CoT 列表
            rewrite_cots: 改写后 CoT 列表
            answers: 答案列表
            lambda_logic: 逻辑奖励权重
            use_logic_reward: 是否使用逻辑奖励

        Returns:
            total_rewards: shape (batch_size,)
            info: 详细信息字典
        """
        # 计算认知奖励
        r_cog = self.compute_cognitive_reward(questions, rewrite_cots, answers)

        # 计算逻辑奖励 (可选)
        if use_logic_reward and original_cots and len(original_cots[0]) > 0:
            r_logic = self.compute_logic_reward(original_cots, rewrite_cots)
        else:
            r_logic = torch.zeros_like(r_cog)

        # 组合
        total = r_cog + lambda_logic * r_logic

        info = {
            "cognitive_rewards": r_cog.cpu().tolist(),
            "logic_rewards": r_logic.cpu().tolist(),
            "total_rewards": total.cpu().tolist(),
        }

        return total, info


# ==================== Verl 接口 ====================

# 全局单例
_unified_reward_model: Optional[UnifiedRewardModel] = None


def get_unified_reward_model(model_path: str, **kwargs) -> UnifiedRewardModel:
    """获取或创建 UnifiedRewardModel 单例"""
    global _unified_reward_model
    if _unified_reward_model is None:
        _unified_reward_model = UnifiedRewardModel(model_path, **kwargs)
    return _unified_reward_model


def compute_unified_reward(
    data: DataProto,
    tokenizer: Any,
    reward_model_path: str = "./models/Qwen2.5-3B-Instruct",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Verl 奖励函数接口

    使用统一奖励模型计算 R = R_cog + λ * R_logic

    Args:
        data: Verl DataProto
        tokenizer: 分词器 (用于解码 prompts/responses)
        reward_model_path: 奖励模型路径
        lambda_logic: 逻辑奖励权重
        use_logic_reward: 是否使用逻辑奖励

    Returns:
        {"reward_tensor": Tensor, "reward_extra_info": dict}
    """
    batch_size = data.batch.batch_size[0]

    # 解码
    prompts = tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
    rewrite_cots = tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    # 提取额外信息
    answers = []
    original_cots = []
    questions = []

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

    # 获取统一奖励模型
    reward_model = get_unified_reward_model(reward_model_path)

    # 计算奖励
    total_rewards, info = reward_model.compute_combined_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
    )

    # 处理异常值
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0).clamp(-100, 100)

    # 转为 token-level (outcome reward)
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
        "reward_extra_info": info,
    }
