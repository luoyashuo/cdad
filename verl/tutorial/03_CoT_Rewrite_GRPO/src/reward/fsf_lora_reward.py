"""
FSF LoRA 奖励模型

使用 Student Model + FSF LoRA Adapter 的方式计算奖励：
- 认知奖励 (R_cog): 使用原始 Student Model（不带adapter）
- FSF 奖励 (R_logic): 使用 Student Model + FSF LoRA Adapter

优势：
1. 不需要额外部署 FSF_JUDGE 服务
2. 共享 Student Model 基座，节省显存
3. 推理时动态切换 adapter
"""

from typing import Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
import json
import re

from verl import DataProto


class FSFLoRARewardModel:
    """
    FSF LoRA 奖励模型

    使用同一个 Student Model 基座完成两个任务：
    1. 认知奖励: 使用原始模型计算 log P(Answer | Question, CoT)
    2. FSF 奖励: 使用模型 + FSF LoRA 分析失败分支比例
    """

    # FSF 分析 prompt
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
        student_model_path: str,
        fsf_lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ):
        """
        初始化 FSF LoRA 奖励模型

        Args:
            student_model_path: Student Model 基座路径
            fsf_lora_path: FSF LoRA adapter 路径（可选，如果没有则跳过FSF计算）
            device: 设备
            dtype: 数据类型
            max_length: 最大序列长度
        """
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.fsf_lora_path = fsf_lora_path

        print(f"[FSFLoRARewardModel] 加载 Student Model 基座: {student_model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型（用于认知奖励）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.base_model.to(device)
        self.base_model.eval()

        # 冻结基座模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 加载 FSF LoRA adapter（如果提供）
        self.fsf_model = None
        if fsf_lora_path:
            print(f"[FSFLoRARewardModel] 加载 FSF LoRA adapter: {fsf_lora_path}")
            self.fsf_model = PeftModel.from_pretrained(
                self.base_model,
                fsf_lora_path,
                is_trainable=False,  # 推理模式
            )
            self.fsf_model.eval()

        print(f"[FSFLoRARewardModel] 加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==================== 认知奖励 (使用原始 Student Model) ====================

    def compute_cognitive_reward(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算认知奖励: log P(Answer | Question, CoT)

        使用原始 Student Model（不带 LoRA adapter）

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

        # 使用基座模型（不带 LoRA）
        with torch.no_grad():
            outputs = self.base_model(
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

    # ==================== FSF 奖励 (使用 Student Model + LoRA) ====================

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
                return min(val, 1.0)

        except (ValueError, AttributeError):
            pass

        return 0.5  # 默认值

    def compute_fsf_batch(self, reasonings: list[str]) -> list[float]:
        """
        批量计算 FSF (Failed Step Fraction)

        使用 Student Model + FSF LoRA adapter

        Args:
            reasonings: 推理过程列表

        Returns:
            fsf_values: FSF 值列表 (0.0 - 1.0)
        """
        if self.fsf_model is None:
            print("[FSFLoRARewardModel] 警告: FSF LoRA 未加载，返回默认值 0.5")
            return [0.5] * len(reasonings)

        fsf_values = []

        for reasoning in reasonings:
            prompt = self.FSF_PROMPT_TEMPLATE.format(reasoning=reasoning[:2000])

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - 100
            ).to(self.device)

            # 使用 FSF LoRA 模型
            with torch.no_grad():
                outputs = self.fsf_model.generate(
                    **inputs,
                    max_new_tokens=100,
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
        original_fsfs = self.compute_fsf_batch(original_cots)
        rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

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
        # 计算认知奖励（使用原始 Student Model）
        r_cog = self.compute_cognitive_reward(questions, rewrite_cots, answers)

        # 计算逻辑奖励（使用 Student Model + FSF LoRA）
        if use_logic_reward and self.fsf_model is not None and original_cots and len(original_cots[0]) > 0:
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


# ==================== 全局单例 ====================

_fsf_lora_reward_model: Optional[FSFLoRARewardModel] = None


def get_fsf_lora_reward_model(
    student_model_path: str,
    fsf_lora_path: Optional[str] = None,
    **kwargs
) -> FSFLoRARewardModel:
    """获取或创建 FSFLoRARewardModel 单例"""
    global _fsf_lora_reward_model
    if _fsf_lora_reward_model is None:
        _fsf_lora_reward_model = FSFLoRARewardModel(
            student_model_path,
            fsf_lora_path=fsf_lora_path,
            **kwargs
        )
    return _fsf_lora_reward_model


# ==================== Verl 接口 ====================

def compute_fsf_lora_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    fsf_lora_path: Optional[str] = None,
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Verl 奖励函数接口

    使用 FSF LoRA 奖励模型计算 R = R_cog + λ * R_logic

    Args:
        data: Verl DataProto
        tokenizer: 分词器 (用于解码 prompts/responses)
        student_model_path: Student Model 路径
        fsf_lora_path: FSF LoRA adapter 路径
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

    # 获取 FSF LoRA 奖励模型
    reward_model = get_fsf_lora_reward_model(student_model_path, fsf_lora_path)

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
