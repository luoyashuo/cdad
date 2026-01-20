"""
认知奖励 (Cognitive Reward) 计算模块

R_cog = log P_student(Answer | Question, Rewrite_CoT)

核心思想: 改写后的 CoT 应该让 Student Model 更容易预测正确答案
"""

from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CognitiveRewardComputer:
    """
    认知奖励计算器

    使用冻结的 Student Model 计算改写后 CoT 对答案预测的增益
    """

    def __init__(
        self,
        student_model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ):
        """
        初始化 Student Model

        Args:
            student_model_path: Student Model 路径
            device: 设备 (默认 CUDA)
            dtype: 数据类型 (默认 bfloat16)
            max_length: 最大序列长度
        """
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        # 加载 Student Model (冻结状态)
        print(f"[CognitiveReward] 加载 Student Model: {student_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[CognitiveReward] Student Model 加载完成, 显存占用: {self._get_memory_usage():.2f} GB")

    def _get_memory_usage(self) -> float:
        """获取当前 GPU 显存占用 (GB)"""
        return torch.cuda.memory_allocated() / 1e9

    def _build_student_input(self, question: str, rewrite_cot: str, answer: str) -> str:
        """
        构造 Student Model 输入

        格式:
        Question: {question}
        Reasoning: {rewrite_cot}
        Answer: {answer}
        """
        return f"""Question: {question}

Reasoning: {rewrite_cot}

Answer: {answer}"""

    def compute_answer_log_prob(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算 log P_student(Answer | Question, Rewrite_CoT)

        Args:
            questions: 问题列表
            rewrite_cots: 改写后的 CoT 列表
            answers: 正确答案列表

        Returns:
            log_probs: shape (batch_size,), 每个样本的 log 概率
        """
        batch_size = len(questions)

        # 构造完整文本和前缀
        full_texts = [
            self._build_student_input(q, cot, ans)
            for q, cot, ans in zip(questions, rewrite_cots, answers)
        ]

        prefixes = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: "
            for q, cot in zip(questions, rewrite_cots)
        ]

        # Tokenize
        prefix_encodings = self.tokenizer(
            prefixes,
            padding=True,
            truncation=True,
            max_length=self.max_length - 100,
            return_tensors="pt"
        ).to(self.device)

        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        prefix_lengths = prefix_encodings.attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_encodings.input_ids,
                attention_mask=full_encodings.attention_mask,
            )
            logits = outputs.logits

        # 计算 Answer 部分的 log 概率
        log_probs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            start_idx = prefix_lengths[i].item()
            end_idx = full_encodings.attention_mask[i].sum().item()

            if start_idx >= end_idx:
                log_probs[i] = -100.0
                continue

            answer_logits = logits[i, start_idx-1:end_idx-1, :]
            answer_labels = full_encodings.input_ids[i, start_idx:end_idx]

            answer_log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = answer_log_probs.gather(1, answer_labels.unsqueeze(1)).squeeze(1)
            log_probs[i] = token_log_probs.sum()

        return log_probs

    def __call__(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算认知奖励

        Args:
            questions: 问题列表
            rewrite_cots: 改写后的 CoT 列表
            answers: 正确答案列表

        Returns:
            rewards: shape (batch_size,), 归一化后的奖励值
        """
        log_probs = self.compute_answer_log_prob(questions, rewrite_cots, answers)
        return log_probs


# 全局单例
_cognitive_reward_computer: Optional[CognitiveRewardComputer] = None


def get_cognitive_reward_computer(student_model_path: str, **kwargs) -> CognitiveRewardComputer:
    """获取或创建 CognitiveRewardComputer 单例"""
    global _cognitive_reward_computer
    if _cognitive_reward_computer is None:
        _cognitive_reward_computer = CognitiveRewardComputer(student_model_path, **kwargs)
    return _cognitive_reward_computer
