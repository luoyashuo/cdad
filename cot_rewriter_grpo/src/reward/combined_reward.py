"""
Combined Reward Function for CoT Rewriter GRPO

Implements the composite reward: R = R_cog + lambda * R_logic

Where:
- R_cog (Cognitive Reward): log P_student(Answer | Question, Rewrite_CoT)
- R_logic (Logic Reward): 1 if FSF(Rewrite) < FSF(Original) else 0
- lambda: Weight for logic reward (default 0.5)

This module provides the VERL-compatible interface for reward computation.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import torch

# Add src to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verl import DataProto

from .student_reward import VLLMStudentRewardClient
from .fsf_reward import VLLMFSFJudge


class CombinedRewardComputer:
    """
    Combined Reward Computer

    Computes R = R_cog + lambda * R_logic using external vLLM services.
    """

    def __init__(
        self,
        student_vllm_url: str = "http://localhost:8100/v1",
        student_model_name: str = "Qwen2.5-3B-Instruct",
        fsf_vllm_url: str = "http://localhost:8101/v1",
        fsf_model_name: str = "Qwen2.5-7B-Instruct",
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
        max_concurrent: int = 32,
        timeout: float = 60.0,
    ):
        """
        Initialize Combined Reward Computer.

        Args:
            student_vllm_url: URL for Student Model vLLM service
            student_model_name: Name of Student Model
            fsf_vllm_url: URL for FSF Judge vLLM service
            fsf_model_name: Name of FSF Judge Model
            lambda_logic: Weight for logic reward
            use_logic_reward: Whether to use logic reward
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward

        # Initialize Student Reward client
        self.student_client = VLLMStudentRewardClient(
            vllm_base_url=student_vllm_url,
            model_name=student_model_name,
            timeout=timeout,
            max_concurrent=max_concurrent,
        )

        # Initialize FSF Judge client (if needed)
        if use_logic_reward:
            self.fsf_judge = VLLMFSFJudge(
                vllm_base_url=fsf_vllm_url,
                model_name=fsf_model_name,
                timeout=timeout,
                max_concurrent=max_concurrent,
            )
        else:
            self.fsf_judge = None

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> Dict[str, Any]:
        """
        Compute combined reward.

        Args:
            questions: List of questions
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings
            answers: List of ground truth answers

        Returns:
            Dictionary with:
                - cognitive_rewards: List of R_cog values
                - logic_rewards: List of R_logic values
                - total_rewards: List of R = R_cog + lambda * R_logic
                - extra_info: Additional information for logging
        """
        batch_size = len(questions)

        # Compute cognitive reward
        r_cog = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
        )

        # Compute logic reward (if enabled)
        if self.use_logic_reward and original_cots and all(c for c in original_cots):
            logic_result = self.fsf_judge.compute_logic_reward(
                original_cots=original_cots,
                rewrite_cots=rewrite_cots,
            )
            r_logic = logic_result["logic_rewards"]
            original_fsfs = logic_result["original_fsfs"]
            rewrite_fsfs = logic_result["rewrite_fsfs"]
        else:
            r_logic = [0.0] * batch_size
            original_fsfs = [0.0] * batch_size
            rewrite_fsfs = [0.0] * batch_size

        # Combine rewards: R = R_cog + lambda * R_logic
        total_rewards = [
            c + self.lambda_logic * l
            for c, l in zip(r_cog, r_logic)
        ]

        return {
            "cognitive_rewards": r_cog,
            "logic_rewards": r_logic,
            "total_rewards": total_rewards,
            "extra_info": {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "lambda_logic": self.lambda_logic,
            },
        }


# Global singleton
_reward_computer: Optional[CombinedRewardComputer] = None


def get_reward_computer(**kwargs) -> CombinedRewardComputer:
    """Get or create Combined Reward Computer singleton."""
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = CombinedRewardComputer(**kwargs)
    return _reward_computer


def compute_cot_rewriter_reward(
    data: DataProto,
    tokenizer: Any,
    student_vllm_url: str = "http://localhost:8100/v1",
    student_model_name: str = "Qwen2.5-3B-Instruct",
    fsf_vllm_url: str = "http://localhost:8101/v1",
    fsf_model_name: str = "Qwen2.5-7B-Instruct",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    max_concurrent: int = 32,
    timeout: float = 60.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    VERL-compatible reward function interface.

    This function is called by VERL's reward manager during training.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        student_vllm_url: URL for Student Model service
        student_model_name: Name of Student Model
        fsf_vllm_url: URL for FSF Judge service
        fsf_model_name: Name of FSF Judge Model
        lambda_logic: Weight for logic reward
        use_logic_reward: Whether to use logic reward
        max_concurrent: Maximum concurrent requests
        timeout: Request timeout

    Returns:
        Dictionary with:
            - reward_tensor: Token-level reward tensor [batch_size, seq_len]
            - reward_extra_info: Additional metrics for logging
    """
    batch_size = data.batch.batch_size[0]

    # Decode prompts and responses
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True,
    )
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True,
    )

    # Extract metadata from non_tensor_batch
    questions = []
    answers = []
    original_cots = []

    for i in range(batch_size):
        # Get reward info from non_tensor_batch
        reward_info = data.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_info, list):
            reward_info = reward_info[i] if i < len(reward_info) else {}

        # Extract ground truth answer
        answer = reward_info.get("ground_truth", "")
        if not answer:
            answer = reward_info.get("answer", "")
        answers.append(answer)

        # Extract original CoT
        orig_cot = ""
        if "original_cot" in data.non_tensor_batch:
            orig_cots_list = data.non_tensor_batch["original_cot"]
            if isinstance(orig_cots_list, list) and i < len(orig_cots_list):
                orig_cot = orig_cots_list[i]
        elif "original_cot" in reward_info:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot)

        # Extract question (from prompt or metadata)
        question = reward_info.get("question", prompts[i])
        questions.append(question)

    # Get or create reward computer
    reward_computer = get_reward_computer(
        student_vllm_url=student_vllm_url,
        student_model_name=student_model_name,
        fsf_vllm_url=fsf_vllm_url,
        fsf_model_name=fsf_model_name,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
        max_concurrent=max_concurrent,
        timeout=timeout,
    )

    # Compute rewards
    result = reward_computer.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
    )

    # Convert to tensor
    total_rewards = torch.tensor(
        result["total_rewards"],
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Handle NaN and extreme values
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0)
    total_rewards = total_rewards.clamp(-100, 100)

    # Convert to token-level rewards (assign reward to last token)
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size,
        response_length,
        device=total_rewards.device,
        dtype=total_rewards.dtype,
    )

    # Get actual sequence lengths from response mask
    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full(
            (batch_size,),
            response_length - 1,
            dtype=torch.long,
        )

    # Assign reward to the last valid token of each sequence
    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": result["cognitive_rewards"],
            "logic_rewards": result["logic_rewards"],
            "total_rewards": result["total_rewards"],
            "mean_cognitive_reward": sum(result["cognitive_rewards"]) / len(result["cognitive_rewards"]),
            "mean_logic_reward": sum(result["logic_rewards"]) / len(result["logic_rewards"]),
            "mean_total_reward": sum(result["total_rewards"]) / len(result["total_rewards"]),
            **result["extra_info"],
        },
    }
