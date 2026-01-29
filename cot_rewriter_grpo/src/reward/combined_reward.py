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

from src.reward.student_reward import HFStudentRewardClient
from src.reward.fsf_reward import HFFSFJudge


class CombinedRewardComputer:
    """
    Combined Reward Computer

    Computes R = R_cog + lambda * R_logic using HuggingFace Transformers directly.
    """

    def __init__(
        self,
        student_model_path: str = "./models/Qwen3-1.7B",
        fsf_model_path: Optional[str] = None,
        device: str = "cuda:3",
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ):
        """
        Initialize Combined Reward Computer.

        Args:
            student_model_path: Path to Student Model
            fsf_model_path: Path to FSF Judge Model (only required if use_logic_reward=True)
            device: Device to run reward models on (should be a single GPU)
            lambda_logic: Weight for logic reward
            use_logic_reward: Whether to use logic reward
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward

        # Initialize Student Reward client
        self.student_client = HFStudentRewardClient(
            model_path=student_model_path,
            device=device,
        )

        # Initialize FSF Judge client (if needed)
        if use_logic_reward:
            if not fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")
            self.fsf_judge = HFFSFJudge(
                model_path=fsf_model_path,
                device=device,
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
        if (
            self.use_logic_reward
            and self.fsf_judge
            and original_cots
            and all(c for c in original_cots)
        ):
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
        total_rewards = [c + self.lambda_logic * l for c, l in zip(r_cog, r_logic)]

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
    student_model_path: str = "./models/Qwen3-1.7B",
    fsf_model_path: Optional[str] = None,
    device: str = "cuda:3",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    field_mapping: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    VERL-compatible reward function interface.

    This function is called by VERL's reward manager during training.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        student_model_path: Path to Student Model
        fsf_model_path: Path to FSF Judge Model
        device: Device to run reward models on
        lambda_logic: Weight for logic reward
        use_logic_reward: Whether to use logic reward
        field_mapping: Optional dict to customize field names in data.
            Keys: "prompt_key", "reward_model_key", "question_key",
                  "original_cot_key", "ground_truth_key"
            Example: {"prompt_key": "problem", "ground_truth_key": "solution"}

    Returns:
        Dictionary with:
            - reward_tensor: Token-level reward tensor [batch_size, seq_len]
            - reward_extra_info: Additional metrics for logging
    """
    # Default field mapping
    default_mapping = {
        "prompt_key": "prompt",
        "reward_model_key": "reward_model",
        "question_key": "question",
        "original_cot_key": "original_cot",
        "ground_truth_key": "ground_truth",
    }

    # Merge with user-provided mapping
    if field_mapping:
        default_mapping.update(field_mapping)

    prompt_key = default_mapping["prompt_key"]
    reward_model_key = default_mapping.get("reward_model_key", "reward_model")
    question_key = default_mapping["question_key"]
    original_cot_key = default_mapping["original_cot_key"]
    ground_truth_key = default_mapping["ground_truth_key"]

    # Handle case where reward_model_key is None or empty (flat structure)
    use_flat_structure = not reward_model_key or reward_model_key == "null"

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
        if use_flat_structure:
            # Use flat structure: read directly from non_tensor_batch
            reward_info = {}
        else:
            # Use nested structure: read from reward_model_key
            reward_info = data.non_tensor_batch.get(reward_model_key, {})
            if isinstance(reward_info, list):
                reward_info = reward_info[i] if i < len(reward_info) else {}

        # Extract ground truth answer (try mapped key first, then fallbacks)
        answer = reward_info.get(ground_truth_key, "")
        if not answer:
            answer = reward_info.get("ground_truth", "")
        if not answer:
            answer = reward_info.get("answer", "")
        # Also check top-level non_tensor_batch
        if not answer and ground_truth_key in data.non_tensor_batch:
            gt_list = data.non_tensor_batch[ground_truth_key]
            if isinstance(gt_list, list) and i < len(gt_list):
                answer = gt_list[i]
        answers.append(answer)

        # Extract original CoT
        orig_cot = ""
        # Check top-level non_tensor_batch first
        if original_cot_key in data.non_tensor_batch:
            orig_cots_list = data.non_tensor_batch[original_cot_key]
            if isinstance(orig_cots_list, list) and i < len(orig_cots_list):
                orig_cot = orig_cots_list[i]
        # Then check reward_info
        elif original_cot_key in reward_info:
            orig_cot = reward_info.get(original_cot_key, "")
        # Fallback to default keys
        elif not orig_cot:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot)

        # Extract question (from prompt or metadata)
        question = reward_info.get(question_key, "")
        if not question:
            question = reward_info.get("question", "")
        if not question:
            question = prompts[i]
        questions.append(question)

    # Get or create reward computer
    reward_computer = get_reward_computer(
        student_model_path=student_model_path,
        fsf_model_path=fsf_model_path,
        device=device,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
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
            "mean_cognitive_reward": sum(result["cognitive_rewards"])
            / len(result["cognitive_rewards"]),
            "mean_logic_reward": sum(result["logic_rewards"]) / len(result["logic_rewards"]),
            "mean_total_reward": sum(result["total_rewards"]) / len(result["total_rewards"]),
            **result["extra_info"],
        },
    }
