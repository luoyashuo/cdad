"""
Combined Reward Function with Load/Offload Support

这个版本使用 load/offload 机制来动态加载和卸载 reward models，
避免长期占用 GPU 显存，适合与 GRPO 训练共享 GPU 资源。

Key Features:
- 模型初始化时保存在 CPU
- 计算 reward 时临时加载到 GPU
- 计算完成后自动卸载回 CPU
- 支持顺序加载和卸载多个模型（Student Model + FSF Judge）
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
from src.reward.student_reward_with_offload import HFStudentRewardClientWithOffload
from src.reward.fsf_reward_with_offload import HFFSFJudgeWithOffload


class CombinedRewardComputerWithOffload:
    """
    Combined Reward Computer with Load/Offload Support

    使用 load/offload 机制动态管理 reward models：
    1. 初始化时，所有模型保存在 CPU
    2. 计算 cognitive reward 时，临时加载 Student Model 到 GPU
    3. 计算 logic reward 时，临时加载 FSF Judge 到 GPU
    4. 每次计算完成后，自动卸载模型回 CPU
    """

    def __init__(
        self,
        student_model_path: str = "./models/Qwen3-1.7B",
        fsf_model_path: Optional[str] = None,
        device: str = "cuda:0",
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
        allow_cpu: bool = True,
        enable_offload: bool = True,
        offload_device: str = "cpu",
    ):
        """
        Initialize Combined Reward Computer with Offload Support.

        Args:
            student_model_path: Path to Student Model
            fsf_model_path: Path to FSF Judge Model (only required if use_logic_reward=True)
            device: GPU device to use for reward computation
            lambda_logic: Weight for logic reward
            use_logic_reward: Whether to use logic reward
            allow_cpu: If True, allow fallback to CPU when CUDA is not available
            enable_offload: If True, enable load/offload mechanism for all models
            offload_device: Device to offload models when not in use (default: "cpu")
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward
        self.enable_offload = enable_offload

        print(f"[CombinedRewardComputer] Initializing with offload support")
        print(f"  Device: {device}")
        print(f"  Offload Enabled: {enable_offload}")
        print(f"  Offload Device: {offload_device}")

        # Initialize Student Reward client with offload support
        self.student_client = HFStudentRewardClientWithOffload(
            model_path=student_model_path,
            device=device,
            allow_cpu=allow_cpu,
            enable_offload=enable_offload,
            offload_device=offload_device,
        )

        # Initialize FSF Judge client (if needed) with offload support
        if use_logic_reward:
            if not fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")
            self.fsf_judge = HFFSFJudgeWithOffload(
                model_path=fsf_model_path,
                device=device,
                allow_cpu=allow_cpu,
                enable_offload=enable_offload,
                offload_device=offload_device,
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
        Compute combined reward with automatic load/offload.

        执行流程（enable_offload=True）：
        1. 加载 Student Model 到 GPU → 计算 cognitive reward → 卸载到 CPU
        2. 加载 FSF Judge 到 GPU → 计算 logic reward → 卸载到 CPU
        3. 组合 rewards 并返回结果

        这样可以确保任何时刻最多只有一个 reward model 占用 GPU。

        Args:
            questions: List of questions
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings (Actor 生成)
            answers: List of ground truth answers

        Returns:
            Dictionary with reward info and extra metrics
        """
        batch_size = len(questions)

        # Step 1: Compute cognitive reward (auto load/offload)
        print(f"[CombinedRewardComputer] Computing cognitive rewards...")
        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,
        )

        r_cog = cog_result["rewards"]
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]

        # Step 2: Compute logic reward (auto load/offload) if enabled
        if (
            self.use_logic_reward
            and self.fsf_judge
            and original_cots
            and all(c for c in original_cots)
        ):
            print(f"[CombinedRewardComputer] Computing logic rewards...")
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

        # Step 3: Combine rewards
        total_rewards = [c + self.lambda_logic * l for c, l in zip(r_cog, r_logic)]

        print(f"[CombinedRewardComputer] Reward computation complete")

        return {
            "cognitive_rewards": r_cog,
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "logic_rewards": r_logic,
            "total_rewards": total_rewards,
            "token_log_probs": token_log_probs,
            "token_ids": token_ids,
            "extra_info": {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "lambda_logic": self.lambda_logic,
            },
        }


# Global singleton
_reward_computer: Optional[CombinedRewardComputerWithOffload] = None


def get_reward_computer_with_offload(**kwargs) -> CombinedRewardComputerWithOffload:
    """Get or create Combined Reward Computer singleton with offload support."""
    global _reward_computer
    if _reward_computer is None:
        # Set default allow_cpu to True for Ray worker compatibility
        if "allow_cpu" not in kwargs:
            kwargs["allow_cpu"] = True
        # Enable offload by default
        if "enable_offload" not in kwargs:
            kwargs["enable_offload"] = True
        _reward_computer = CombinedRewardComputerWithOffload(**kwargs)
    return _reward_computer


def compute_cot_rewriter_reward_with_offload(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen3-1.7B",
    fsf_model_path: Optional[str] = None,
    device: str = "cuda:0",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    allow_cpu: bool = True,
    enable_offload: bool = True,
    offload_device: str = "cpu",
    field_mapping: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    VERL-compatible reward function with load/offload support.

    This function is called by VERL's reward manager during training.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        student_model_path: Path to Student Model
        fsf_model_path: Path to FSF Judge Model
        device: GPU device to use for reward computation (e.g., "cuda:0")
        lambda_logic: Weight for logic reward
        use_logic_reward: Whether to use logic reward
        allow_cpu: If True, allow fallback to CPU when CUDA is not available
        enable_offload: If True, enable load/offload mechanism (recommended)
        offload_device: Device to offload models when not in use (default: "cpu")
        field_mapping: Optional dict to customize field names in data

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

    if field_mapping:
        default_mapping.update(field_mapping)

    prompt_key = default_mapping["prompt_key"]
    reward_model_key = default_mapping.get("reward_model_key", "reward_model")
    question_key = default_mapping["question_key"]
    original_cot_key = default_mapping["original_cot_key"]
    ground_truth_key = default_mapping["ground_truth_key"]

    use_flat_structure = not reward_model_key or reward_model_key == "null"

    batch_size = data.batch["responses"].shape[0]

    # Decode prompts and responses
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True,
    )
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True,
    )

    # Extract metadata
    questions = []
    answers = []
    original_cots = []

    for i in range(batch_size):
        if use_flat_structure:
            reward_info = {}
        else:
            reward_info = data.non_tensor_batch.get(reward_model_key, {})
            if isinstance(reward_info, list):
                reward_info = reward_info[i] if i < len(reward_info) else {}

        # Extract ground truth answer
        answer = reward_info.get(ground_truth_key, "")
        if not answer:
            answer = reward_info.get("ground_truth", "")
        if not answer:
            answer = reward_info.get("answer", "")
        if not answer and ground_truth_key in data.non_tensor_batch:
            gt_list = data.non_tensor_batch[ground_truth_key]
            if isinstance(gt_list, list) and i < len(gt_list):
                answer = gt_list[i]
        answers.append(answer)

        # Extract original CoT
        orig_cot = ""
        if original_cot_key in data.non_tensor_batch:
            orig_cots_list = data.non_tensor_batch[original_cot_key]
            if isinstance(orig_cots_list, list) and i < len(orig_cots_list):
                orig_cot = orig_cots_list[i]
        elif original_cot_key in reward_info:
            orig_cot = reward_info.get(original_cot_key, "")
        elif not orig_cot:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot)

        # Extract question
        question = reward_info.get(question_key, "")
        if not question:
            question = reward_info.get("question", "")
        if not question:
            question = prompts[i]
        questions.append(question)

    # Get or create reward computer with offload support
    reward_computer = get_reward_computer_with_offload(
        student_model_path=student_model_path,
        fsf_model_path=fsf_model_path,
        device=device,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
        allow_cpu=allow_cpu,
        enable_offload=enable_offload,
        offload_device=offload_device,
    )

    # Compute rewards (with automatic load/offload)
    result = reward_computer.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
    )

    # Convert to tensor
    device_tensor = data.batch["responses"].device
    total_rewards = torch.tensor(
        result["total_rewards"],
        device=device_tensor,
        dtype=torch.bfloat16,
    )

    total_rewards = torch.nan_to_num(total_rewards, nan=0.0)
    total_rewards = total_rewards.clamp(-100, 100)

    # Convert to token-level rewards
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size,
        response_length,
        device=device_tensor,
        dtype=total_rewards.dtype,
    )

    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full(
            (batch_size,),
            response_length - 1,
            dtype=torch.long,
            device=device_tensor,
        )

    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": result["cognitive_rewards"],
            "rewrite_mean_log_probs": result["rewrite_mean_log_probs"],
            "original_mean_log_probs": result["original_mean_log_probs"],
            "logic_rewards": result["logic_rewards"],
            "total_rewards": result["total_rewards"],
            "mean_cognitive_reward": sum(result["cognitive_rewards"])
            / len(result["cognitive_rewards"]),
            "mean_logic_reward": sum(result["logic_rewards"]) / len(result["logic_rewards"]),
            "mean_total_reward": sum(result["total_rewards"]) / len(result["total_rewards"]),
            **result["extra_info"],
        },
    }
