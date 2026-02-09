"""
Combined Reward Function for CoT Rewriter GRPO

Reward 计算逻辑:
- 使用 Qwen3 chat template 格式计算 Student Model 的 log prob
- answer_log_prob(Q, CoT, A) = avg(token_log_probs) + alpha_cog * min(token_log_probs)
- R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
- alpha_cog 为可配置参数，默认为 0.01
- (可选) R_logic: FSF Judge 比较失败分支比例

差值公式的含义:
- 如果 rewrite 后答案的 log prob 更高，说明 rewrite 质量更好，reward 为正
- 如果 rewrite 后答案的 log prob 更低，说明 rewrite 质量更差，reward 为负

This module provides the VERL-compatible interface for reward computation.

Supports two backends:
- sglang: Uses SGLang Engine for both Student and FSF Judge (recommended)
- vllm: Uses vLLM for Student and HuggingFace for FSF Judge
"""

from typing import Any, Dict, List, Optional

import torch

from verl import DataProto

from verl.experimental.reward.cot_rewriter.training_logger import get_logger

# Training logger for reward computation
logger = get_logger("reward")


class CombinedRewardComputer:
    """
    Combined Reward Computer

    Computes R = R_cog + lambda * R_logic using Student Model.

    Supports two backends:
    - sglang: Uses SGLang Engine for both Student and FSF Judge (recommended)
    - vllm: Uses vLLM for Student and HuggingFace for FSF Judge
    """

    def __init__(
        self,
        student_model_path: str = "./models/Qwen3-1.7B",
        fsf_model_path: Optional[str] = None,
        device: str = "cuda:3",
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
        alpha_cog: float = 0.01,
        student_devices: Optional[List[str]] = None,
        reward_micro_batch_size: Optional[int] = None,
        gpu_batch_size: int = 8,
        vllm_gpu_memory_utilization: float = 0.3,
        reward_backend: str = "vllm",  # "sglang" or "vllm"
        sglang_mem_fraction_static: float = 0.85,  # SGLang: mem_fraction_static
        student_max_model_len: int = 8192,  # max model len for Student Model
        student_micro_batch_size: int = 4,  # micro batch size for Student Model
        student_max_num_seqs: int = 1,  # limit concurrent sequences (vLLM only)
        student_tensor_parallel_size: int = 1,  # TP size for Student Model (1=disable)
        student_logprob_last_n_tokens: Optional[int] = None,  # SGLang: only compute last N tokens
        **kwargs,  # Ignore deprecated params
    ):
        """
        Initialize Combined Reward Computer.

        Args:
            student_model_path: Path to Student Model
            fsf_model_path: Path to FSF Judge Model (only required if use_logic_reward=True)
            device: Device to run FSF Judge on (Student uses student_devices)
            lambda_logic: Weight for logic reward
            use_logic_reward: Whether to use logic reward
            alpha_cog: Weight for min log prob term in cognitive reward (default: 0.01)
            student_devices: List of devices for Student (e.g., ["cuda:0", "cuda:1"])
            reward_micro_batch_size: If set, process reward computation in micro-batches to avoid OOM.
            gpu_batch_size: Batch size for GPU forward pass (default: 8).
            vllm_gpu_memory_utilization: GPU memory fraction for vLLM per device (default: 0.3)
            reward_backend: Backend to use - "sglang" or "vllm" (default: "vllm")
            sglang_mem_fraction_static: GPU memory fraction for SGLang static allocation (default: 0.85)
            student_max_model_len: Maximum sequence length for Student Model
            student_micro_batch_size: Micro batch size for Student Model inference
            student_max_num_seqs: Maximum concurrent sequences in vLLM (lower saves memory, vLLM only)
            student_tensor_parallel_size: Tensor parallelism size for Student Model (default: 1)
            student_logprob_last_n_tokens: SGLang only - only compute last N tokens' logprobs to save memory
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward
        self.alpha_cog = alpha_cog
        self.reward_micro_batch_size = reward_micro_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.reward_backend = reward_backend

        # Initialize Student Reward client based on backend
        if reward_backend == "sglang":
            from verl.experimental.reward.cot_rewriter.sglang_student_reward import SGLangStudentRewardClient
            self.student_client = SGLangStudentRewardClient(
                model_path=student_model_path,
                tensor_parallel_size=student_tensor_parallel_size,
                mem_fraction_static=sglang_mem_fraction_static,
                max_model_len=student_max_model_len,
                devices=student_devices,
                micro_batch_size=student_micro_batch_size,
                logprob_last_n_tokens=student_logprob_last_n_tokens,
            )
        else:
            from verl.experimental.reward.cot_rewriter.student_reward import VLLMStudentRewardClient
            self.student_client = VLLMStudentRewardClient(
                model_path=student_model_path,
                tensor_parallel_size=student_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=student_max_model_len,
                devices=student_devices,
                micro_batch_size=student_micro_batch_size,
                max_num_seqs=student_max_num_seqs,
            )

        # Initialize FSF Judge client (if needed)
        if use_logic_reward:
            if not fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")

            if reward_backend == "sglang":
                from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFJudge
                self.fsf_judge = SGLangFSFJudge(
                    model_path=fsf_model_path,
                    device=device,
                    gpu_memory_utilization=sglang_gpu_memory_utilization,
                )
            else:
                from verl.experimental.reward.cot_rewriter.fsf_reward import HFFSFJudge
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

        Reward 逻辑:
        - answer_log_prob(Q, CoT, A) = avg(token_log_probs) + alpha_cog × min(token_log_probs)
        - R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
        - (可选) R_logic = FSF based reward
        - Total = R_cog + lambda * R_logic

        Args:
            questions: List of questions
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings (Actor 生成)
            answers: List of ground truth answers

        Returns:
            Dictionary with reward info and extra metrics
        """
        batch_size = len(questions)
        
        # Log per-sample prompts before reward computation
        logger.log_per_sample_rewards(
            questions=questions,
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            rewards=[0.0] * batch_size,  # Will be updated after computation
            prefix="[REWARD_PRE]",
            current_idx=0,
            total=batch_size,
            max_text_len=100,
            max_samples=2
        )

        # Compute cognitive reward
        # R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
        # 其中 answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)
        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,
            alpha_cog=self.alpha_cog,
        )

        # 获取计算结果
        r_cog = cog_result["rewards"]
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]

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
        
        # Log per-sample prompts with computed rewards
        logger.log_per_sample_rewards(
            questions=questions,
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            rewards=total_rewards,
            prefix="[REWARD_POST]",
            current_idx=0,
            total=batch_size,
            max_text_len=100,
            max_samples=2
        )

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
_reward_computer: Optional[CombinedRewardComputer] = None


def get_reward_computer(**kwargs) -> CombinedRewardComputer:
    """Get or create Combined Reward Computer singleton."""
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = CombinedRewardComputer(**kwargs)
    return _reward_computer


def cleanup_reward_computer():
    """Cleanup the global reward computer to release GPU memory."""
    global _reward_computer
    if _reward_computer is not None:
        logger.info("[REWARD] Cleaning up reward computer...")
        if hasattr(_reward_computer, 'student_client') and _reward_computer.student_client is not None:
            _reward_computer.student_client.cleanup()
        if hasattr(_reward_computer, 'fsf_judge') and _reward_computer.fsf_judge is not None:
            # FSF Judge cleanup if needed
            del _reward_computer.fsf_judge
        _reward_computer = None

        # Force garbage collection
        import gc
        import torch
        import time
        import subprocess
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for GPU memory to be released by killed Ray actors
        logger.info("[REWARD] Waiting for GPU memory to be released...")
        time.sleep(3)  # Give time for Ray actors to fully release GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Log GPU memory status after cleanup
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            logger.info("[REWARD] nvidia-smi AFTER Student vLLM cleanup:")
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    gpu_idx = parts[0].strip()
                    used_mb = int(parts[1].strip())
                    total_mb = int(parts[2].strip())
                    used_gb = used_mb / 1024
                    total_gb = total_mb / 1024
                    logger.info(f"[REWARD]   GPU {gpu_idx}: {used_gb:.1f}GB / {total_gb:.1f}GB ({used_mb}MB used)")
        except Exception as e:
            logger.warning(f"[REWARD] nvidia-smi error: {e}")

        logger.info("[REWARD] Reward computer cleanup completed")


def compute_cot_rewriter_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen3-1.7B",
    fsf_model_path: Optional[str] = None,
    device: str = "cuda:5",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    alpha_cog: float = 0.01,
    alpha: Optional[float] = None,  # Alias for alpha_cog (from config)
    field_mapping: Optional[Dict[str, str]] = None,
    student_devices: Optional[List[str]] = None,
    reward_micro_batch_size: Optional[int] = None,
    gpu_batch_size: int = 8,
    vllm_gpu_memory_utilization: float = 0.3,
    cleanup_after_compute: bool = True,
    reward_backend: str = "vllm",  # "sglang" or "vllm"
    sglang_mem_fraction_static: float = 0.85,  # SGLang: mem_fraction_static
    student_max_model_len: int = 8192,  # max model len for Student Model
    student_micro_batch_size: int = 4,  # micro batch size for Student Model
    student_max_num_seqs: int = 1,  # limit concurrent sequences in vLLM
    student_tensor_parallel_size: int = 1,  # TP size for Student Model
    student_logprob_last_n_tokens: Optional[int] = None,  # SGLang: only compute last N tokens
    truncation_penalty: float = 0.5,  # Penalty for truncated responses (configured as positive)
    format_penalty: float = 0.5,  # Penalty for format violations (configured as positive)
    **kwargs,
) -> Dict[str, Any]:
    # Use alpha if provided (from config), otherwise use alpha_cog
    if alpha is not None:
        alpha_cog = alpha
    """
    VERL-compatible reward function interface.

    This function is called by VERL's reward manager during training.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        student_model_path: Path to Student Model
        fsf_model_path: Path to FSF Judge Model
        device: Device to run FSF Judge on. Default is cuda:5.
        lambda_logic: Weight for logic reward
        use_logic_reward: Whether to use logic reward
        alpha_cog: Weight for min log prob term in cognitive reward (default: 0.01)
        field_mapping: Optional dict to customize field names in data.
            Keys: "prompt_key", "reward_model_key", "question_key",
                  "original_cot_key", "ground_truth_key"
        student_devices: List of CUDA devices for Student (e.g., ["cuda:0", "cuda:1"])
        reward_micro_batch_size: Micro-batch size for reward computation
        gpu_batch_size: Batch size for GPU forward pass
        vllm_gpu_memory_utilization: GPU memory fraction for vLLM per device
        cleanup_after_compute: Whether to cleanup Student after reward computation
            to release GPU memory for Actor reload. Default is True.
        reward_backend: Backend to use - "sglang" or "vllm" (default: "vllm")
        sglang_mem_fraction_static: GPU memory fraction for SGLang static allocation
        student_max_model_len: Maximum sequence length for Student Model
        student_micro_batch_size: Micro batch size for Student Model inference
        student_max_num_seqs: Maximum concurrent sequences in vLLM (vLLM only)
        student_tensor_parallel_size: Tensor parallelism size for Student Model
        student_logprob_last_n_tokens: SGLang only - only compute last N tokens' logprobs

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

    # Extract metadata from non_tensor_batch
    questions = []
    answers = []
    original_cots = []

    for i in range(batch_size):
        # Get reward info from non_tensor_batch
        if use_flat_structure:
            reward_info = {}
        else:
            reward_info_data = data.non_tensor_batch.get(reward_model_key, {})
            if hasattr(reward_info_data, '__len__') and not isinstance(reward_info_data, dict):
                reward_info = reward_info_data[i] if i < len(reward_info_data) else {}
            else:
                reward_info = reward_info_data
            if not isinstance(reward_info, dict):
                reward_info = {}

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
            orig_cots_data = data.non_tensor_batch[original_cot_key]
            if hasattr(orig_cots_data, '__len__') and i < len(orig_cots_data):
                orig_cot = orig_cots_data[i]
                if hasattr(orig_cot, 'item'):
                    orig_cot = orig_cot.item()
        if not orig_cot and original_cot_key in reward_info:
            orig_cot = reward_info.get(original_cot_key, "")
        if not orig_cot:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot if orig_cot else "")

        # Extract question
        question = reward_info.get(question_key, "")
        if not question:
            question = reward_info.get("question", "")
        if not question:
            question = prompts[i]
        questions.append(question)

    print_samples = kwargs.get("print_samples", 2)

    # Compute rewards for ALL samples (no truncation filtering)
    reward_computer = get_reward_computer(
        student_model_path=student_model_path,
        fsf_model_path=fsf_model_path,
        device=device,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
        alpha_cog=alpha_cog,
        student_devices=student_devices,
        reward_micro_batch_size=reward_micro_batch_size,
        gpu_batch_size=gpu_batch_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        reward_backend=reward_backend,
        sglang_mem_fraction_static=sglang_mem_fraction_static,
        student_max_model_len=student_max_model_len,
        student_micro_batch_size=student_micro_batch_size,
        student_max_num_seqs=student_max_num_seqs,
        student_tensor_parallel_size=student_tensor_parallel_size,
        student_logprob_last_n_tokens=student_logprob_last_n_tokens,
    )

    # Compute rewards for all samples
    result = reward_computer.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
    )

    # Convert to tensor
    device = data.batch["responses"].device
    total_rewards = torch.tensor(
        result["total_rewards"],
        device=device,
        dtype=torch.bfloat16,
    )

    # Apply penalties
    # 1. Truncation penalty: if rollout result was truncated (finish_reason == "length")
    # 2. Format penalty: if response (lowercased) contains "# question" or "# rewrit"
    finish_reasons = data.non_tensor_batch.get("finish_reasons", [])
    # Penalties are configured as positive values, applied as negative to rewards
    format_check_strings = ["# question", "# rewrit"]  # 检查这两个字符串（lower 后）

    penalties = [0.0] * batch_size
    truncation_penalties = [0.0] * batch_size  # Track truncation penalties separately for logging
    format_penalties = [0.0] * batch_size  # Track format penalties separately for logging
    truncated_flags = [0] * batch_size  # Track truncated flags for DAPO filter_groups

    for i in range(batch_size):
        # Truncation penalty (configured as positive, applied as negative)
        if i < len(finish_reasons) and finish_reasons[i] == "length":
            penalties[i] -= truncation_penalty
            truncation_penalties[i] = -truncation_penalty
            truncated_flags[i] = 1  # Mark as truncated for DAPO filter_groups
            logger.info(f"[PENALTY] Sample {i}: Truncated (finish_reason=length), penalty=-{truncation_penalty}")

        # Format penalty: check if full response (lowercased) contains forbidden strings
        if rewrite_cots and i < len(rewrite_cots):
            response_lower = rewrite_cots[i].lower()
            for check_string in format_check_strings:
                if check_string in response_lower:
                    penalties[i] -= format_penalty
                    format_penalties[i] = -format_penalty
                    logger.info(
                        f"[PENALTY] Sample {i}: Format violation ('{check_string}' found in response), "
                        f"response_preview='{rewrite_cots[i][:100]}...', penalty=-{format_penalty}"
                    )
                    break  # 只惩罚一次

    # Apply penalties to total_rewards
    total_rewards = total_rewards + torch.tensor(penalties, device=device, dtype=torch.bfloat16)

    # Handle NaN and extreme values
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0)
    total_rewards = total_rewards.clamp(-100, 100)

    # Convert to token-level rewards
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size,
        response_length,
        device=device,
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
            device=device,
        )

    # Assign reward to the last valid token of each sequence
    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    reward_result = {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": result["cognitive_rewards"],
            "rewrite_mean_log_probs": result["rewrite_mean_log_probs"],
            "original_mean_log_probs": result["original_mean_log_probs"],
            "logic_rewards": result["logic_rewards"],
            "total_rewards": result["total_rewards"],
            "original_fsfs": result["extra_info"]["original_fsfs"],
            "rewrite_fsfs": result["extra_info"]["rewrite_fsfs"],
            # Penalties for SwanLab logging
            "truncation_penalties": truncation_penalties,
            "format_penalties": format_penalties,
            # DAPO filter_groups support: "truncated" flag (1 if truncated, 0 otherwise)
            "truncated": truncated_flags,
        },
    }

    # Cleanup Student vLLM to release GPU memory for Actor vLLM reload
    if cleanup_after_compute:
        cleanup_reward_computer()

    return reward_result
