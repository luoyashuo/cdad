"""
Combined Reward Function for CoT Rewriter GRPO

Reward 计算逻辑:
- 使用 Qwen3 chat template 格式计算 Student Model 的 log prob
- R_cog = avg(log P_student(Answer | Question, Rewrite_CoT)) + alpha_cog * min(token_log_probs)
- R_cog = answer_log_prob(rewritten)  # 直接使用 rewrite 的 log prob
- alpha_cog 为可配置参数，默认为 0.01
- (可选) R_logic: FSF Judge 比较失败分支比例

This module provides the VERL-compatible interface for reward computation.

Supports two backends:
- sglang: Uses SGLang Engine for both Student and FSF Judge (recommended)
- vllm: Uses vLLM for Student and HuggingFace for FSF Judge
"""

from typing import Any, Dict, List, Optional

import torch

from verl import DataProto

from verl.experimental.reward.cot_rewriter.logging_config import logger


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
        sglang_gpu_memory_utilization: float = 0.4,
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
            sglang_gpu_memory_utilization: GPU memory fraction for SGLang per device (default: 0.4)
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward
        self.alpha_cog = alpha_cog
        self.reward_micro_batch_size = reward_micro_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.reward_backend = reward_backend

        logger.info(f"[CombinedRewardComputer] Initializing")
        logger.info(f"  Reward Backend: {reward_backend}")
        logger.info(f"  Device (FSF): {device}")
        logger.info(f"  Alpha Cog: {alpha_cog}")
        logger.info(f"  Student Devices: {student_devices}")
        logger.info(f"  Reward Micro Batch Size: {reward_micro_batch_size}")
        logger.info(f"  GPU Batch Size: {gpu_batch_size}")

        # Initialize Student Reward client based on backend
        if reward_backend == "sglang":
            logger.info(f"[CombinedRewardComputer] Using SGLang student reward (FAST mode)")
            logger.info(f"  SGLang GPU Memory Utilization: {sglang_gpu_memory_utilization}")
            from verl.experimental.reward.cot_rewriter.sglang_student_reward import SGLangStudentRewardClient
            self.student_client = SGLangStudentRewardClient(
                model_path=student_model_path,
                gpu_memory_utilization=sglang_gpu_memory_utilization,
                devices=student_devices,
            )
        else:
            logger.info(f"[CombinedRewardComputer] Using vLLM student reward (FAST mode)")
            logger.info(f"  vLLM GPU Memory Utilization: {vllm_gpu_memory_utilization}")
            from verl.experimental.reward.cot_rewriter.student_reward import VLLMStudentRewardClient
            self.student_client = VLLMStudentRewardClient(
                model_path=student_model_path,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                devices=student_devices,
            )

        # Initialize FSF Judge client (if needed)
        if use_logic_reward:
            if not fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")

            if reward_backend == "sglang":
                logger.info(f"[CombinedRewardComputer] Using SGLang FSF Judge")
                from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFJudge
                self.fsf_judge = SGLangFSFJudge(
                    model_path=fsf_model_path,
                    device=device,
                    gpu_memory_utilization=sglang_gpu_memory_utilization,
                )
            else:
                logger.info(f"[CombinedRewardComputer] Using HuggingFace FSF Judge")
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
        - answer_log_prob(q, cot, a) = avg(token_log_probs) + alpha_cog × min(token_log_probs)
        - R_cog = answer_log_prob(rewritten)  # 直接使用 rewrite 的 log prob
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
        logger.info(f"[REWARD] Computing reward for batch_size={batch_size}")

        # Compute cognitive reward
        logger.info(f"[REWARD] Computing cognitive reward with vLLM...")
        logger.info(f"[REWARD] Formula: reward = (avg + {self.alpha_cog}*min)(rewritten) - (avg + {self.alpha_cog}*min)(original)")
        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,
            alpha_cog=self.alpha_cog,
        )
        logger.info(f"[REWARD] Cognitive reward computed")

        # 获取计算结果
        r_cog = cog_result["rewards"]
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]

        # Log statistics
        valid_rewards = [r for r in r_cog if abs(r) < 99.0]
        if valid_rewards:
            logger.info(f"[REWARD] R_cog stats: mean={sum(valid_rewards)/len(valid_rewards):.4f}, "
                       f"min={min(valid_rewards):.4f}, max={max(valid_rewards):.4f}")

        # Compute logic reward (if enabled)
        if (
            self.use_logic_reward
            and self.fsf_judge
            and original_cots
            and all(c for c in original_cots)
        ):
            logger.info(f"[REWARD] Computing logic reward...")
            logic_result = self.fsf_judge.compute_logic_reward(
                original_cots=original_cots,
                rewrite_cots=rewrite_cots,
            )
            r_logic = logic_result["logic_rewards"]
            original_fsfs = logic_result["original_fsfs"]
            rewrite_fsfs = logic_result["rewrite_fsfs"]
            logger.info(f"[REWARD] Logic reward computed")
        else:
            logger.info(f"[REWARD] Skipping logic reward (disabled or no original cots)")
            r_logic = [0.0] * batch_size
            original_fsfs = [0.0] * batch_size
            rewrite_fsfs = [0.0] * batch_size

        # Combine rewards: R = R_cog + lambda * R_logic
        logger.info(f"[REWARD] Combining rewards with lambda_logic={self.lambda_logic}")
        total_rewards = [c + self.lambda_logic * l for c, l in zip(r_cog, r_logic)]
        logger.info(f"[REWARD] Reward computation completed. Returning results...")

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
    field_mapping: Optional[Dict[str, str]] = None,
    student_devices: Optional[List[str]] = None,
    reward_micro_batch_size: Optional[int] = None,
    gpu_batch_size: int = 8,
    vllm_gpu_memory_utilization: float = 0.3,
    cleanup_after_compute: bool = True,
    reward_backend: str = "vllm",  # "sglang" or "vllm"
    sglang_gpu_memory_utilization: float = 0.4,
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
        sglang_gpu_memory_utilization: GPU memory fraction for SGLang per device

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
    logger.info(f"[REWARD_FN] compute_cot_rewriter_reward called with batch_size={batch_size}")

    # Decode prompts and responses
    logger.info(f"[REWARD_FN] Decoding prompts and responses...")
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True,
    )
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True,
    )
    logger.info(f"[REWARD_FN] Decoded {len(prompts)} prompts and {len(rewrite_cots)} responses")

    # Detect truncated responses
    # A response is considered truncated if:
    # 1. It reaches max_response_length (response_mask is all 1s)
    # 2. It doesn't end with EOS token
    truncated_indices = set()
    truncation_penalty = kwargs.get("truncation_penalty", -1.0)

    response_ids = data.batch["responses"]  # [batch_size, seq_len]
    response_mask = data.batch.get("response_mask", None)
    eos_token_id = tokenizer.eos_token_id

    for i in range(batch_size):
        is_truncated = False

        # Method 1: Check if response_mask is all 1s (reached max length)
        if response_mask is not None:
            mask_sum = response_mask[i].sum().item()
            max_len = response_mask[i].shape[0]
            if mask_sum >= max_len * 0.99:  # Allow small tolerance
                is_truncated = True

        # Method 2: Check if last non-padding token is EOS
        if not is_truncated:
            # Find last non-padding token
            resp_ids = response_ids[i]
            # Get the actual response length from mask
            if response_mask is not None:
                actual_len = int(response_mask[i].sum().item())
            else:
                # Find first padding token (usually 0 or pad_token_id)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                actual_len = (resp_ids != pad_id).sum().item()

            if actual_len > 0:
                last_token_id = resp_ids[actual_len - 1].item()
                if last_token_id != eos_token_id:
                    is_truncated = True

        if is_truncated:
            truncated_indices.add(i)

    if truncated_indices:
        logger.warning(f"[REWARD_FN] Detected {len(truncated_indices)} truncated responses (will get reward={truncation_penalty})")
        logger.info(f"[REWARD_FN] Truncated indices: {sorted(truncated_indices)[:10]}{'...' if len(truncated_indices) > 10 else ''}")

    # Extract metadata from non_tensor_batch
    logger.info(f"[REWARD_FN] Extracting metadata from non_tensor_batch...")
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

    logger.info(f"[REWARD_FN] Extracted {len(questions)} questions, {len(answers)} answers, {len(original_cots)} original_cots")

    non_empty_cots = sum(1 for c in original_cots if c)
    logger.info(f"[REWARD_FN] Non-empty original_cots: {non_empty_cots}/{len(original_cots)}")
    if non_empty_cots == 0:
        logger.warning(f"[REWARD_FN] WARNING: No original_cots found!")

    print_samples = kwargs.get("print_samples", 2)

    # Filter out truncated samples for reward computation
    non_truncated_indices = [i for i in range(batch_size) if i not in truncated_indices]

    if non_truncated_indices:
        # Get or create reward computer only if there are non-truncated samples
        logger.info(f"[REWARD_FN] Getting reward computer (backend={reward_backend})...")
        logger.info(f"[REWARD_FN] Computing rewards for {len(non_truncated_indices)} non-truncated samples (skipping {len(truncated_indices)} truncated)")
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
            sglang_gpu_memory_utilization=sglang_gpu_memory_utilization,
        )

        # Filter data for non-truncated samples
        filtered_questions = [questions[i] for i in non_truncated_indices]
        filtered_original_cots = [original_cots[i] for i in non_truncated_indices]
        filtered_rewrite_cots = [rewrite_cots[i] for i in non_truncated_indices]
        filtered_answers = [answers[i] for i in non_truncated_indices]

        # Compute rewards only for non-truncated samples
        logger.info(f"[REWARD_FN] Calling reward_computer.compute_reward()...")
        filtered_result = reward_computer.compute_reward(
            questions=filtered_questions,
            original_cots=filtered_original_cots,
            rewrite_cots=filtered_rewrite_cots,
            answers=filtered_answers,
        )
        logger.info(f"[REWARD_FN] Reward computer returned results for {len(non_truncated_indices)} samples")
    else:
        logger.warning(f"[REWARD_FN] All {batch_size} samples are truncated! Skipping Student model computation.")
        filtered_result = None

    # Build full result arrays with truncation penalties
    result = {
        "cognitive_rewards": [truncation_penalty] * batch_size,
        "rewrite_mean_log_probs": [-100.0] * batch_size,
        "original_mean_log_probs": [-100.0] * batch_size,
        "logic_rewards": [0.0] * batch_size,
        "total_rewards": [truncation_penalty] * batch_size,
        "token_log_probs": [[]] * batch_size,
        "token_ids": [[]] * batch_size,
        "extra_info": {
            "original_fsfs": [0.0] * batch_size,
            "rewrite_fsfs": [0.0] * batch_size,
        },
    }

    # Fill in results for non-truncated samples
    if filtered_result is not None:
        for j, i in enumerate(non_truncated_indices):
            result["cognitive_rewards"][i] = filtered_result["cognitive_rewards"][j]
            result["rewrite_mean_log_probs"][i] = filtered_result["rewrite_mean_log_probs"][j]
            result["original_mean_log_probs"][i] = filtered_result["original_mean_log_probs"][j]
            result["logic_rewards"][i] = filtered_result["logic_rewards"][j]
            result["total_rewards"][i] = filtered_result["total_rewards"][j]
            result["token_log_probs"][i] = filtered_result.get("token_log_probs", [[]])[j] if j < len(filtered_result.get("token_log_probs", [])) else []
            result["token_ids"][i] = filtered_result.get("token_ids", [[]])[j] if j < len(filtered_result.get("token_ids", [])) else []
            result["extra_info"]["original_fsfs"][i] = filtered_result["extra_info"]["original_fsfs"][j]
            result["extra_info"]["rewrite_fsfs"][i] = filtered_result["extra_info"]["rewrite_fsfs"][j]

    # Log truncation statistics
    if truncated_indices:
        logger.info(f"[REWARD_FN] Truncation summary: {len(truncated_indices)}/{batch_size} samples truncated, assigned reward={truncation_penalty}")

    # Convert to tensor
    logger.info(f"[REWARD_FN] Converting rewards to tensors...")
    device = data.batch["responses"].device
    total_rewards = torch.tensor(
        result["total_rewards"],
        device=device,
        dtype=torch.bfloat16,
    )

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

    # Print rollout samples with rewards for review
    if print_samples > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"[ROLLOUT OUTPUT WITH REWARDS] Showing first {min(print_samples, batch_size)} samples:")
        logger.info(f"[TRUNCATION STATS] {len(truncated_indices)}/{batch_size} truncated")
        logger.info(f"{'='*60}")
        for i in range(min(print_samples, batch_size)):
            truncated_marker = " [TRUNCATED]" if i in truncated_indices else ""
            logger.info(f"\n[Sample {i}]{truncated_marker}")
            logger.info(f"[question] {prompts[i][:500]}..." if len(prompts[i]) > 500 else f"[question] {prompts[i]}")
            logger.info(f"[rollout_output] {rewrite_cots[i][:1000]}..." if len(rewrite_cots[i]) > 1000 else f"[rollout_output] {rewrite_cots[i]}")
            if i < len(original_cots) and original_cots[i]:
                logger.info(f"[old_cot] {original_cots[i][:500]}..." if len(original_cots[i]) > 500 else f"[old_cot] {original_cots[i]}")
            logger.info(f"[total_reward] {result['total_rewards'][i]}{' (truncation penalty)' if i in truncated_indices else ''}")
            logger.info(f"[cognitive_reward] {result['cognitive_rewards'][i]}")
            logger.info("-" * 50)
        logger.info(f"{'='*60}\n")

    # Build truncated mask (per-sample, can be chunked by VERL)
    truncated_mask = [1.0 if i in truncated_indices else 0.0 for i in range(batch_size)]

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
            # Per-sample truncation mask (1.0 = truncated, 0.0 = not truncated)
            # This can be properly chunked by VERL's data sharding
            "truncated_mask": truncated_mask,
        },
    }

    logger.info(f"[REWARD_FN] compute_cot_rewriter_reward completed")

    # Cleanup Student vLLM to release GPU memory for Actor vLLM reload
    if cleanup_after_compute:
        logger.info(f"[REWARD_FN] Cleaning up Student vLLM to release GPU memory...")
        cleanup_reward_computer()

    return reward_result
