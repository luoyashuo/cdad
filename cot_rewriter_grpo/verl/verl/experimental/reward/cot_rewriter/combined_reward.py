"""
Combined Reward Function for CoT Rewriter GRPO

Reward 计算逻辑:
- 使用 Qwen3 chat template 格式计算 Student Model 的困惑度 (Perplexity)
- PPL(Q, CoT, A) = exp(-mean(token_log_probs))
- R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
        = PPL(original) - PPL(rewrite)

困惑度公式的含义:
- PPL 越低表示模型预测越准确
- 如果 rewrite 后的 PPL 更低（模型预测更准确），则 PPL(original) - PPL(rewrite) > 0，reward 为正
- 如果 rewrite 后的 PPL 更高（模型预测更差），则 PPL(original) - PPL(rewrite) < 0，reward 为负
- (可选) R_logic: FSF Judge 比较失败分支比例

This module provides the VERL-compatible interface for reward computation.

Supports two backends:
- sglang: Uses SGLang Engine for both Student and FSF Judge (recommended)
- vllm: Uses vLLM for Student and HuggingFace for FSF Judge
"""

from typing import Any, Dict, List, Optional, Tuple
import re

import torch

from verl import DataProto

from verl.experimental.reward.cot_rewriter.training_logger import get_logger

# Training logger for reward computation
logger = get_logger("reward")


def extract_rewrite_reasoning(text: str) -> Tuple[str, bool, str]:
    """
    Extract content from <rewrite_reasoning> tags with STRICT format checking.

    Strict format requirements:
    - The text must START with <rewrite_reasoning> (after stripping leading whitespace)
    - The text must END with </rewrite_reasoning> (after stripping trailing whitespace)
    - Must have EXACTLY ONE opening tag and ONE closing tag
    - Tags must be properly paired

    Args:
        text: The raw response text from the model

    Returns:
        Tuple of (extracted_content, success_flag, error_reason)
        - extracted_content: The content inside the tags if valid, empty string otherwise
        - success_flag: True if format is strictly valid, False otherwise
        - error_reason: Description of why validation failed (for logging)
    """
    if not text:
        return "", False, "empty_text"

    # Strip leading/trailing whitespace for format checking
    stripped_text = text.strip()

    # Count tags - must be exactly 1 of each
    open_tag_count = stripped_text.count('<rewrite_reasoning>')
    close_tag_count = stripped_text.count('</rewrite_reasoning>')

    # Check tag counts
    if open_tag_count != 1:
        return "", False, f"wrong_open_tag_count:{open_tag_count}"
    if close_tag_count != 1:
        return "", False, f"wrong_close_tag_count:{close_tag_count}"

    # Strict format check:
    # 1. Must start with <rewrite_reasoning>
    # 2. Must end with </rewrite_reasoning>
    starts_with_open_tag = stripped_text.startswith('<rewrite_reasoning>')
    ends_with_close_tag = stripped_text.endswith('</rewrite_reasoning>')

    if not starts_with_open_tag:
        return "", False, "not_start_with_open_tag"
    if not ends_with_close_tag:
        return "", False, "not_end_with_close_tag"

    # Extract content between tags
    # Remove opening tag and closing tag
    content = stripped_text[len('<rewrite_reasoning>'):-len('</rewrite_reasoning>')].strip()

    # Additional check: content should not contain the tags (nested tags not allowed)
    if '<rewrite_reasoning>' in content or '</rewrite_reasoning>' in content:
        return "", False, "nested_tags_detected"

    return content, True, "ok"


class CombinedRewardComputer:
    """
    Combined Reward Computer

    Computes R = R_cog + lambda * R_logic using Student Model.

    Supports two backends:
    - sglang: Uses SGLang Engine for both Student and FSF Judge (recommended)
    - vllm: Uses vLLM for Student and HuggingFace for FSF Judge
    """

    # Step size for tiered reward bins (0.05 per bin)
    TIER_STEP = 0.05
    # Reward per tier (each tier gets tier_index * tier_reward)
    TIER_REWARD = 0.1

    @staticmethod
    def apply_tiered_reward(
        r_cog: float,
        tier_step: float = TIER_STEP,
        tier_reward: float = TIER_REWARD
    ) -> float:
        """
        Apply tiered (step-wise) reward function to R_cog.

        Rules:
        - If r_cog < 0: return 0 (no reward for negative deltas)
        - If r_cog >= 0: assign to tier based on value

        Formula: reward = (floor(r_cog / tier_step) + 1) * tier_reward

        Examples with tier_step=0.05, tier_reward=0.1:
        - 0.0-0.05 -> 0.1  (tier_index=0: (0+1)*0.1=0.1)
        - 0.05-0.10 -> 0.2  (tier_index=1: (1+1)*0.1=0.2)
        - 0.10-0.15 -> 0.3  (tier_index=2: (2+1)*0.1=0.3)

        Examples with tier_step=0.1, tier_reward=0.05:
        - 0.0-0.1 -> 0.05  (tier_index=0: (0+1)*0.05=0.05)
        - 0.1-0.2 -> 0.10  (tier_index=1: (1+1)*0.05=0.10)
        - 0.2-0.3 -> 0.15  (tier_index=2: (2+1)*0.05=0.15)

        Args:
            r_cog: Raw cognitive reward value
            tier_step: 区间大小，每个阶梯的 R_cog 范围 (default: 0.05)
            tier_reward: 每阶梯奖励值，第一阶梯得 tier_reward，第二阶梯得 2*tier_reward (default: 0.1)

        Returns:
            Tiered reward value
        """
        if r_cog < 0:
            return 0.0
        tier_index = int(r_cog / tier_step)  # floor division
        return (tier_index + 1) * tier_reward

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
        too_short_threshold: float = 0.5,  # Deprecated: use too_short_threshold_low/high instead
        too_short_threshold_low: float = 0.4,  # Lower threshold for too_short_penalty
        too_short_threshold_high: float = 0.8,  # Upper threshold for too_short_penalty
        too_short_max_penalty: float = 0.5,  # Maximum penalty for too short outputs
        use_tiered_reward: bool = True,  # Whether to use tiered reward for R_cog
        tier_step: float = 0.05,  # 区间大小：每个阶梯的 R_cog 范围
        tier_reward: float = 0.1,  # 每阶梯奖励值：第一阶梯得 tier_reward，第二阶梯得 2*tier_reward
        reward_mode: str = "ppl",  # "ppl" or "logprob": Reward calculation mode
                                   # - "ppl": R_cog = PPL(original) - PPL(rewrite)
                                   # - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
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
            too_short_threshold: Deprecated. Use too_short_threshold_low/high instead.
            too_short_threshold_low: Lower threshold for too_short_penalty (default: 0.4). When
                rewrite_ratio < low, penalty is at maximum. Between low and high, penalty scales linearly.
            too_short_threshold_high: Upper threshold for too_short_penalty (default: 0.8). When
                rewrite_ratio >= high, no penalty is applied.
            too_short_max_penalty: Maximum penalty for too short outputs (default: 0.5).
            use_tiered_reward: Whether to use tiered (step-wise) reward for R_cog (default: True).
            tier_step: 区间大小，每个阶梯的 R_cog 区间 (default: 0.05).
            tier_reward: 每阶梯奖励值，第一阶梯得 tier_reward，第二阶梯得 2*tier_reward (default: 0.1).
            reward_mode: Reward calculation mode (default: "ppl").
                - "ppl": R_cog = PPL(original) - PPL(rewrite), where PPL = exp(-mean_log_prob)
                - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original),
                  where answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)

        阶梯式奖励配置示例:
        - tier_step=0.05, tier_reward=0.1:
            0.0-0.05 -> 0.1, 0.05-0.1 -> 0.2, 0.1-0.15 -> 0.3, ...
        - tier_step=0.1, tier_reward=0.05:
            0.0-0.1 -> 0.05, 0.1-0.2 -> 0.1, 0.2-0.3 -> 0.15, ...
        - tier_step=0.03, tier_reward=0.2:
            0.0-0.03 -> 0.2, 0.03-0.06 -> 0.4, 0.06-0.09 -> 0.6, ...
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward
        self.alpha_cog = alpha_cog
        self.reward_micro_batch_size = reward_micro_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.reward_backend = reward_backend
        # Use low/high thresholds directly (no longer need backward compatibility check)
        # The caller should provide too_short_threshold_low/high explicitly
        self.too_short_threshold_low = too_short_threshold_low
        self.too_short_threshold_high = too_short_threshold_high
        self.too_short_max_penalty = too_short_max_penalty
        self.use_tiered_reward = use_tiered_reward
        self.tier_step = tier_step
        self.tier_reward = tier_reward
        self.reward_mode = reward_mode

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
                reward_mode=reward_mode,
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
                reward_mode=reward_mode,
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

        Reward 逻辑 (根据 reward_mode 选择):
        - "ppl" 模式:
            - PPL(Q, CoT, A) = exp(-mean(token_log_probs))
            - R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
                    = PPL(original) - PPL(rewrite)
        - "logprob" 模式:
            - answer_log_prob(Q, CoT, A) = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            - R_cog = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, Original_CoT, A)
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

        # Compute cognitive reward (mode depends on self.reward_mode)
        if self.reward_mode == "ppl":
            # R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
            #      = PPL(original) - PPL(rewrite)
            # 其中 PPL = exp(-mean_log_prob)
            pass
        else:  # "logprob" mode
            # R_cog = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, Original_CoT, A)
            # 其中 answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            pass
        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,
            alpha_cog=self.alpha_cog,
        )

        # 获取计算结果
        r_cog_raw = cog_result["rewards"]
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]

        # Apply tiered reward transformation if enabled
        if self.use_tiered_reward:
            r_cog = [self.apply_tiered_reward(r, self.tier_step, self.tier_reward) for r in r_cog_raw]
            # Log tiered reward transformation for first few samples
            for i in range(min(3, batch_size)):
                if r_cog_raw[i] != r_cog[i]:
                    logger.info(
                        f"[TIERED_REWARD] Sample {i}: r_cog_raw={r_cog_raw[i]:.4f} -> r_cog_tiered={r_cog[i]:.4f} "
                        f"(tier_step={self.tier_step}, tier_reward={self.tier_reward})"
                    )
        else:
            r_cog = r_cog_raw

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

        # Compute too_short_penalty: penalize rewrites that are too short compared to original_cot
        # Penalty formula with low/high thresholds:
        #   ratio = rewrite_token_len / original_token_len
        #   - If ratio >= high: penalty = 0
        #   - If low <= ratio < high: penalty = -max_penalty * (high - ratio) / (high - low)
        #   - If ratio < low: penalty = -max_penalty
        too_short_penalties = [0.0] * batch_size
        length_ratios = [0.0] * batch_size  # Track length ratios for DAPO filter_groups

        # Get tokenizer from student_client for token length calculation
        tokenizer = getattr(self.student_client, 'tokenizer', None)
        if tokenizer is None:
            logger.warning("[TOO_SHORT_PENALTY] Tokenizer not available, skipping too_short_penalty")
        else:
            for i in range(batch_size):
                original_text = original_cots[i] if original_cots[i] else ""
                rewrite_text = rewrite_cots[i] if rewrite_cots[i] else ""

                if original_text:
                    # Calculate token lengths using tokenizer
                    original_token_len = len(tokenizer.encode(original_text, add_special_tokens=False))
                    rewrite_token_len = len(tokenizer.encode(rewrite_text, add_special_tokens=False)) if rewrite_text else 0

                    if original_token_len > 0:
                        ratio = rewrite_token_len / original_token_len
                        length_ratios[i] = ratio  # Store for DAPO filter_groups
                        low = self.too_short_threshold_low
                        high = self.too_short_threshold_high

                        if ratio >= high:
                            # No penalty when ratio is above high threshold
                            penalty = 0.0
                        elif ratio < low:
                            # Maximum penalty when ratio is below low threshold
                            penalty = -self.too_short_max_penalty
                        else:
                            # Linear penalty between low and high: 0 at high, max_penalty at low
                            penalty = -self.too_short_max_penalty * (high - ratio) / (high - low)

                        if penalty != 0:
                            too_short_penalties[i] = penalty
                            if i < 3:  # Log first few samples
                                logger.info(
                                    f"[TOO_SHORT_PENALTY] Sample {i}: "
                                    f"original_token_len={original_token_len}, rewrite_token_len={rewrite_token_len}, "
                                    f"ratio={ratio:.3f} (low={low}, high={high}), penalty={penalty:.4f}"
                                )

        # Log length_ratio statistics for DAPO filter_groups
        valid_ratios = [r for r in length_ratios if r > 0]
        if valid_ratios:
            low = self.too_short_threshold_low
            high = self.too_short_threshold_high
            num_above_low = sum(1 for r in valid_ratios if r >= low)
            num_above_high = sum(1 for r in valid_ratios if r >= high)
            logger.info(
                f"[LENGTH_RATIO] batch_size={batch_size}, "
                f"mean={sum(valid_ratios)/len(valid_ratios):.3f}, "
                f"min={min(valid_ratios):.3f}, max={max(valid_ratios):.3f}, "
                f"low={low}, high={high}, "
                f"num_above_low={num_above_low}/{len(valid_ratios)}, "
                f"num_above_high={num_above_high}/{len(valid_ratios)}"
            )

        # Combine rewards: R = R_cog + lambda * R_logic + too_short_penalty
        total_rewards = [c + self.lambda_logic * l + p for c, l, p in zip(r_cog, r_logic, too_short_penalties)]
        
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
                "too_short_penalties": too_short_penalties,
                "length_ratios": length_ratios,  # For DAPO filter_groups
            },
        }


# Global singleton
_reward_computer: Optional[CombinedRewardComputer] = None
_reward_computer_kwargs: Optional[dict] = None  # Track kwargs used to create the singleton


def get_reward_computer(**kwargs) -> CombinedRewardComputer:
    """Get or create Combined Reward Computer singleton."""
    global _reward_computer, _reward_computer_kwargs
    if _reward_computer is None:
        _reward_computer_kwargs = kwargs.copy()
        logger.info(f"[REWARD] Creating new CombinedRewardComputer with too_short_threshold_low={kwargs.get('too_short_threshold_low')}, too_short_threshold_high={kwargs.get('too_short_threshold_high')}")
        _reward_computer = CombinedRewardComputer(**kwargs)
    else:
        # Log if kwargs differ from initial creation
        threshold_low = kwargs.get('too_short_threshold_low')
        threshold_high = kwargs.get('too_short_threshold_high')
        initial_low = _reward_computer_kwargs.get('too_short_threshold_low')
        initial_high = _reward_computer_kwargs.get('too_short_threshold_high')
        if threshold_low != initial_low or threshold_high != initial_high:
            logger.warning(
                f"[REWARD] Reward computer already exists with different thresholds! "
                f"Initial: low={initial_low}, high={initial_high}. "
                f"Current call: low={threshold_low}, high={threshold_high}. "
                f"Using initial values."
            )
    return _reward_computer


def cleanup_reward_computer():
    """Cleanup the global reward computer to release GPU memory."""
    global _reward_computer, _reward_computer_kwargs
    if _reward_computer is not None:
        logger.info("[REWARD] Cleaning up reward computer...")
        if hasattr(_reward_computer, 'student_client') and _reward_computer.student_client is not None:
            _reward_computer.student_client.cleanup()
        if hasattr(_reward_computer, 'fsf_judge') and _reward_computer.fsf_judge is not None:
            # FSF Judge cleanup if needed
            del _reward_computer.fsf_judge
        _reward_computer = None
        _reward_computer_kwargs = None  # Reset kwargs for next creation

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
    too_short_threshold: float = 0.5,  # Deprecated: use too_short_threshold_low/high instead
    too_short_threshold_low: float = 0.4,  # Lower threshold for too_short_penalty
    too_short_threshold_high: float = 0.8,  # Upper threshold for too_short_penalty
    too_short_max_penalty: float = 0.5,  # Maximum penalty for too short outputs
    use_tiered_reward: bool = True,  # Whether to use tiered (step-wise) reward for R_cog
    tier_step: float = 0.05,  # 区间大小：每个阶梯的 R_cog 范围
    tier_reward: float = 0.1,  # 每阶梯奖励值：第一阶梯得 tier_reward，第二阶梯得 2*tier_reward
    reward_mode: str = "ppl",  # "ppl" or "logprob": Reward calculation mode
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
        # IMPORTANT: Use data[i] to get per-sample data (like VERL's naive.py/prime.py/dapo.py)
        # This ensures each sample gets its own reward_model dict, not a shared reference
        data_item = data[i]

        if use_flat_structure:
            reward_info = {}
        else:
            reward_info = data_item.non_tensor_batch.get(reward_model_key, {})
            if not isinstance(reward_info, dict):
                reward_info = {}

        # Extract ground truth answer
        answer = reward_info.get(ground_truth_key, "")
        if not answer:
            answer = reward_info.get("ground_truth", "")
        if not answer:
            answer = reward_info.get("answer", "")
        if not answer and ground_truth_key in data_item.non_tensor_batch:
            gt_data = data_item.non_tensor_batch[ground_truth_key]
            if gt_data:
                answer = gt_data
        answers.append(answer)

        # Extract original CoT
        # First try from data_item.non_tensor_batch directly (like VERL's batch.py)
        orig_cot = ""
        if original_cot_key in data_item.non_tensor_batch:
            orig_cot_data = data_item.non_tensor_batch[original_cot_key]
            if orig_cot_data:
                orig_cot = orig_cot_data
                if hasattr(orig_cot, 'item'):
                    orig_cot = orig_cot.item()
        # Fallback to reward_model (like VERL's naive.py/prime.py)
        if not orig_cot and original_cot_key in reward_info:
            orig_cot = reward_info.get(original_cot_key, "")
        if not orig_cot:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot if orig_cot else "")

        # Extract question (like VERL's naive.py/prime.py)
        question = reward_info.get(question_key, "")
        if not question:
            question = reward_info.get("question", "")
        if not question:
            question = prompts[i]
        questions.append(question)

    print_samples = kwargs.get("print_samples", 2)

    # Extract <rewrite_reasoning> content from rewrite_cots for reward computation
    # and track which samples have valid format
    extracted_rewrite_cots = []
    format_valid_flags = []  # 1 if valid, 0 if invalid

    for i, rewrite_cot in enumerate(rewrite_cots):
        extracted_cot, has_valid_format, error_reason = extract_rewrite_reasoning(rewrite_cot)
        # For valid format: use extracted content; for invalid format: use empty string (no cognitive reward)
        extracted_rewrite_cots.append(extracted_cot if has_valid_format else "")
        format_valid_flags.append(1 if has_valid_format else 0)
        if not has_valid_format:
            stripped_preview = rewrite_cot.strip()[:100] if rewrite_cot else ""
            logger.info(
                f"[REWARD] Sample {i}: Format violation - reason={error_reason}. "
                f"Preview: '{stripped_preview}...'. Will apply format penalty without cognitive reward."
            )

    # Prepare result containers for all samples
    batch_size = len(questions)
    cognitive_rewards = [0.0] * batch_size
    rewrite_mean_log_probs = [0.0] * batch_size
    original_mean_log_probs = [0.0] * batch_size
    logic_rewards = [0.0] * batch_size
    total_rewards = [0.0] * batch_size
    token_log_probs = [[] for _ in range(batch_size)]
    token_ids = [[] for _ in range(batch_size)]
    original_fsfs = [0.0] * batch_size
    rewrite_fsfs = [0.0] * batch_size

    # Compute rewards for ALL samples (both valid and invalid format)
    # For samples with invalid format, we use the original text instead of extracted content
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
        too_short_threshold=too_short_threshold,
        too_short_threshold_low=too_short_threshold_low,
        too_short_threshold_high=too_short_threshold_high,
        too_short_max_penalty=too_short_max_penalty,
        use_tiered_reward=use_tiered_reward,
        tier_step=tier_step,
        tier_reward=tier_reward,
        reward_mode=reward_mode,
    )

    # Compute rewards for all samples
    result = reward_computer.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=extracted_rewrite_cots,
        answers=answers,
    )

    # Fill in results for all samples
    cognitive_rewards = result["cognitive_rewards"]
    rewrite_mean_log_probs = result["rewrite_mean_log_probs"]
    original_mean_log_probs = result["original_mean_log_probs"]
    logic_rewards = result["logic_rewards"]
    total_rewards = result["total_rewards"]
    token_log_probs = result["token_log_probs"]
    token_ids = result["token_ids"]
    original_fsfs = result["extra_info"]["original_fsfs"]
    rewrite_fsfs = result["extra_info"]["rewrite_fsfs"]
    too_short_penalties = result["extra_info"].get("too_short_penalties", [0.0] * batch_size)
    length_ratios = result["extra_info"].get("length_ratios", [0.0] * batch_size)

    # Apply penalties
    # 1. Truncation penalty: if rollout result was truncated (finish_reason == "length")
    # 2. Format penalty: if response does not contain valid <rewrite_reasoning> tags
    finish_reasons = data.non_tensor_batch.get("finish_reasons", [])

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

        # Format penalty: check if response contains valid <rewrite_reasoning> tags
        # Note: format_valid_flags is already computed above
        if format_valid_flags[i] == 0:
            penalties[i] -= format_penalty
            format_penalties[i] = -format_penalty
            logger.info(
                f"[PENALTY] Sample {i}: Format violation (no valid <rewrite_reasoning> tags found), "
                f"response_preview='{rewrite_cots[i][:100]}...', penalty=-{format_penalty}, "
                f"no cognitive reward computed"
            )

    # Convert total_rewards to tensor and apply penalties
    device = data.batch["responses"].device
    total_rewards_tensor = torch.tensor(
        total_rewards,
        device=device,
        dtype=torch.bfloat16,
    )
    total_rewards_tensor = total_rewards_tensor + torch.tensor(penalties, device=device, dtype=torch.bfloat16)

    # Handle NaN and extreme values
    total_rewards_tensor = torch.nan_to_num(total_rewards_tensor, nan=0.0)
    total_rewards_tensor = total_rewards_tensor.clamp(-100, 100)

    # Convert to token-level rewards
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size,
        response_length,
        device=device,
        dtype=total_rewards_tensor.dtype,
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
        token_level_rewards[i, seq_lengths[i]] = total_rewards_tensor[i]

    reward_result = {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": cognitive_rewards,
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "logic_rewards": logic_rewards,
            "total_rewards": total_rewards,
            "original_fsfs": original_fsfs,
            "rewrite_fsfs": rewrite_fsfs,
            # Penalties for SwanLab logging
            "truncation_penalties": truncation_penalties,
            "format_penalties": format_penalties,
            "too_short_penalties": too_short_penalties,
            # DAPO filter_groups support: "truncated" flag (1 if truncated, 0 otherwise)
            "truncated": truncated_flags,
            # DAPO filter_groups support: "format_valid" flag (1 if valid, 0 if invalid)
            # Used to filter out groups with all invalid format samples
            "format_valid": format_valid_flags,
            # DAPO filter_groups support: "length_ratio" metric for length-based filtering
            "length_ratio": length_ratios,
        },
    }

    # Cleanup Student vLLM to release GPU memory for Actor vLLM reload
    if cleanup_after_compute:
        cleanup_reward_computer()

    return reward_result


def compute_length_ratio_only(
    data: DataProto,
    tokenizer: Any,
    field_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Lightweight function that only computes length_ratio for filtering.

    This function does NOT load the Student Model, making it fast for pre-filtering.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for computing token lengths
        field_mapping: Optional dict to customize field names in data

    Returns:
        Dictionary with:
            - length_ratio: List of length ratios (rewrite_len / original_len)
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
    original_cot_key = default_mapping["original_cot_key"]

    # Handle case where reward_model_key is None or empty (flat structure)
    use_flat_structure = not reward_model_key or reward_model_key == "null"

    batch_size = data.batch["responses"].shape[0]

    # Decode responses
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True,
    )

    # Extract metadata from non_tensor_batch
    original_cots = []
    for i in range(batch_size):
        # Extract original CoT
        # IMPORTANT: Use data[i] to get per-sample data (like VERL's naive.py/prime.py/dapo.py)
        # This ensures each sample gets its own reward_model dict, not a shared reference
        data_item = data[i]

        orig_cot = ""
        if original_cot_key in data_item.non_tensor_batch:
            orig_cot_data = data_item.non_tensor_batch[original_cot_key]
            if orig_cot_data:
                orig_cot = orig_cot_data
                if hasattr(orig_cot, 'item'):
                    orig_cot = orig_cot.item()
        if not orig_cot and not use_flat_structure:
            reward_info = data_item.non_tensor_batch.get(reward_model_key, {})
            if isinstance(reward_info, dict):
                orig_cot = reward_info.get(original_cot_key, "")
        original_cots.append(orig_cot if orig_cot else "")

    # Compute length_ratios without loading Student Model
    length_ratios = [0.0] * batch_size
    for i in range(batch_size):
        original_text = original_cots[i] if original_cots[i] else ""
        rewrite_text = rewrite_cots[i] if rewrite_cots[i] else ""

        if original_text:
            original_token_len = len(tokenizer.encode(original_text, add_special_tokens=False))
            rewrite_token_len = len(tokenizer.encode(rewrite_text, add_special_tokens=False)) if rewrite_text else 0
            if original_token_len > 0:
                length_ratios[i] = rewrite_token_len / original_token_len

    # Log statistics
    valid_ratios = [r for r in length_ratios if r > 0]
    if valid_ratios:
        logger.info(
            f"[LENGTH_RATIO_ONLY] batch_size={batch_size}, "
            f"mean={sum(valid_ratios)/len(valid_ratios):.3f}, "
            f"min={min(valid_ratios):.3f}, max={max(valid_ratios):.3f}"
        )

    return {
        "length_ratio": length_ratios,
    }
