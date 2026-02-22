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
from difflib import SequenceMatcher

import numpy as np
import torch

from verl import DataProto

from verl.experimental.reward.cot_rewriter.training_logger import get_logger

# Import Correct Judge (lazy import in code to avoid circular dependency)

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
    - Must NOT contain "## Problem" (indicates format error)

    FALLBACK extraction (when strict format fails):
    - Uses strip().lstrip("<rewrite_reasoning>").rstrip("</rewrite_reasoning>")
    - Returns the extracted content with success_flag=False
    - This allows reward computation even with malformed format

    Args:
        text: The raw response text from the model

    Returns:
        Tuple of (extracted_content, success_flag, error_reason)
        - extracted_content: The content inside the tags (strict or fallback extraction)
        - success_flag: True if format is strictly valid, False otherwise
        - error_reason: Description of why validation failed (for logging)
    """
    if not text:
        return "", False, "empty_text"

    # Check for "## " marker (format error)
    if "## " in text: 
        return "", False, "contains_problem_marker"

    # Define tags
    open_tag = '<rewrite_reasoning>'
    close_tag = '</rewrite_reasoning>'

    # Strip leading/trailing whitespace for format checking
    stripped_text = text.strip()

    # Count tags - must be exactly 1 of each
    open_tag_count = stripped_text.count(open_tag)
    close_tag_count = stripped_text.count(close_tag)

    # Strict format check:
    # 1. Must have exactly 1 opening tag and 1 closing tag
    # 2. Must start with <rewrite_reasoning>
    # 3. Must end with </rewrite_reasoning>
    starts_with_open_tag = stripped_text.startswith(open_tag)
    ends_with_close_tag = stripped_text.endswith(close_tag)
    has_exact_tags = open_tag_count == 1 and close_tag_count == 1

    # Check for nested tags
    has_nested = False
    error_reason = ""
    if has_exact_tags and starts_with_open_tag and ends_with_close_tag:
        content = stripped_text[len(open_tag):-len(close_tag)]
        has_nested = open_tag in content or close_tag in content
        # 增加一些发现的Format Error
        mention_rewrite_error = False
        content_prefix = content.strip()[:60].lower()
        if "rewrit" in content_prefix:
            mention_rewrite_error = True
            error_reason = "mention_rewrite_in_reasoning"
        elif "process" in content_prefix or "reasoning" in content_prefix:
            mention_rewrite_error = True
            error_reason = "mention_reasoning_process_in_reasoning"
        elif "bridge" in content_prefix or "clari" in content_prefix or 'deriv' in content_prefix or "refin" in content_prefix:
            mention_rewrite_error = True
            error_reason = "mention_bridge_or_clari_in_reasoning"
        if not has_nested and not mention_rewrite_error:
            # Strict format valid
            return content.strip(), True, "ok"

    # === STRICT FORMAT FAILED - USE FALLBACK EXTRACTION ===
    # Use strip().lstrip(open_tag).rstrip(close_tag) for fallback extraction
    # This handles cases like:
    # - "some text <rewrite_reasoning>content</rewrite_reasoning>"
    # - "<rewrite_reasoning>content</rewrite_reasoning> some text"
    # - multiple tags (takes first match)

    
    if open_tag_count == 0:
        error_reason = "missing_open_tag"
    elif close_tag_count == 0:
        error_reason = "missing_close_tag"
    elif open_tag_count > 1:
        error_reason = f"multiple_open_tags:{open_tag_count}"
    elif close_tag_count > 1:
        error_reason = f"multiple_close_tags:{close_tag_count}"
    elif not starts_with_open_tag:
        error_reason = "not_start_with_open_tag"
    elif not ends_with_close_tag:
        error_reason = "not_end_with_close_tag"
    elif has_nested:
        error_reason = "nested_tags_detected"
    else:
        pass

    # Fallback extraction: strip whitespace, then remove tags from ends
    fallback_content = (
        stripped_text
        .removeprefix(open_tag)  # Remove from start if present
        .removesuffix(close_tag)  # Remove from end if present
    )

    # If tags are in the middle (not at ends), use lstrip/rstrip as fallback
    if open_tag in fallback_content or close_tag in fallback_content:
        # More aggressive fallback: find first opening and last closing
        open_idx = stripped_text.find(open_tag)
        close_idx = stripped_text.rfind(close_tag)
        if open_idx != -1 and close_idx != -1 and close_idx > open_idx + len(open_tag):
            fallback_content = stripped_text[open_idx + len(open_tag):close_idx].strip()
        else:
            # Last resort: try lstrip/rstrip (Python 3.9+ str methods)
            temp = stripped_text
            if temp.startswith(open_tag):
                temp = temp[len(open_tag):]
            if temp.endswith(close_tag):
                temp = temp[:-len(close_tag)]
            fallback_content = temp.strip()

    return fallback_content, False, error_reason


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
        lambda_useful: float = 1.0,  # Weight for R_useful (default: 1.0)
        use_logic_reward: bool = True,
        use_useful_reward: bool = True,  # Whether to use R_useful in reward calculation
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
        ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
        too_short_threshold: float = 0.5,  # Deprecated: use too_short_threshold_low/high instead
        too_short_threshold_low: float = 0.4,  # Lower threshold for too_short_penalty
        too_short_threshold_high: float = 0.8,  # Upper threshold for too_short_penalty
        too_short_max_penalty: float = 0.5,  # Maximum penalty for too short outputs
        too_long_threshold_low: float = 1.5,  # Lower threshold for too_long_penalty
        too_long_threshold_high: float = 2.0,  # Upper threshold for too_long_penalty
        too_long_max_penalty: float = 0.5,  # Maximum penalty for too long outputs
        penalty_mode: str = "relative",  # "relative" or "absolute": How to apply penalties
                                        # - "relative": penalty = coefficient * r_cog_range
                                        # - "absolute": penalty = coefficient (fixed value)
        use_tiered_reward: bool = True,  # Whether to use tiered reward for R_cog
        tier_step: float = 0.05,  # 区间大小：每个阶梯的 R_cog 范围
        tier_reward: float = 0.1,  # 每阶梯奖励值：第一阶梯得 tier_reward，第二阶梯得 2*tier_reward
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
                                   # - "ppl": R_cog = avg_prob(rewrite) = 1 / PPL(rewrite) [formula 2: absolute quality]
                                   #           Also logs formula 1: avg_prob(rewrite)/avg_prob(original) - 1 as observation
                                   #           where avg_prob = exp(mean_log_prob), PPL = 1/avg_prob
                                   # - "ppl_delta": R_cog = max(0, avg_prob(rewrite) - avg_prob(original)) (delta with negative clamp only)
                                   # - "ppl_delta_norm_sigmoid": R_cog = sigmoid(normalized_delta / temperature) with group normalization
                                   # - "ppl_ratio": R_cog = min(PPL(original) / PPL(rewrite), ceiling)
                                   # - "cot_effect_ratio": R_cog = min(cot_effect(rewrite) / cot_effect(original), ceiling)
                                   # - "cot_effect_delta_sigmoid": R_cog = sigmoid(max(0, delta) / (|cot_effect(original)| + ε) / τ), delta = avg_prob(rewrite) - avg_prob(original)
                                   # - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
        use_reward_delta: bool = True,  # If True (default), R_cog = rewrite - original (delta).
                                      # If False, R_cog = rewrite only (absolute value).
                                      # Deprecated alias: reward_only_use_logprob (inverted logic!)
        reward_only_use_logprob: bool = None,  # Deprecated: use use_reward_delta instead
                                                # True = use absolute value (no delta), False = use delta
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ppl_original/ppl_rewrite, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        # ========== FSF Reward Configuration ==========
        # FSF reward using build_cot_rewrite_from_sharegpt.py message format
        # GPU devices are auto-detected based on fsf_tensor_parallel_size
        fsf_temperature: float = 0.6,  # Sampling temperature for FSF
        fsf_top_k: int = 20,  # Sampling top-k for FSF
        fsf_top_p: float = 0.95,  # Sampling top-p for FSF
        fsf_max_new_tokens: int = 8000,  # Max new tokens for FSF
        fsf_gpu_memory_utilization: float = 0.4,  # GPU memory for FSF
        fsf_max_model_len: int = 12000,  # Max model len for FSF
        fsf_tensor_parallel_size: int = 1,  # TP size for FSF
        fsf_reward_scale: float = 1.0,  # Scale factor for FSF reward (replaces lambda_logic)
        fsf_micro_batch_size: int = 4,  # Micro batch size for FSF inference (to avoid OOM)
        fsf_skip_group_on_parse_error: bool = True,  # Skip entire group if any FSF parsing fails
        save_fsf_dot_outputs: bool = False,  # Whether to save raw FSF DOT outputs to rollout_data
        fsf_reward_mode: str = 'direct',  # FSF reward calculation mode: 'direct' or 'threshold'
                                         # - 'direct': R_logic = -rewrite_fsf (default)
                                         # - 'threshold': R_logic = -1 if fsf > 0 else 0
        fsf_penalty_mode: str = 'absolute',  # FSF penalty mode: 'absolute' or 'relative'
                                            # - 'absolute': fsf_reward_scale is used directly (default)
                                            # - 'relative': fsf_reward_scale is multiplied by r_cog range
        save_prompts: bool = False,  # Whether to save request prompts for student_model and fsf_judge
        r_logic_repeat_n: int = 1,  # Number of times to repeat FSF calculation for R_logic (default: 1).
                                     # Higher values (e.g., 3) reduce LLM judge volatility by averaging multiple samples.
                                     # R_logic is computed as the average of n independent FSF calculations.
        # ========== Repeat Penalty Configuration ==========
        similarity_threshold: float = 0.9,  # Threshold for repeat penalty (0.0-1.0). If similarity between
                                            # original_cot and rewrite_cot exceeds this, apply penalty.
        similarity_penalty: float = 0.5,  # Penalty coefficient for high similarity (default: 0.5).
                                          # Applied when similarity > similarity_threshold.
        # ========== Correct Judge Configuration ==========
        # Determines if the reasoning process leads to the correct solution
        use_correct_judge: bool = False,  # Whether to use Correct Judge (default: False)
        correct_judge_model_path: str = "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B",  # Model path for Correct Judge
        correct_judge_tensor_parallel_size: int = 1,  # TP size for Correct Judge
        correct_judge_gpu_memory_utilization: float = 0.4,  # GPU memory for Correct Judge
        correct_judge_max_model_len: int = 8192,  # Max model len for Correct Judge
        correct_judge_micro_batch_size: int = 8,  # Micro batch size for Correct Judge inference
        correct_judge_penalty_value: float = 0.0,  # R_cog value when correct judge says "No" (ppl_delta mode: 0)
        correct_judge_max_new_tokens: int = 10,  # Max tokens for Correct Judge output (short output for speed)
        correct_judge_filter_no_samples: bool = False,  # If True, filter out samples where judge says "No" from GRPO training
        **kwargs,  # Ignore deprecated params
    ):
        """
        Initialize Combined Reward Computer.

        Args:
            student_model_path: Path to Student Model
            fsf_model_path: Path to FSF Judge Model (only required if use_logic_reward=True)
            device: Device to run FSF Judge on (Student uses student_devices)
            lambda_logic: Weight for logic reward
            lambda_useful: Weight for R_useful (default: 1.0). Set to 0 to disable R_useful in training.
            use_logic_reward: Whether to use logic reward
            use_useful_reward: Whether to use R_useful in reward calculation (default: True).
                When False, R_useful is computed but not included in total reward.
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
            ppl_on_full_response: If True, compute PPL on full response (Q+CoT+A).
                If False (default), compute PPL on answer portion only.
            too_short_threshold: Deprecated. Use too_short_threshold_low/high instead.
            too_short_threshold_low: Lower threshold for too_short_penalty (default: 0.4). When
                rewrite_ratio < low, penalty is at maximum. Between low and high, penalty scales linearly.
            too_short_threshold_high: Upper threshold for too_short_penalty (default: 0.8). When
                rewrite_ratio >= high, no penalty is applied.
            too_short_max_penalty: Maximum penalty for too short outputs (default: 0.5).
            too_long_threshold_low: Lower threshold for too_long_penalty (default: 1.5). When
                rewrite_ratio >= high, penalty is at maximum. Between low and high, penalty scales linearly.
            too_long_threshold_high: Upper threshold for too_long_penalty (default: 2.0). When
                rewrite_ratio < low, no penalty is applied.
            too_long_max_penalty: Maximum penalty for too long outputs (default: 0.5).
            penalty_mode: How to apply penalty values (default: "relative").
                - "relative": penalty = coefficient * r_cog_range (scales with cognitive reward spread)
                - "absolute": penalty = coefficient (fixed value, independent of rewards)
            use_tiered_reward: Whether to use tiered (step-wise) reward for R_cog (default: True).
            tier_step: 区间大小，每个阶梯的 R_cog 区间 (default: 0.05).
            tier_reward: 每阶梯奖励值，第一阶梯得 tier_reward，第二阶梯得 2*tier_reward (default: 0.1).
            reward_mode: Reward calculation mode (default: "ppl").
                - "ppl": R_cog = avg_prob(rewrite) = 1 / PPL(rewrite) (absolute quality)
                - "ppl_delta_norm_sigmoid": R_cog = max(0, avg_prob(rewrite) - avg_prob(original)) (delta with negative clamp)
                  where avg_prob = exp(mean_log_prob), this rewards improvements and clamps negative deltas to 0
                - "ppl_ratio": R_cog = min(PPL(original) / PPL(rewrite), ceiling)
                  where PPL = exp(-mean_log_prob), ceiling is controlled by ppl_ratio_ceiling (default: 0.95)
                - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original),
                  where answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            ppl_ratio_ceiling: Ceiling for ppl_ratio mode (default: 0.95). When reward_mode="ppl_ratio",
                R_cog = min(PPL(original) / PPL(rewrite), ppl_ratio_ceiling). This prevents the reward
                from growing too large when PPL(rewrite) is much smaller than PPL(original).
            use_reward_delta: If True (default), R_cog = value(rewrite) - value(original), computing
                the improvement delta. If False, R_cog = value(rewrite) only (absolute value).
                For "ppl" mode: delta = (avg_prob(rewrite) - avg_prob(original)) / avg_prob(original)
                For "logprob" mode: delta = logprob(rewrite) - logprob(original) (higher is better)
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead. This parameter has
                inverted logic: True means absolute value (no delta), False means use delta.
            # ========== FSF Parameters ==========
            fsf_temperature: Sampling temperature for FSF (default: 0.6)
            fsf_top_k: Sampling top-k for FSF (default: 20)
            fsf_top_p: Sampling top-p for FSF (default: 0.95)
            fsf_max_new_tokens: Max new tokens for FSF (default: 8000)
            fsf_gpu_memory_utilization: GPU memory fraction for FSF (default: 0.4)
            fsf_max_model_len: Max model len for FSF (default: 12000)
            fsf_tensor_parallel_size: TP size for FSF, GPUs are auto-detected (default: 1)
            fsf_reward_scale: Scale factor for FSF reward (replaces lambda_logic, default: 1.0)
            fsf_micro_batch_size: Micro batch size for FSF inference to avoid OOM (default: 4).
                Lower values reduce memory but increase computation time.
            fsf_skip_group_on_parse_error: If True (default), skip FSF reward for entire group
                when any sample fails to parse. If False, use 0.0 for failed samples.
            r_logic_repeat_n: Number of times to repeat FSF calculation for R_logic (default: 1).
                Higher values (e.g., 3) reduce LLM judge volatility by averaging multiple samples.
                R_logic is computed as the average of n independent FSF calculations.
            similarity_threshold: Threshold for repeat penalty (default: 0.9). If the similarity
                between original_cot and rewrite_cot (computed using SequenceMatcher) exceeds
                this value, a penalty is applied to discourage near-identical rewrites.
            similarity_penalty: Penalty coefficient for high similarity (default: 0.5). Applied
                when similarity > similarity_threshold, scaled by penalty_mode if "relative".
            use_correct_judge: Whether to use Correct Judge to verify if the reasoning leads to
                the correct solution (default: False). When enabled, Correct Judge is loaded
                after Student Model is cleaned up, and checks if rewrite_cot leads to solution.
            correct_judge_model_path: Model path for Correct Judge (default: Qwen3-4B).
            correct_judge_tensor_parallel_size: Tensor parallelism size for Correct Judge (default: 1).
            correct_judge_gpu_memory_utilization: GPU memory fraction for Correct Judge (default: 0.4).
            correct_judge_max_model_len: Maximum sequence length for Correct Judge (default: 8192).
            correct_judge_micro_batch_size: Micro batch size for Correct Judge inference (default: 8).
            correct_judge_penalty_value: R_cog value to use when Correct Judge returns "No"
                (reasoning doesn't lead to solution). Default is 0.0 for ppl_delta mode.
            correct_judge_max_new_tokens: Max tokens for Correct Judge output (default: 10).
                Shorter output speeds up inference. Typical output is <judge>Yes</judge> or <judge>No</judge>.
            correct_judge_filter_no_samples: If True, filter out samples where Correct Judge returns "No"
                from GRPO advantage calculation (default: False). When enabled, samples marked as incorrect
                (is_correct=False) or parse failed (is_correct=None) will have train_mask=False and won't
                participate in GRPO training.

        阶梯式奖励配置示例:
        - tier_step=0.05, tier_reward=0.1:
            0.0-0.05 -> 0.1, 0.05-0.1 -> 0.2, 0.1-0.15 -> 0.3, ...
        - tier_step=0.1, tier_reward=0.05:
            0.0-0.1 -> 0.05, 0.1-0.2 -> 0.1, 0.2-0.3 -> 0.15, ...
        - tier_step=0.03, tier_reward=0.2:
            0.0-0.03 -> 0.2, 0.03-0.06 -> 0.4, 0.06-0.09 -> 0.6, ...
        """
        self.lambda_logic = lambda_logic
        self.lambda_useful = lambda_useful
        self.use_logic_reward = use_logic_reward
        self.use_useful_reward = use_useful_reward
        self.alpha_cog = alpha_cog
        self.reward_micro_batch_size = reward_micro_batch_size
        self.gpu_batch_size = gpu_batch_size
        self.reward_backend = reward_backend
        # Use low/high thresholds directly (no longer need backward compatibility check)
        # The caller should provide too_short_threshold_low/high explicitly
        self.too_short_threshold_low = too_short_threshold_low
        self.too_short_threshold_high = too_short_threshold_high
        self.too_short_max_penalty = too_short_max_penalty
        self.too_long_threshold_low = too_long_threshold_low
        self.too_long_threshold_high = too_long_threshold_high
        self.too_long_max_penalty = too_long_max_penalty
        self.penalty_mode = penalty_mode
        self.use_tiered_reward = use_tiered_reward
        self.tier_step = tier_step
        self.tier_reward = tier_reward
        self.reward_mode = reward_mode
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_temperature

        # Handle reward_only_use_logprob deprecation and map to use_reward_delta
        # reward_only_use_logprob has INVERTED logic compared to use_reward_delta
        # - reward_only_use_logprob=True means absolute value (use_reward_delta=False)
        # - reward_only_use_logprob=False means delta (use_reward_delta=True)
        if reward_only_use_logprob is not None:
            import warnings
            warnings.warn(
                "[DEPRECATION] reward_only_use_logprob is deprecated. Use use_reward_delta instead. "
                "Note: reward_only_use_logprob has inverted logic (True=absolute, False=delta). "
                "Setting use_reward_delta accordingly.",
                DeprecationWarning,
                stacklevel=2
            )
            self.use_reward_delta = not reward_only_use_logprob  # Invert the logic!
        else:
            self.use_reward_delta = use_reward_delta
        self.reward_only_use_logprob = reward_only_use_logprob  # Keep for logging

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
                ppl_on_full_response=ppl_on_full_response,
                ppl_ratio_ceiling=ppl_ratio_ceiling,
                cot_effect_ratio_ceiling=cot_effect_ratio_ceiling,
                cot_effect_epsilon=cot_effect_epsilon,
                cot_effect_temperature=cot_effect_temperature,
                ppl_delta_temperature=ppl_delta_temperature,
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
                ppl_ratio_ceiling=ppl_ratio_ceiling,
                cot_effect_ratio_ceiling=cot_effect_ratio_ceiling,
                cot_effect_epsilon=cot_effect_epsilon,
                cot_effect_temperature=cot_effect_temperature,
                ppl_delta_temperature=ppl_delta_temperature,
            )

        # ========== Initialize FSF Reward ==========
        # FSF Judge is NOT initialized here - it will be created on-demand in compute_reward
        # to ensure proper memory management: student -> cleanup -> FSF -> cleanup
        self.fsf_model_path = fsf_model_path
        self.fsf_temperature = fsf_temperature
        self.fsf_top_k = fsf_top_k
        self.fsf_top_p = fsf_top_p
        self.fsf_max_new_tokens = fsf_max_new_tokens
        self.fsf_gpu_memory_utilization = fsf_gpu_memory_utilization
        self.fsf_max_model_len = fsf_max_model_len
        self.fsf_tensor_parallel_size = fsf_tensor_parallel_size
        self.fsf_reward_scale = fsf_reward_scale  # Scale factor for FSF reward (replaces lambda_logic)
        self.fsf_micro_batch_size = fsf_micro_batch_size  # Micro batch size for FSF inference
        self.fsf_skip_group_on_parse_error = fsf_skip_group_on_parse_error  # Skip group on parse error
        self.save_fsf_dot_outputs = save_fsf_dot_outputs  # Whether to save raw DOT outputs
        self.fsf_reward_mode = fsf_reward_mode  # FSF reward calculation mode: 'direct' or 'threshold'
        self.fsf_penalty_mode = fsf_penalty_mode  # FSF penalty mode: 'absolute' or 'relative'
        self.save_prompts = save_prompts  # Whether to save request prompts for debugging
        self.r_logic_repeat_n = r_logic_repeat_n  # Number of times to repeat FSF calculation for averaging
        self.fsf_judge = None  # Created on-demand in compute_reward

        # ========== Repeat Penalty Parameters ==========
        self.similarity_threshold = similarity_threshold
        self.similarity_penalty = similarity_penalty

        # ========== Correct Judge Parameters ==========
        self.use_correct_judge = use_correct_judge
        self.correct_judge_model_path = correct_judge_model_path
        self.correct_judge_tensor_parallel_size = correct_judge_tensor_parallel_size
        self.correct_judge_gpu_memory_utilization = correct_judge_gpu_memory_utilization
        self.correct_judge_max_model_len = correct_judge_max_model_len
        self.correct_judge_micro_batch_size = correct_judge_micro_batch_size
        self.correct_judge_penalty_value = correct_judge_penalty_value
        self.correct_judge_max_new_tokens = correct_judge_max_new_tokens
        self.correct_judge_filter_no_samples = correct_judge_filter_no_samples
        self.correct_judge = None  # Created on-demand in compute_reward

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        data: Any = None,  # Optional: DataProto for accessing pre-computed FSF values
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
            data: Optional DataProto for accessing pre-computed FSF values from training data

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
            # R_cog (formula 2) = avg_prob(rewrite) = 1 / PPL(rewrite) [absolute quality]
            # Also computes formula 1 = avg_prob(rewrite)/avg_prob(original) - 1 for logging
            # where avg_prob = exp(mean_log_prob), PPL = 1/avg_prob
            pass
        else:  # "logprob" mode
            # R_cog = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, Original_CoT, A)
            # where answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            pass

        # Pass use_reward_delta to student client if supported (negated because old param has inverted logic)
        compute_kwargs = {"alpha_cog": self.alpha_cog}
        if hasattr(self.student_client, 'use_reward_delta'):
            compute_kwargs["use_reward_delta"] = self.use_reward_delta
        # Backward compatibility: also pass old param name if client expects it
        if hasattr(self.student_client, 'reward_only_use_logprob'):
            compute_kwargs["reward_only_use_logprob"] = not self.use_reward_delta  # Invert for old API
        # Save prompts for debugging if requested
        if self.save_prompts:
            compute_kwargs["save_prompts"] = True

        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,
            **compute_kwargs,
        )

        # 获取计算结果
        r_cog_raw = cog_result["rewards"]
        r_useful = cog_result.get("useful_rewards", [0.0] * batch_size)  # R_useful: CoT有用程度
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        no_cot_mean_log_probs = cog_result.get("no_cot_mean_log_probs", [0.0] * batch_size)  # Baseline (no CoT)
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]
        # PPL values (ppl mode only)
        rewrite_ppls = cog_result.get("rewrite_ppls", [float('nan')] * batch_size)
        original_ppls = cog_result.get("original_ppls", [float('nan')] * batch_size)
        # Relative improvement rates (formula 1 observation value)
        relative_improvement_rates = cog_result.get("relative_improvement_rates", [float('nan')] * batch_size)

        # Compute reward_deltas: raw difference in mean log probs (without alpha_cog * min term)
        # This tracks the pure log_prob improvement, excluding the worst-token penalty
        reward_deltas = []
        for i in range(batch_size):
            delta = rewrite_mean_log_probs[i] - original_mean_log_probs[i]  # Pure mean difference
            reward_deltas.append(delta)

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

        # Extract saved_prompts if available (for debugging)
        saved_prompts = cog_result.get("saved_prompts", None) if self.save_prompts else None

        # Compute logic reward (if enabled)
        # FSF is computed on-demand with proper memory management
        # Memory management: student is cleaned up before FSF computation
        r_logic = [0.0] * batch_size

        # === Load pre-computed original_fsf from training data ===
        # Training data has 'fsf' field with pre-computed FSF values
        # We use this as original_fsf instead of computing it via FSF model
        original_fsfs = []
        if data is not None:
            for i in range(batch_size):
                data_item = data[i]
                if 'fsf' in data_item.non_tensor_batch:
                    fsf_val = data_item.non_tensor_batch['fsf']
                    if hasattr(fsf_val, 'item'):
                        fsf_val = fsf_val.item()
                    original_fsfs.append(fsf_val if fsf_val is not None else 0.0)
                else:
                    original_fsfs.append(0.0)
        else:
            # No data provided, initialize all FSF values to 0
            original_fsfs = [0.0] * batch_size

        # Log pre-computed FSF for debugging
        for i, fsf_val in enumerate(original_fsfs):
            if fsf_val != 0.0:
                logger.info(f"[FSF] Sample {i}: pre_computed_original_fsf={fsf_val:.4f}")
        rewrite_fsfs = [0.0] * batch_size
        original_dot_outputs = [None] * batch_size  # Raw DOT outputs for original CoTs
        rewrite_dot_outputs = [None] * batch_size  # Raw DOT outputs for rewrite CoTs
        fsf_parse_failed_count = 0  # Count of samples with parse failures
        fsf_group_skipped = False  # Whether the entire group's FSF reward was skipped

        # Try FSF if enabled
        if self.use_logic_reward:
            if not self.fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")

            # Cleanup student client first to free GPU memory
            logger.info("[FSF] Cleaning up student client before loading FSF model...")
            if hasattr(self.student_client, 'cleanup'):
                self.student_client.cleanup()

            # Create FSF Client on-demand
            from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient
            self.fsf_judge = SGLangFSFClient(
                model_path=self.fsf_model_path,
                mem_fraction_static=self.fsf_gpu_memory_utilization,
                max_model_len=self.fsf_max_model_len,
                tensor_parallel_size=self.fsf_tensor_parallel_size,
                temperature=self.fsf_temperature,
                top_k=self.fsf_top_k,
                top_p=self.fsf_top_p,
                max_new_tokens=self.fsf_max_new_tokens,
                micro_batch_size=self.fsf_micro_batch_size,
            )

            try:
                # Compute rewrite FSF only (original_fsf is loaded from dataset)
                # Support repeated sampling for more stable R_logic
                # OPTIMIZATION: Instead of calling FSF N times, expand the batch and call once
                repeat_n = self.r_logic_repeat_n

                # Expand rewrite_cots by repeat_n: [cot0, cot0, cot0, cot1, cot1, cot1, ...]
                # This allows computing all repeated samples in a single batch call
                expanded_rewrite_cots = []
                sample_to_expanded_map = []  # Maps (sample_idx, run_idx) -> expanded_idx
                expanded_idx = 0
                for sample_idx in range(len(rewrite_cots)):
                    for run_idx in range(repeat_n):
                        expanded_rewrite_cots.append(rewrite_cots[sample_idx])
                        sample_to_expanded_map.append((sample_idx, run_idx))
                        expanded_idx += 1

                if repeat_n > 1:
                    logger.info(f"[FSF] Computing FSF for {len(rewrite_cots)} rewrite CoTs with repeat_n={repeat_n} "
                               f"(expanded to {len(expanded_rewrite_cots)} total requests in single batch, "
                               f"save_dot_outputs={self.save_fsf_dot_outputs}, save_prompts={self.save_prompts})...")
                else:
                    logger.info(f"[FSF] Computing FSF for {len(rewrite_cots)} rewrite CoTs only "
                               f"(original_fsf loaded from dataset, save_dot_outputs={self.save_fsf_dot_outputs}, save_prompts={self.save_prompts})...")

                # Single batch call for all repeated samples
                fsf_result = self.fsf_judge.compute_rewrite_fsf_only(
                    rewrite_cots=expanded_rewrite_cots,
                    return_dot_outputs=self.save_fsf_dot_outputs,
                    save_prompts=self.save_prompts,
                )

                expanded_rewrite_fsfs = fsf_result["rewrite_fsfs"]

                # Extract DOT outputs if needed
                if self.save_fsf_dot_outputs:
                    expanded_dot_outputs = fsf_result.get("rewrite_dot_outputs", [None] * len(expanded_rewrite_cots))

                # Extract saved prompts if needed
                if self.save_prompts and "saved_prompts" in fsf_result:
                    saved_prompts = fsf_result["saved_prompts"]
                    logger.info(f"[FSF] Received saved_prompts keys: {list(saved_prompts.keys())}")

                # Reorganize results: group FSF values by original sample index
                rewrite_fsf_accumulator = [[] for _ in range(batch_size)]
                for expanded_idx, fsf_val in enumerate(expanded_rewrite_fsfs):
                    sample_idx, run_idx = sample_to_expanded_map[expanded_idx]
                    if fsf_val is not None:
                        rewrite_fsf_accumulator[sample_idx].append(fsf_val)

                # Reorganize DOT outputs: save all runs' outputs as a list for each sample
                # When r_logic_repeat_n > 1, collect all DOT outputs for each sample
                # When r_logic_repeat_n == 1, save single output (for backward compatibility)
                rewrite_dot_outputs = [None] * batch_size
                if self.save_fsf_dot_outputs:
                    # Initialize lists for each sample
                    rewrite_dot_outputs_lists = [[] for _ in range(batch_size)]
                    for expanded_idx, dot_output in enumerate(expanded_dot_outputs):
                        sample_idx, run_idx = sample_to_expanded_map[expanded_idx]
                        # Collect all runs' outputs
                        rewrite_dot_outputs_lists[sample_idx].append(dot_output)

                    # Convert to final format:
                    # - If repeat_n > 1: store as list of all runs' outputs
                    # - If repeat_n == 1: store as single value (backward compatibility)
                    for sample_idx in range(batch_size):
                        if repeat_n > 1:
                            # Store all runs as a list
                            rewrite_dot_outputs[sample_idx] = rewrite_dot_outputs_lists[sample_idx]
                            if sample_idx < 3 and rewrite_dot_outputs_lists[sample_idx]:
                                logger.info(
                                    f"[FSF] Sample {sample_idx}: Collected {len(rewrite_dot_outputs_lists[sample_idx])} DOT outputs "
                                    f"(repeat_n={repeat_n}, saved as list to rewrite_dot_outputs)"
                                )
                        else:
                            # Store single value for backward compatibility
                            rewrite_dot_outputs[sample_idx] = rewrite_dot_outputs_lists[sample_idx][0] if rewrite_dot_outputs_lists[sample_idx] else None

                # Compute average FSF for each sample
                rewrite_fsfs = []
                for i, fsf_list in enumerate(rewrite_fsf_accumulator):
                    if len(fsf_list) == 0:
                        # All runs failed for this sample
                        rewrite_fsfs.append(None)
                    else:
                        # Average the FSF values across runs
                        avg_fsf = sum(fsf_list) / len(fsf_list)
                        rewrite_fsfs.append(avg_fsf)
                        if repeat_n > 1 and i < 3:  # Log first few samples
                            logger.info(f"[FSF] Sample {i}: FSF averaged over {len(fsf_list)} runs = {avg_fsf:.4f} (values: {fsf_list})")

                # Count samples with unparseable FSF (all runs failed)
                rewrite_failed = sum(1 for f in rewrite_fsfs if f is None)
                fsf_parse_failed_count = rewrite_failed

                # Compute logic rewards based on fsf_reward_mode
                # original_fsf is only used as observation/metric, not in reward calculation
                # Lower rewrite_fsf = better (fewer failed steps) -> higher reward
                r_logic = []
                for orig_fsf, rew_fsf in zip(original_fsfs, rewrite_fsfs):
                    if rew_fsf is None:
                        # Parse failed (all runs failed), use 0 for logic reward
                        r_logic.append(0.0)
                    else:
                        # Apply reward based on fsf_reward_mode
                        if self.fsf_reward_mode == 'threshold':
                            # R_logic = -1 if fsf > 0 else 0
                            # Only apply penalty if there are any failed steps
                            r_logic.append(-1.0 if rew_fsf > 0 else 0.0)
                        else:  # 'direct' mode (default)
                            # R_logic = -rewrite_fsf (negative reward proportional to failed step fraction)
                            r_logic.append(-rew_fsf)

                has_any_parse_failure = fsf_parse_failed_count > 0

                if has_any_parse_failure and self.fsf_skip_group_on_parse_error:
                    # Skip entire group when any parsing fails (conservative approach)
                    # This means R_logic = 0 for all samples, but other rewards (R_cog, R_useful) are still applied
                    logger.warning(
                        f"[FSF] FSF parsing failed for {fsf_parse_failed_count} rewrite samples. "
                        f"Skipping FSF reward for entire group (fsf_skip_group_on_parse_error=True). "
                        f"R_logic will be 0 for all samples, but R_cog and R_useful are still applied."
                    )
                    # Set all FSF rewards to 0 for group
                    r_logic = [0.0 for _ in range(batch_size)]
                    rewrite_fsfs = [0.0 for _ in range(batch_size)]
                    rewrite_dot_outputs = [None for _ in range(batch_size)]
                    fsf_group_skipped = True
                else:
                    fsf_group_skipped = False
                    if has_any_parse_failure:
                        logger.warning(
                            f"[FSF] FSF parsing failed for {fsf_parse_failed_count} rewrite samples. "
                            f"Using partial results (fsf_skip_group_on_parse_error=False). "
                            f"Failed samples have R_logic=0."
                        )

                # Log FSF comparison statistics
                valid_pairs = [(o, r) for o, r in zip(original_fsfs, rewrite_fsfs) if r is not None]
                if valid_pairs:
                    improvements = sum(1 for o, r in valid_pairs if r < o)
                    logger.info(
                        f"[FSF] Statistics: {len(valid_pairs)} valid pairs, "
                        f"{improvements} improvements (rewrite_fsf < original_fsf), "
                        f"mean original_fsf={sum(o for o, _ in valid_pairs)/len(valid_pairs):.4f}, "
                        f"mean rewrite_fsf={sum(r for _, r in valid_pairs)/len(valid_pairs):.4f}"
                    )
            finally:
                # FSF Judge cleanup - shutdown workers after compute_logic_reward
                if self.fsf_judge is not None:
                    self.fsf_judge.shutdown()
                    self.fsf_judge = None

        # ========== Correct Judge Computation ==========
        # Correct Judge is computed after FSF (if enabled) with proper memory management
        # Memory management: student -> FSF -> cleanup -> Correct Judge -> cleanup
        is_correct_results = [True] * batch_size  # Default: all correct (no penalty)
        correct_judge_parse_failed_count = 0  # Count of samples with parse failures

        if self.use_correct_judge:
            # Create Correct Judge Client on-demand
            from verl.experimental.reward.cot_rewriter.sglang_correct_judge import SGLangCorrectJudgeClient

            logger.info("[CORRECT_JUDGE] Creating Correct Judge client after Student Model cleanup...")
            self.correct_judge = SGLangCorrectJudgeClient(
                model_path=self.correct_judge_model_path,
                tensor_parallel_size=self.correct_judge_tensor_parallel_size,
                mem_fraction_static=self.correct_judge_gpu_memory_utilization,
                max_model_len=self.correct_judge_max_model_len,
                micro_batch_size=self.correct_judge_micro_batch_size,
                max_new_tokens=self.correct_judge_max_new_tokens,
            )

            try:
                logger.info(f"[CORRECT_JUDGE] Computing correct judge for {len(questions)} samples "
                           f"(save_prompts={self.save_prompts})...")

                judge_result = self.correct_judge.compute_correct_judge(
                    questions=questions,
                    rewrite_cots=rewrite_cots,
                    solutions=answers,
                    save_prompts=self.save_prompts,
                )

                is_correct_results = judge_result["is_correct"]

                # Extract saved prompts if needed
                if self.save_prompts and "saved_prompts" in judge_result:
                    if saved_prompts is None:
                        saved_prompts = judge_result["saved_prompts"]
                    else:
                        # Merge saved prompts
                        for key, values in judge_result["saved_prompts"].items():
                            if key not in saved_prompts:
                                saved_prompts[key] = []
                            saved_prompts[key].extend(values)

                # Count parse failures and apply penalty
                correct_judge_parse_failed_count = sum(1 for x in is_correct_results if x is None)

                # Log statistics
                correct_count = sum(1 for x in is_correct_results if x is True)
                incorrect_count = sum(1 for x in is_correct_results if x is False)
                logger.info(
                    f"[CORRECT_JUDGE] Statistics: "
                    f"{correct_count} correct, {incorrect_count} incorrect, "
                    f"{correct_judge_parse_failed_count} parse failures"
                )

                # Log first few samples
                for i in range(min(3, batch_size)):
                    if is_correct_results[i] is not None:
                        result_str = "correct" if is_correct_results[i] else "incorrect"
                        logger.info(f"[CORRECT_JUDGE] Sample {i}: {result_str}")

            finally:
                # Correct Judge cleanup - shutdown workers after computation
                if self.correct_judge is not None:
                    self.correct_judge.shutdown()
                    self.correct_judge = None

        # Compute too_short_penalty and too_long_penalty based on length ratio
        # Penalty formula with low/high thresholds:
        #   ratio = rewrite_token_len / original_token_len
        # too_short_penalty (ratio < too_short_threshold_high):
        #   - If ratio >= too_short_threshold_high: penalty_coefficient = 0
        #   - If too_short_threshold_low <= ratio < too_short_threshold_high: penalty_coefficient = max_penalty * (high - ratio) / (high - low)
        #   - If ratio < too_short_threshold_low: penalty_coefficient = max_penalty
        # too_long_penalty (ratio > too_long_threshold_low):
        #   - If ratio <= too_long_threshold_low: penalty_coefficient = 0
        #   - If too_long_threshold_low < ratio <= too_long_threshold_high: penalty_coefficient = max_penalty * (ratio - low) / (high - low)
        #   - If ratio > too_long_threshold_high: penalty_coefficient = max_penalty
        too_short_penalties = [0.0] * batch_size
        too_short_penalty_coeffs = [0.0] * batch_size  # Coefficients before applying penalty_mode
        too_long_penalties = [0.0] * batch_size
        too_long_penalty_coeffs = [0.0] * batch_size  # Coefficients before applying penalty_mode
        length_ratios = [0.0] * batch_size  # Track length ratios for DAPO filter_groups

        # Get tokenizer from student_client for token length calculation
        tokenizer = getattr(self.student_client, 'tokenizer', None)
        logger.info(f"[LENGTH_PENALTY_DEBUG] tokenizer available: {tokenizer is not None}, student_client type: {type(self.student_client).__name__}")
        if tokenizer is None:
            logger.warning("[LENGTH_PENALTY] Tokenizer not available, skipping length penalties")
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

                        # Compute too_short_penalty coefficient
                        low_short = self.too_short_threshold_low
                        high_short = self.too_short_threshold_high

                        if ratio >= high_short:
                            # No penalty when ratio is above high threshold
                            short_penalty_coeff = 0.0
                        elif ratio < low_short:
                            # Maximum penalty coefficient when ratio is below low threshold
                            short_penalty_coeff = self.too_short_max_penalty
                        else:
                            # Linear penalty between low and high: 0 at high, max_coeff at low
                            short_penalty_coeff = self.too_short_max_penalty * (high_short - ratio) / (high_short - low_short)

                        too_short_penalty_coeffs[i] = short_penalty_coeff
                        if short_penalty_coeff != 0 and i < 3:  # Log first few samples
                            logger.info(
                                f"[TOO_SHORT_PENALTY] Sample {i}: "
                                f"original_token_len={original_token_len}, rewrite_token_len={rewrite_token_len}, "
                                f"ratio={ratio:.3f} (low={low_short}, high={high_short}), penalty_coeff={short_penalty_coeff:.4f}"
                            )

                        # Compute too_long_penalty coefficient
                        low_long = self.too_long_threshold_low
                        high_long = self.too_long_threshold_high

                        if ratio <= low_long:
                            # No penalty when ratio is below or equal to low threshold
                            long_penalty_coeff = 0.0
                        elif ratio > high_long:
                            # Maximum penalty coefficient when ratio is above high threshold
                            long_penalty_coeff = self.too_long_max_penalty
                        else:
                            # Linear penalty between low and high: 0 at low, max_coeff at high
                            long_penalty_coeff = self.too_long_max_penalty * (ratio - low_long) / (high_long - low_long)

                        too_long_penalty_coeffs[i] = long_penalty_coeff
                        if long_penalty_coeff != 0 and i < 3:  # Log first few samples
                            logger.info(
                                f"[TOO_LONG_PENALTY] Sample {i}: "
                                f"original_token_len={original_token_len}, rewrite_token_len={rewrite_token_len}, "
                                f"ratio={ratio:.3f} (low={low_long}, high={high_long}), penalty_coeff={long_penalty_coeff:.4f}"
                            )

        # Log length_ratio statistics for DAPO filter_groups
        valid_ratios = [r for r in length_ratios if r > 0]
        if valid_ratios:
            low_short = self.too_short_threshold_low
            high_short = self.too_short_threshold_high
            low_long = self.too_long_threshold_low
            high_long = self.too_long_threshold_high
            num_above_low_short = sum(1 for r in valid_ratios if r >= low_short)
            num_above_high_short = sum(1 for r in valid_ratios if r >= high_short)
            num_above_low_long = sum(1 for r in valid_ratios if r > low_long)
            num_above_high_long = sum(1 for r in valid_ratios if r > high_long)
            logger.info(
                f"[LENGTH_RATIO] batch_size={batch_size}, "
                f"mean={sum(valid_ratios)/len(valid_ratios):.3f}, "
                f"min={min(valid_ratios):.3f}, max={max(valid_ratios):.3f}, "
                f"too_short: low={low_short}, high={high_short} (num_above_low={num_above_low_short}/{len(valid_ratios)}, num_above_high={num_above_high_short}/{len(valid_ratios)}), "
                f"too_long: low={low_long}, high={high_long} (num_above_low={num_above_low_long}/{len(valid_ratios)}, num_above_high={num_above_high_long}/{len(valid_ratios)})"
            )

        # Log summary of penalty coefficients for debugging
        logger.info(
            f"[PENALTY_SUMMARY] batch_size={batch_size}, "
            f"too_short_penalty_coeffs (first 3): {too_short_penalty_coeffs[:3]}, "
            f"too_long_penalty_coeffs (first 3): {too_long_penalty_coeffs[:3]}, "
            f"too_short_penalties (first 3): {too_short_penalties[:3]}, "
            f"too_long_penalties (first 3): {too_long_penalties[:3]}"
        )

        # Compute r_cog_range (used for both FSF and length penalty scaling in relative mode)
        r_cog_range = max(r_cog) - min(r_cog) if len(r_cog) > 1 else 1.0
        r_cog_range = max(r_cog_range, 1e-6)  # Avoid zero or extremely small ranges
        logger.info(
            f"[R_COG_RANGE] r_cog_range={r_cog_range:.4f} (min={min(r_cog):.4f}, max={max(r_cog):.4f})"
        )

        # Compute FSF reward scale based on fsf_penalty_mode
        # If fsf_penalty_mode == "relative", scale fsf_reward_scale by r_cog range
        if self.fsf_penalty_mode == "relative":
            fsf_scale = self.fsf_reward_scale * r_cog_range
            logger.info(
                f"[FSF_PENALTY_MODE] relative, fsf_reward_scale={self.fsf_reward_scale:.4f}, "
                f"actual_fsf_scale={fsf_scale:.4f}"
            )
        else:  # fsf_penalty_mode == "absolute" (default)
            fsf_scale = self.fsf_reward_scale
            logger.info(
                f"[FSF_PENALTY_MODE] absolute, fsf_reward_scale={self.fsf_reward_scale:.4f}"
            )

        # ========== Apply Correct Judge Penalty ==========
        # If Correct Judge returns False (incorrect), force R_cog to penalty value
        r_cog_before_correct_judge = r_cog.copy()  # Keep copy for logging
        correct_judge_penalties = [0.0] * batch_size  # Track penalties for logging

        if self.use_correct_judge:
            for i in range(batch_size):
                if is_correct_results[i] is False:
                    # Incorrect: force R_cog to penalty value
                    original_r_cog = r_cog[i]
                    r_cog[i] = self.correct_judge_penalty_value
                    penalty = original_r_cog - self.correct_judge_penalty_value
                    correct_judge_penalties[i] = penalty
                    if i < 3:  # Log first few samples
                        logger.info(
                            f"[CORRECT_JUDGE] Sample {i}: Incorrect! "
                            f"R_cog forced from {original_r_cog:.4f} to {self.correct_judge_penalty_value:.4f} "
                            f"(penalty={penalty:.4f})"
                        )
                elif is_correct_results[i] is None:
                    # Parse failure: treat as incorrect
                    original_r_cog = r_cog[i]
                    r_cog[i] = self.correct_judge_penalty_value
                    penalty = original_r_cog - self.correct_judge_penalty_value
                    correct_judge_penalties[i] = penalty
                    if i < 3:  # Log first few samples
                        logger.info(
                            f"[CORRECT_JUDGE] Sample {i}: Parse failure! "
                            f"R_cog forced from {original_r_cog:.4f} to {self.correct_judge_penalty_value:.4f} "
                            f"(penalty={penalty:.4f})"
                        )

        # Compute base rewards: R_base = R_cog + lambda_useful * R_useful + fsf_scale * R_logic
        # Note: R_useful is only included if use_useful_reward is True. fsf_scale is computed based on fsf_penalty_mode.
        if self.use_useful_reward:
            base_rewards = [c + self.lambda_useful * u + fsf_scale * l
                           for c, u, l in zip(r_cog, r_useful, r_logic)]
        else:
            base_rewards = [c + fsf_scale * l
                           for c, l in zip(r_cog, r_logic)]

        # Determine penalty scaling factor based on penalty_mode
        # DEBUG: Log penalty_mode value to track unexpected "s" value
        logger.info(f"[PENALTY_MODE_DEBUG] self.penalty_mode={repr(self.penalty_mode)}, type={type(self.penalty_mode)}")
        if self.penalty_mode == "relative":
            # Use r_cog_range for relative penalty scaling
            penalty_scale = r_cog_range
            logger.info(
                f"[PENALTY_MODE] relative, batch_size={batch_size}, "
                f"penalty_scale=r_cog_range={penalty_scale:.4f}"
            )
        else:  # penalty_mode == "absolute"
            # Use absolute penalties (coefficient is the actual penalty value)
            penalty_scale = 1.0  # No scaling
            logger.info(
                f"[PENALTY_MODE] absolute, batch_size={batch_size}, "
                f"penalties applied as fixed values (no scaling)"
            )

        # Apply penalties based on penalty_mode
        # too_short_penalties/too_long_penalties store coefficients for logging
        # too_short_penalties_actual/too_long_penalties_actual store actual penalty values
        too_short_penalties_actual = [0.0] * batch_size
        too_long_penalties_actual = [0.0] * batch_size
        for i in range(batch_size):
            too_short_penalties_actual[i] = -too_short_penalty_coeffs[i] * penalty_scale
            too_short_penalties[i] = too_short_penalty_coeffs[i]  # Log coefficient
            if too_short_penalty_coeffs[i] != 0:
                logger.info(
                    f"[TOO_SHORT_PENALTY] Sample {i}: "
                    f"penalty_coeff={too_short_penalty_coeffs[i]:.4f} * {penalty_scale:.4f} = {too_short_penalties_actual[i]:.4f} ({self.penalty_mode})"
                )

            too_long_penalties_actual[i] = -too_long_penalty_coeffs[i] * penalty_scale
            too_long_penalties[i] = too_long_penalty_coeffs[i]  # Log coefficient
            if too_long_penalty_coeffs[i] != 0:
                logger.info(
                    f"[TOO_LONG_PENALTY] Sample {i}: "
                    f"penalty_coeff={too_long_penalty_coeffs[i]:.4f} * {penalty_scale:.4f} = {too_long_penalties_actual[i]:.4f} ({self.penalty_mode})"
                )

        # ========== Compute Repeat Penalty ==========
        # Calculate similarity between original_cot and rewrite_cot using SequenceMatcher
        # Apply penalty if similarity > similarity_threshold
        repeat_penalties_actual = [0.0] * batch_size
        repeat_penalty_coeffs = [0.0] * batch_size  # Coefficients for logging
        similarities = [0.0] * batch_size  # Store similarity values for logging

        for i in range(batch_size):
            original_text = original_cots[i] if original_cots[i] else ""
            rewrite_text = rewrite_cots[i] if rewrite_cots[i] else ""

            if original_text and rewrite_text:
                # Calculate similarity using SequenceMatcher
                sm = SequenceMatcher(None, original_text, rewrite_text)
                blocks = sm.get_matching_blocks()
                total_match = sum(block.size for block in blocks)
                similarity = 2 * total_match / (len(original_text) + len(rewrite_text))
                similarities[i] = similarity

                # Apply penalty if similarity exceeds threshold
                if similarity > self.similarity_threshold:
                    repeat_penalty_coeffs[i] = self.similarity_penalty
                    repeat_penalties_actual[i] = -self.similarity_penalty * penalty_scale

                    if i < 3:  # Log first few samples
                        logger.info(
                            f"[REPEAT_PENALTY] Sample {i}: "
                            f"similarity={similarity:.4f} > threshold={self.similarity_threshold}, "
                            f"penalty_coeff={self.similarity_penalty:.4f} -> actual={repeat_penalties_actual[i]:.4f} ({self.penalty_mode})"
                        )
                elif i < 3 and similarity > 0.7:  # Log high similarity warnings
                    logger.info(
                        f"[REPEAT_PENALTY] Sample {i}: "
                        f"similarity={similarity:.4f} <= threshold={self.similarity_threshold}, no penalty applied"
                    )

        # Log repeat penalty statistics
        high_sim_count = sum(1 for s in similarities if s > self.similarity_threshold)
        if high_sim_count > 0:
            logger.info(
                f"[REPEAT_PENALTY] batch_size={batch_size}, "
                f"{high_sim_count} samples with similarity > {self.similarity_threshold}, "
                f"mean_similarity={sum(similarities)/len(similarities):.4f}, "
                f"max_similarity={max(similarities):.4f}"
            )

        # Combine rewards: R = R_base + too_short_penalty + too_long_penalty + repeat_penalty
        total_rewards = [b + s + l + r for b, s, l, r in zip(base_rewards, too_short_penalties_actual, too_long_penalties_actual, repeat_penalties_actual)]

        # Return penalty coefficients AND actual penalty values for use in format/truncation penalty calculation
        # Store in extra_info for external use
        extra_info_penalty_coeffs = {
            "base_rewards": base_rewards,
            "r_cog_range": r_cog_range,
            "penalty_scale": penalty_scale,
            "penalty_mode": self.penalty_mode,
            "too_short_penalty_coeffs": too_short_penalty_coeffs,
            "too_long_penalty_coeffs": too_long_penalty_coeffs,
            # Actual penalty values applied (negative values)
            "too_short_penalties_actual": too_short_penalties_actual,
            "too_long_penalties_actual": too_long_penalties_actual,
            # Repeat penalty
            "repeat_penalty_coeffs": repeat_penalty_coeffs,
            "repeat_penalties_actual": repeat_penalties_actual,
            "similarities": similarities,
        }

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

        # ========== Generate train_mask based on correct_judge_filter_no_samples ==========
        # train_mask: True = sample participates in GRPO training, False = sample is filtered out
        if self.correct_judge_filter_no_samples:
            train_mask = []
            filtered_count = 0
            for i in range(batch_size):
                # Filter out samples where judge says "No" (False) or parse failed (None)
                if is_correct_results[i] is False or is_correct_results[i] is None:
                    train_mask.append(False)
                    filtered_count += 1
                else:
                    train_mask.append(True)
            logger.info(
                f"[CORRECT_JUDGE_FILTER] Filtered {filtered_count}/{batch_size} samples "
                f"({100*filtered_count/batch_size:.1f}%) from GRPO training"
            )
        else:
            # No filtering: all samples participate
            train_mask = [True] * batch_size

        return {
            "cognitive_rewards": r_cog,
            "useful_rewards": r_useful,  # R_useful: CoT有用程度
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "no_cot_mean_log_probs": no_cot_mean_log_probs,  # Baseline (no CoT)
            "rewrite_ppls": rewrite_ppls,  # PPL values for rewrite (ppl mode only)
            "original_ppls": original_ppls,  # PPL values for original (ppl mode only)
            "relative_improvement_rates": relative_improvement_rates,  # Formula 1: avg_prob(rewrite)/avg_prob(original) - 1
            "logic_rewards": r_logic,
            "total_rewards": total_rewards,
            "token_log_probs": token_log_probs,
            "token_ids": token_ids,
            "reward_deltas": reward_deltas,  # NEW: Log raw rewrite-original deltas
            "original_fsfs": original_fsfs,  # FSF values for original CoTs (pre-computed from dataset)
            "rewrite_fsfs": rewrite_fsfs,  # FSF values for rewrite CoTs (computed by FSF model)
            # Note: original_dot_outputs is omitted (original cot does not call FSF judge)
            "fsf_parse_failed_count": fsf_parse_failed_count,  # Number of samples with FSF parse failures
            "fsf_group_skipped": fsf_group_skipped,  # Whether FSF reward was skipped for entire group
            # === Correct Judge Keys ===
            "is_correct_results": is_correct_results,  # Correct Judge results (True=correct, False=incorrect, None=parse failure)
            "correct_judge_parse_failed_count": correct_judge_parse_failed_count,  # Number of samples with parse failures
            "r_cog_before_correct_judge": r_cog_before_correct_judge,  # R_cog values before Correct Judge penalty
            "correct_judge_penalties": correct_judge_penalties,  # Penalties applied by Correct Judge
            "train_mask": train_mask,  # Training mask: True=participate in GRPO, False=filtered out (when correct_judge_filter_no_samples=True)
            # === TOP-LEVEL KEYS for compute_cot_rewriter_reward to access ===
            "too_short_penalties": too_short_penalties,  # Penalty coefficients (0-0.5)
            "too_long_penalties": too_long_penalties,  # Penalty coefficients (0-0.5)
            "repeat_penalties": repeat_penalty_coeffs,  # Repeat penalty coefficients
            # === NESTED extra_info ===
            "extra_info": {
                "lambda_logic": self.lambda_logic,
                "lambda_useful": self.lambda_useful,
                "too_short_penalties": too_short_penalties,
                "too_short_penalty_coeffs": too_short_penalty_coeffs,  # Coefficients for logging
                "too_long_penalties": too_long_penalties,  # NEW: too_long_penalty coefficients
                "too_long_penalty_coeffs": too_long_penalty_coeffs,  # NEW: too_long_penalty coefficients
                "repeat_penalties": repeat_penalty_coeffs,  # Repeat penalty coefficients
                "repeat_penalties_actual": repeat_penalties_actual,  # Actual repeat penalties applied
                "similarities": similarities,  # Similarity values between original and rewrite
                "length_ratios": length_ratios,  # For DAPO filter_groups
                # Penalty info for external use (format/truncation penalties)
                "penalty_mode": self.penalty_mode,  # "relative" or "absolute"
                "base_rewards": base_rewards,
                "r_cog_range": r_cog_range,  # Range of cognitive rewards
                "penalty_scale": penalty_scale,  # Actual scaling factor used for penalties
                # === DEBUG: Pre-computed FSF from training data ===
                "pre_computed_original_fsfs": original_fsfs,  # Pre-computed FSF for original CoTs (from training data)
                # === FSF DOT outputs (for saving to rollout_data/val_data) ===
                "rewrite_dot_outputs": rewrite_dot_outputs,  # Raw DOT outputs for rewrite CoTs (if save_fsf_dot_outputs=True)
                # === Saved prompts (for debugging) ===
                "saved_prompts": saved_prompts,  # Request prompts for student_model (if save_prompts=True)
            }
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
            # FSF Judge cleanup - properly shutdown to release GPU memory
            _reward_computer.fsf_judge.shutdown()
            _reward_computer.fsf_judge = None
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
    use_useful_reward: bool = True,  # Whether to use R_useful in reward calculation
    alpha_cog: float = 0.01,
    alpha: Optional[float] = None,  # Alias for alpha_cog (from config)
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
    ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
    truncation_penalty: float = 0.5,  # Penalty for truncated responses (configured as positive)
    format_penalty: float = 0.5,  # Penalty for format violations (configured as positive)
    too_short_threshold: float = 0.5,  # Deprecated: use too_short_threshold_low/high instead
    too_short_threshold_low: float = 0.4,  # Lower threshold for too_short_penalty
    too_short_threshold_high: float = 0.8,  # Upper threshold for too_short_penalty
    too_short_max_penalty: float = 0.5,  # Maximum penalty for too short outputs
    too_long_threshold_low: float = 1.5,  # Lower threshold for too_long_penalty
    too_long_threshold_high: float = 2.0,  # Upper threshold for too_long_penalty
    too_long_max_penalty: float = 0.5,  # Maximum penalty for too long outputs
    penalty_mode: str = "relative",  # "relative" or "absolute": How to apply penalties
    use_tiered_reward: bool = True,  # Whether to use tiered (step-wise) reward for R_cog
    tier_step: float = 0.05,  # 区间大小：每个阶梯的 R_cog 范围
    tier_reward: float = 0.1,  # 每阶梯奖励值：第一阶梯得 tier_reward，第二阶梯得 2*tier_reward
    reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
    ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
    cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
    cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
    cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
    ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
    use_reward_delta: bool = True,  # If True (default), compute delta (rewrite - original).
                                       # If False, use absolute value.
                                       # Deprecated alias: reward_only_use_logprob (inverted logic!)
    reward_only_use_logprob: bool = None,  # Deprecated: use use_reward_delta instead
    # ========== FSF Reward Parameters ==========
    # GPU devices are auto-detected based on fsf_tensor_parallel_size
    fsf_temperature: float = 0.6,  # Sampling temperature for FSF
    fsf_top_k: int = 20,  # Sampling top-k for FSF
    fsf_top_p: float = 0.95,  # Sampling top-p for FSF
    fsf_max_new_tokens: int = 8000,  # Max new tokens for FSF
    fsf_gpu_memory_utilization: float = 0.4,  # GPU memory for FSF
    fsf_max_model_len: int = 12000,  # Max model len for FSF
    fsf_tensor_parallel_size: int = 1,  # TP size for FSF (GPUs auto-detected)
    fsf_reward_scale: float = 1.0,  # Scale factor for FSF reward (replaces lambda_logic)
    fsf_micro_batch_size: int = 4,  # Micro batch size for FSF inference (avoid OOM)
    fsf_skip_group_on_parse_error: bool = True,  # Skip entire group on parse error
    save_fsf_dot_outputs: bool = False,  # Whether to save raw FSF DOT outputs to rollout_data
    fsf_reward_mode: str = 'direct',  # FSF reward mode: 'direct' or 'threshold'
    fsf_penalty_mode: str = 'absolute',  # FSF penalty mode: 'absolute' or 'relative'
                                      # - 'absolute': fsf_reward_scale is used directly (default)
                                      # - 'relative': fsf_reward_scale is multiplied by r_cog range
    save_prompts: bool = False,  # Whether to save request prompts for student_model
    r_logic_repeat_n: int = 1,  # Number of times to repeat FSF calculation for R_logic (default: 1).
                                # Higher values (e.g., 3) reduce LLM judge volatility by averaging multiple samples.
    # ========== Repeat Penalty Parameters ==========
    similarity_threshold: float = 0.9,  # Threshold for repeat penalty (0.0-1.0). If similarity between
                                        # original_cot and rewrite_cot exceeds this, apply penalty.
    similarity_penalty: float = 0.5,  # Penalty coefficient for high similarity (default: 0.5).
                                      # Applied when similarity > similarity_threshold.
    # ========== Correct Judge Parameters ==========
    use_correct_judge: bool = False,  # Whether to use Correct Judge (default: False)
    correct_judge_model_path: str = "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B",  # Model path for Correct Judge
    correct_judge_tensor_parallel_size: int = 1,  # TP size for Correct Judge
    correct_judge_gpu_memory_utilization: float = 0.4,  # GPU memory for Correct Judge
    correct_judge_max_model_len: int = 8192,  # Max model len for Correct Judge
    correct_judge_micro_batch_size: int = 8,  # Micro batch size for Correct Judge inference
    correct_judge_penalty_value: float = 0.0,  # R_cog value when correct judge says "No" (ppl_delta mode: 0)
    correct_judge_max_new_tokens: int = 10,  # Max tokens for Correct Judge output (short output for speed)
    correct_judge_filter_no_samples: bool = False,  # If True, filter out samples where judge says "No" from GRPO training
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
        fsf_reward_scale: Scale factor for FSF reward (replaces lambda_logic, default: 1.0)
        fsf_micro_batch_size: Micro batch size for FSF inference to avoid OOM (default: 4)
        fsf_skip_group_on_parse_error: If True (default), skip entire group when any FSF parse fails
        r_logic_repeat_n: Number of times to repeat FSF calculation for R_logic (default: 1).
            Higher values (e.g., 3) reduce LLM judge volatility by averaging multiple samples.
            R_logic is computed as the average of n independent FSF calculations.
        similarity_threshold: Threshold for repeat penalty (default: 0.9). If the similarity
            between original_cot and rewrite_cot (computed using SequenceMatcher) exceeds
            this value, a penalty is applied to discourage near-identical rewrites.
        similarity_penalty: Penalty coefficient for high similarity (default: 0.5). Applied
            when similarity > similarity_threshold, scaled by penalty_mode if "relative".
        use_correct_judge: Whether to use Correct Judge to verify if the reasoning leads to
            the correct solution (default: False). When enabled, Correct Judge is loaded
            after Student Model is cleaned up, and checks if rewrite_cot leads to solution.
        correct_judge_model_path: Model path for Correct Judge (default: Qwen3-4B).
        correct_judge_tensor_parallel_size: Tensor parallelism size for Correct Judge (default: 1).
        correct_judge_gpu_memory_utilization: GPU memory fraction for Correct Judge (default: 0.4).
        correct_judge_max_model_len: Maximum sequence length for Correct Judge (default: 8192).
        correct_judge_micro_batch_size: Micro batch size for Correct Judge inference (default: 8).
        correct_judge_penalty_value: R_cog value to use when Correct Judge returns "No"
            (reasoning doesn't lead to solution). Default is 0.0 for ppl_delta mode.
        correct_judge_max_new_tokens: Max tokens for Correct Judge output (default: 10).
            Shorter output speeds up inference. Typical output is <judge>Yes</judge> or <judge>No</judge>.
        correct_judge_filter_no_samples: If True, filter out samples where Correct Judge returns "No"
            from GRPO advantage calculation (default: False). When enabled, samples marked as incorrect
            (is_correct=False) or parse failed (is_correct=None) will have train_mask=False and won't
            participate in GRPO training. The train_mask is returned in rollout_data for use by the trainer.

    Returns:
        Dictionary with:
            - reward_tensor: Token-level reward tensor [batch_size, seq_len]
            - reward_extra_info: Additional metrics for logging
    """
    # Fixed field names (standardized in rl_dataset_jsonl.py)
    prompt_key = "prompt"
    reward_model_key = "reward_model"
    question_key = "question"
    original_cot_key = "original_cot"
    ground_truth_key = "ground_truth"

    # Check if using flat structure (no reward_model wrapper)
    use_flat_structure = False

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

        # NEW: Extract pre-computed FSF from training data (for debugging)
        original_fsf = None
        if "fsf" in reward_info:
            original_fsf = reward_info.get("fsf", None)
        elif "fsf" in data_item.non_tensor_batch:
            fsf_data = data_item.non_tensor_batch["fsf"]
            if fsf_data is not None:
                if hasattr(fsf_data, 'item'):
                    original_fsf = fsf_data.item()
                else:
                    original_fsf = fsf_data

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
        # Always use extracted content (strict or fallback), format_valid_flag controls penalty only
        extracted_rewrite_cots.append(extracted_cot)
        format_valid_flags.append(1 if has_valid_format else 0)
        if not has_valid_format:
            stripped_preview = rewrite_cot.strip()[:100] if rewrite_cot else ""
            extracted_preview = extracted_cot[:100] if extracted_cot else ""
            logger.info(
                f"[REWARD] Sample {i}: Format violation - reason={error_reason}. "
                f"Preview: '{stripped_preview}...'. "
                f"Using fallback extraction: '{extracted_preview}...' (format penalty will be applied)"
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
        use_useful_reward=use_useful_reward,
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
        ppl_on_full_response=ppl_on_full_response,
        too_short_threshold=too_short_threshold,
        too_short_threshold_low=too_short_threshold_low,
        too_short_threshold_high=too_short_threshold_high,
        too_short_max_penalty=too_short_max_penalty,
        too_long_threshold_low=too_long_threshold_low,
        too_long_threshold_high=too_long_threshold_high,
        too_long_max_penalty=too_long_max_penalty,
        penalty_mode=penalty_mode,
        use_tiered_reward=use_tiered_reward,
        tier_step=tier_step,
        tier_reward=tier_reward,
        reward_mode=reward_mode,
        ppl_ratio_ceiling=ppl_ratio_ceiling,
        cot_effect_ratio_ceiling=cot_effect_ratio_ceiling,
        cot_effect_epsilon=cot_effect_epsilon,
        cot_effect_temperature=cot_effect_temperature,
        ppl_delta_temperature=ppl_delta_temperature,
        use_reward_delta=use_reward_delta,
        reward_only_use_logprob=reward_only_use_logprob,  # For backward compatibility
        # ========== FSF Parameters ==========
        fsf_temperature=fsf_temperature,
        fsf_top_k=fsf_top_k,
        fsf_top_p=fsf_top_p,
        fsf_max_new_tokens=fsf_max_new_tokens,
        fsf_gpu_memory_utilization=fsf_gpu_memory_utilization,
        fsf_max_model_len=fsf_max_model_len,
        fsf_tensor_parallel_size=fsf_tensor_parallel_size,
        fsf_reward_scale=fsf_reward_scale,
        fsf_micro_batch_size=fsf_micro_batch_size,
        fsf_skip_group_on_parse_error=fsf_skip_group_on_parse_error,
        save_fsf_dot_outputs=save_fsf_dot_outputs,
        fsf_reward_mode=fsf_reward_mode,
        fsf_penalty_mode=fsf_penalty_mode,
        save_prompts=save_prompts,
        r_logic_repeat_n=r_logic_repeat_n,
        # ========== Repeat Penalty Parameters ==========
        similarity_threshold=similarity_threshold,
        similarity_penalty=similarity_penalty,
        # ========== Correct Judge Parameters ==========
        use_correct_judge=use_correct_judge,
        correct_judge_model_path=correct_judge_model_path,
        correct_judge_tensor_parallel_size=correct_judge_tensor_parallel_size,
        correct_judge_gpu_memory_utilization=correct_judge_gpu_memory_utilization,
        correct_judge_max_model_len=correct_judge_max_model_len,
        correct_judge_micro_batch_size=correct_judge_micro_batch_size,
        correct_judge_penalty_value=correct_judge_penalty_value,
        correct_judge_max_new_tokens=correct_judge_max_new_tokens,
        correct_judge_filter_no_samples=correct_judge_filter_no_samples,
    )

    # Compute rewards for all samples
    result = reward_computer.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=extracted_rewrite_cots,
        answers=answers,
        data=data,
    )

    # Fill in results for all samples
    cognitive_rewards = result["cognitive_rewards"]
    useful_rewards = result.get("useful_rewards", [0.0] * batch_size)  # R_useful: CoT有用程度
    rewrite_mean_log_probs = result["rewrite_mean_log_probs"]
    original_mean_log_probs = result["original_mean_log_probs"]
    no_cot_mean_log_probs = result.get("no_cot_mean_log_probs", [0.0] * batch_size)  # Baseline (no CoT)
    logic_rewards = result["logic_rewards"]
    total_rewards = result["total_rewards"]
    token_log_probs = result["token_log_probs"]
    token_ids = result["token_ids"]
    original_fsfs = result.get("original_fsfs", [0.0] * batch_size)
    rewrite_fsfs = result.get("rewrite_fsfs", [0.0] * batch_size)
    # Extract DOT outputs if available (when save_fsf_dot_outputs=True)
    # Note: original_dot_outputs is not extracted (original cot does not call FSF judge)
    rewrite_dot_outputs = result["extra_info"].get("rewrite_dot_outputs", [None] * batch_size)
    # FSF parse success status: True if FSF value is not None, False otherwise
    # Note: original_fsf_parse_success is not needed (original cot FSF is pre-computed in dataset)
    rewrite_fsf_parse_success = [1 if f is not None else 0 for f in rewrite_fsfs]
    too_short_penalties = result.get("too_short_penalties", [0.0] * batch_size)
    too_short_penalty_coeffs = result["extra_info"].get("too_short_penalty_coeffs", [0.0] * batch_size)
    too_long_penalties = result["extra_info"].get("too_long_penalties", [0.0] * batch_size)
    too_long_penalty_coeffs = result["extra_info"].get("too_long_penalty_coeffs", [0.0] * batch_size)
    length_ratios = result["extra_info"].get("length_ratios", [0.0] * batch_size)
    base_rewards = result["extra_info"].get("base_rewards", [0.0] * batch_size)
    penalty_scale = result["extra_info"].get("penalty_scale", 1.0)

    # Apply penalties based on penalty_mode
    # 1. Truncation penalty: if rollout result was truncated (finish_reason == "length")
    # 2. Format penalty: if response does not contain valid <rewrite_reasoning> tags
    finish_reasons = data.non_tensor_batch.get("finish_reasons", [])
    penalty_mode_from_result = result["extra_info"].get("penalty_mode", "relative")

    penalties = [0.0] * batch_size
    truncation_penalties = [0.0] * batch_size  # Track truncation penalties separately for logging
    format_penalties = [0.0] * batch_size  # Track format penalties separately for logging
    truncated_flags = [0] * batch_size  # Track truncated flags for DAPO filter_groups

    for i in range(batch_size):
        # Truncation penalty: penalty = truncation_penalty_coeff * penalty_scale (if relative) or truncation_penalty_coeff (if absolute)
        if i < len(finish_reasons) and finish_reasons[i] == "length":
            if penalty_mode_from_result == "relative":
                trunc_penalty = -truncation_penalty * penalty_scale
                penalty_log_msg = f"penalty={truncation_penalty} * {penalty_scale:.4f} = {trunc_penalty:.4f} (relative)"
            else:  # absolute
                trunc_penalty = -truncation_penalty
                penalty_log_msg = f"penalty={truncation_penalty:.4f} (absolute)"
            penalties[i] += trunc_penalty
            truncation_penalties[i] = trunc_penalty  # Log actual penalty value
            truncated_flags[i] = 1  # Mark as truncated for DAPO filter_groups
            logger.info(
                f"[TRUNCATION_PENALTY] Sample {i}: Truncated (finish_reason=length), "
                f"{penalty_log_msg}"
            )

        # Format penalty: penalty = format_penalty_coeff * penalty_scale (if relative) or format_penalty_coeff (if absolute)
        # Note: format_valid_flags is already computed above
        if format_valid_flags[i] == 0:
            if penalty_mode_from_result == "relative":
                fmt_penalty = -format_penalty * penalty_scale
                penalty_log_msg = f"penalty={format_penalty} * {penalty_scale:.4f} = {fmt_penalty:.4f} (relative)"
            else:  # absolute
                fmt_penalty = -format_penalty
                penalty_log_msg = f"penalty={format_penalty:.4f} (absolute)"
            penalties[i] += fmt_penalty
            format_penalties[i] = fmt_penalty  # Log actual penalty value
            logger.info(
                f"[FORMAT_PENALTY] Sample {i}: Format violation (no valid <rewrite_reasoning> tags found), "
                f"response_preview='{rewrite_cots[i][:100]}...', {penalty_log_msg}"
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
            "useful_rewards": useful_rewards,  # R_useful: CoT有用程度
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "no_cot_mean_log_probs": no_cot_mean_log_probs,  # Baseline (no CoT)
            "logic_rewards": logic_rewards,
            "total_rewards": total_rewards,
            "original_fsfs": original_fsfs,  # Pre-computed from dataset (observation only)
            "rewrite_fsfs": rewrite_fsfs,    # Computed by FSF judge (used for R_logic)
            # FSF parse success status: 1 if FSF value is not None, 0 otherwise
            # Note: original_fsf_parse_success is omitted (original cot FSF is pre-computed in dataset)
            "rewrite_fsf_parse_success": rewrite_fsf_parse_success,  # FSF 解析是否成功 (rewrite)
            # FSF DOT outputs (if save_fsf_dot_outputs=True)
            # Note: original_dot_outputs is omitted (original cot does not call FSF judge)
            "rewrite_dot_outputs": rewrite_dot_outputs,  # Raw DOT outputs for rewrite CoTs only
            # Penalties for SwanLab logging
            "truncation_penalties": truncation_penalties,
            "format_penalties": format_penalties,
            "too_short_penalties": too_short_penalties,
            "too_long_penalties": too_long_penalties,  # Penalty coefficients (0-0.5)
            "too_short_penalties_actual": result.get("extra_info", {}).get("too_short_penalties_actual", [0.0] * batch_size),  # Actual penalties applied
            "too_long_penalties_actual": result.get("extra_info", {}).get("too_long_penalties_actual", [0.0] * batch_size),  # Actual penalties applied
            "repeat_penalties": result.get("repeat_penalties", [0.0] * batch_size),  # Repeat penalty coefficients
            "repeat_penalties_actual": result.get("extra_info", {}).get("repeat_penalties_actual", [0.0] * batch_size),  # Actual repeat penalties applied
            "similarities": result.get("extra_info", {}).get("similarities", [0.0] * batch_size),  # Similarity values between original and rewrite
            "penalty_mode": penalty_mode_from_result,  # "relative" or "absolute"
            "saved_prompts": result.get("extra_info", {}).get("saved_prompts", None),  # Request prompts for debugging (if save_prompts=True)
            "fsf_reward_mode": fsf_reward_mode,  # FSF reward mode: 'direct' or 'threshold'
            # DAPO filter_groups support: "truncated" flag (1 if truncated, 0 otherwise)
            "truncated": truncated_flags,
            # DAPO filter_groups support: "format_valid" flag (1 if valid, 0 if invalid)
            # Used to filter out groups with all invalid format samples
            "format_valid": format_valid_flags,
            # DAPO filter_groups support: "length_ratio" metric for length-based filtering
            "length_ratio": length_ratios,
            # NEW: reward_deltas for logging rewrite-original improvement
            "reward_deltas": result.get("reward_deltas", [0.0] * batch_size),
            # PPL values (ppl mode only) - for debugging/observation
            "rewrite_ppls": result.get("rewrite_ppls", [float('nan')] * batch_size),
            "original_ppls": result.get("original_ppls", [float('nan')] * batch_size),
            # Relative improvement rates (formula 1 observation value)
            "relative_improvement_rates": result.get("relative_improvement_rates", [float('nan')] * batch_size),
            # FSF: Parse failure count and group skip flag
            "fsf_parse_failed_count": np.array(result.get("fsf_parse_failed_count", 0), dtype=object),  # Number of samples with parse failures
            "fsf_group_skipped": np.array(result.get("fsf_group_skipped", False), dtype=object),  # Whether FSF reward was skipped for entire group
            # Correct Judge: Results and penalties
            "is_correct_results": result.get("is_correct_results", [True] * batch_size),  # Correct Judge results
            "correct_judge_parse_failed_count": np.array(result.get("correct_judge_parse_failed_count", 0), dtype=object),  # Number of parse failures
            "r_cog_before_correct_judge": result.get("r_cog_before_correct_judge", cognitive_rewards),  # R_cog before penalty
            "correct_judge_penalties": result.get("correct_judge_penalties", [0.0] * batch_size),  # Penalties applied
            "train_mask": result.get("train_mask", [True] * batch_size),  # Training mask for GRPO (True=participate, False=filtered)
        },
    }

    # Cleanup Student vLLM to release GPU memory for Actor vLLM reload
    if cleanup_after_compute:
        cleanup_reward_computer()

    return reward_result


def compute_length_ratio_only(
    data: DataProto,
    tokenizer: Any,
) -> Dict[str, Any]:
    """
    Lightweight function that only computes length_ratio for filtering.

    This function does NOT load the Student Model, making it fast for pre-filtering.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for computing token lengths

    Returns:
        Dictionary with:
            - length_ratio: List of length ratios (rewrite_len / original_len)
            - format_valid: List of format validity flags (1 if valid, 0 if invalid)
            - reward_extra_info: Additional metrics for logging
    """
    # Fixed field names (standardized in rl_dataset_jsonl.py)
    reward_model_key = "reward_model"
    original_cot_key = "original_cot"

    # Check if using flat structure (no reward_model wrapper)
    use_flat_structure = False

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

    # Extract <rewrite_reasoning> content and check format validity
    extracted_rewrite_cots = []
    format_valid_flags = []
    for rewrite_cot in rewrite_cots:
        extracted_cot, has_valid_format, _ = extract_rewrite_reasoning(rewrite_cot)
        extracted_rewrite_cots.append(extracted_cot)  # Always use extracted content (strict or fallback)
        format_valid_flags.append(1 if has_valid_format else 0)

    # Compute length_ratios without loading Student Model
    length_ratios = [0.0] * batch_size
    for i in range(batch_size):
        original_text = original_cots[i] if original_cots[i] else ""
        # Use extracted content (strict or fallback) for length calculation
        rewrite_text = extracted_rewrite_cots[i] if extracted_rewrite_cots[i] else ""

        if original_text:
            original_token_len = len(tokenizer.encode(original_text, add_special_tokens=False))
            rewrite_token_len = len(tokenizer.encode(rewrite_text, add_special_tokens=False)) if rewrite_text else 0
            if original_token_len > 0:
                length_ratios[i] = rewrite_token_len / original_token_len

    # Log statistics
    valid_ratios = [r for r in length_ratios if r > 0]
    valid_format_count = sum(format_valid_flags)
    if valid_ratios:
        logger.info(
            f"[LENGTH_RATIO_ONLY] batch_size={batch_size}, "
            f"format_valid={valid_format_count}/{batch_size}, "
            f"mean={sum(valid_ratios)/len(valid_ratios):.3f}, "
            f"min={min(valid_ratios):.3f}, max={max(valid_ratios):.3f}"
        )

    return {
        "length_ratio": length_ratios,
        "format_valid": format_valid_flags,
    }
