"""
CoT Rewriter Reward Manager for VERL

This reward manager properly interfaces with compute_cot_rewriter_reward function
which expects (data, tokenizer, ...) arguments instead of the standard
(data_source, solution_str, ground_truth, extra_info) interface.
"""

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from verl.experimental.reward.cot_rewriter.logging_config import logger


@register("cot_rewriter")
class CoTRewriterRewardManager:
    """Reward manager specifically designed for CoT Rewriter GRPO training.

    This manager directly calls compute_cot_rewriter_reward with the full DataProto
    and tokenizer, instead of the sample-by-sample interface used by NaiveRewardManager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "reward_model",
        **kwargs,
    ):
        """Initialize CoT Rewriter Reward Manager.

        Args:
            tokenizer: Tokenizer for decoding
            num_examine: Number of samples to examine for debugging
            compute_score: The compute_cot_rewriter_reward function (wrapped with kwargs)
            reward_fn_key: Key for reward function in data
            **kwargs: Additional keyword arguments
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for a batch of data.

        This method directly calls compute_cot_rewriter_reward with the full batch,
        which is more efficient for models that benefit from batch processing.

        Args:
            data: DataProto object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False: torch.Tensor of rewards
        """
        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return data.batch["rm_scores"]

        if self.compute_score is None:
            raise ValueError(
                "compute_score (compute_cot_rewriter_reward) must be provided. "
                "Please configure custom_reward_function in your config."
            )

        # Call compute_cot_rewriter_reward directly with the full batch
        # The function is wrapped by _call_with_kwargs which merges reward_kwargs
        # We pass data and tokenizer as keyword arguments
        import time
        batch_size = len(data)
        print(f"[REWARD_MANAGER] Starting compute_score for {batch_size} samples...")
        start_time = time.time()
        result = self.compute_score(data=data, tokenizer=self.tokenizer)
        elapsed = time.time() - start_time
        print(f"[REWARD_MANAGER] compute_score completed in {elapsed:.2f}s")

        # Process the result
        if isinstance(result, dict):
            # Result from compute_cot_rewriter_reward contains:
            # - reward_tensor: [batch_size, seq_len] tensor
            # - reward_extra_info: dict with additional metrics
            reward_tensor = result.get("reward_tensor")
            reward_extra_info = result.get("reward_extra_info", {})

            if reward_tensor is None:
                # Fallback: create zero tensor
                batch_size = len(data)
                response_length = data.batch["responses"].shape[1]
                device = data.batch["responses"].device
                reward_tensor = torch.zeros(
                    batch_size, response_length, device=device, dtype=torch.float32
                )

            # Print some samples for debugging
            if self.num_examine > 0:
                self._print_samples(data, result, self.num_examine)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
        else:
            # Scalar result - should not happen with compute_cot_rewriter_reward
            # but handle it for robustness
            batch_size = len(data)
            response_length = data.batch["responses"].shape[1]
            device = data.batch["responses"].device
            reward_tensor = torch.zeros(
                batch_size, response_length, device=device, dtype=torch.float32
            )

            # Assign the scalar reward to all samples at the last position
            reward_tensor[:, -1] = float(result)

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            else:
                return reward_tensor

    def _print_samples(self, data: DataProto, result: dict, num_examine: int):
        """Print sample outputs for debugging."""
        batch_size = len(data)
        num_to_print = min(num_examine, batch_size)

        extra_info = result.get("reward_extra_info", {})
        total_rewards = extra_info.get("total_rewards", [])
        cognitive_rewards = extra_info.get("cognitive_rewards", [])
        rewrite_mean_log_probs = extra_info.get("rewrite_mean_log_probs", [])
        original_mean_log_probs = extra_info.get("original_mean_log_probs", [])

        logger.info(f"\n{'='*80}")
        logger.info(f"[ROLLOUT SAMPLES] Showing {num_to_print} samples for review")
        logger.info(f"{'='*80}")

        for i in range(num_to_print):
            # Decode prompt and response
            prompt_ids = data.batch["prompts"][i]
            response_ids = data.batch["responses"][i]

            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Extract original_cot from non_tensor_batch
            original_cot = ""
            reward_model_data = data.non_tensor_batch.get("reward_model", None)
            if reward_model_data is not None:
                if hasattr(reward_model_data, '__len__') and i < len(reward_model_data):
                    rm_item = reward_model_data[i]
                    if isinstance(rm_item, dict):
                        original_cot = rm_item.get("original_cot", "")

            logger.info(f"\n[Sample {i}]")
            logger.info(f"[question] {prompt_str[:500]}..." if len(prompt_str) > 500 else f"[question] {prompt_str}")
            logger.info(f"[rollout_output] {response_str[:800]}..." if len(response_str) > 800 else f"[rollout_output] {response_str}")
            if original_cot:
                logger.info(f"[old_cot] {str(original_cot)[:500]}..." if len(str(original_cot)) > 500 else f"[old_cot] {original_cot}")

            # Print reward info
            if i < len(total_rewards):
                logger.info(f"[total_reward] {total_rewards[i]}")
            if i < len(cognitive_rewards):
                logger.info(f"[cognitive_reward] {cognitive_rewards[i]}")
            if i < len(rewrite_mean_log_probs):
                logger.info(f"[rewrite_mean_log_prob] {rewrite_mean_log_probs[i]:.4f}")
            if i < len(original_mean_log_probs):
                logger.info(f"[original_mean_log_prob] {original_mean_log_probs[i]:.4f}")
            logger.info("-" * 80)

        logger.info(f"{'='*80}\n")
