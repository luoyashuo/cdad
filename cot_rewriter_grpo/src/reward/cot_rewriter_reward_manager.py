"""
CoT Rewriter Reward Manager for VERL

This reward manager properly interfaces with compute_cot_rewriter_reward function
which expects (data, tokenizer, ...) arguments instead of the standard
(data_source, solution_str, ground_truth, extra_info) interface.
"""

import inspect
from typing import Any, Optional

from omegaconf import DictConfig
from verl.experimental.reward_loop.reward_manager import register

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@register("cot_rewriter_reward_manager")  # 注册名称，用于配置文件引用
class CoTRewriterRewardManager(RewardManagerBase):
    """Reward manager specifically designed for CoT Rewriter GRPO training.

    This manager properly interfaces with compute_cot_rewriter_reward function
    which requires (data, tokenizer, ...) as its primary arguments.
    """

    def __init__(
        self,
        config=None,
        tokenizer=None,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        num_examine=0,
        reward_fn_key="reward_model",
        **kwargs,
    ):
        """Initialize CoT Rewriter Reward Manager.

        Args:
            config: Configuration object (optional, for backward compatibility)
            tokenizer: Tokenizer for decoding
            compute_score: The compute_cot_rewriter_reward function (wrapped with kwargs)
            reward_router_address: Optional reward router address
            reward_model_tokenizer: Optional reward model tokenizer
            num_examine: Number of samples to examine (unused, for compatibility)
            reward_fn_key: Key for reward function in data (unused, for compatibility)
            **kwargs: Additional keyword arguments (for compatibility)
        """
        # Handle both calling conventions:
        # 1. RewardManagerBase style: (config, tokenizer, ...)
        # 2. AbstractRewardManager style: (tokenizer, num_examine, ...)
        if config is None:
            # Called with AbstractRewardManager style, create a minimal config
            config = DictConfig({})

        if tokenizer is None:
            raise ValueError("CoTRewriterRewardManager requires 'tokenizer' parameter.")

        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.is_async_reward_score = (
            inspect.iscoroutinefunction(self.compute_score)
            if self.compute_score
            else False
        )
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        # Store num_examine for compatibility (not used)
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

    async def run_single(self, data: DataProto) -> dict:
        """Run reward computation for a single data item.

        Args:
            data: DataProto containing a single data item (batch_size=1)

        Returns:
            Dictionary with reward_score and reward_extra_info
        """
        assert len(data) == 1, "Only support single data item"

        if self.compute_score is None:
            raise ValueError(
                "compute_score (compute_cot_rewriter_reward) must be provided. "
                "Please configure custom_reward_function in your config."
            )

        # Call compute_cot_rewriter_reward with (data, tokenizer) interface
        # The function is already wrapped with reward_kwargs via _call_with_kwargs
        if self.is_async_reward_score:
            result = await self.compute_score(data=data, tokenizer=self.tokenizer)
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(data=data, tokenizer=self.tokenizer),
            )

        # Process the result
        reward_extra_info = {}

        if isinstance(result, dict):
            # Result from compute_cot_rewriter_reward contains:
            # - reward_tensor: [batch_size, seq_len] tensor
            # - reward_extra_info: dict with additional metrics

            # Extract the reward value (typically from the last token position)
            reward_tensor = result.get("reward_tensor")
            if reward_tensor is not None:
                # Get reward from the last non-zero position
                # reward_tensor shape: [batch_size, seq_len]
                batch_size = reward_tensor.shape[0]
                # Sum over sequence dimension to get total reward per sample
                total_rewards = reward_tensor.sum(dim=-1)  # [batch_size]
                score = total_rewards[0].item()  # First (and only) item
            else:
                # Fallback: try to get score from extra_info
                extra_info = result.get("reward_extra_info", {})
                score = extra_info.get("mean_total_reward", 0.0)

            # Copy all extra info
            for key, value in result.get("reward_extra_info", {}).items():
                reward_extra_info[key] = value
        else:
            # Scalar result
            score = float(result)
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
