"""
CoT Rewriter Reward Manager for VERL

This reward manager properly interfaces with compute_cot_rewriter_reward function
which expects (data, tokenizer, ...) arguments instead of the standard
(data_source, solution_str, ground_truth, extra_info) interface.
"""

import asyncio
import inspect
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from verl.experimental.reward_loop.reward_manager import register
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


# 修改注册装饰器，允许重复注册相同的类
def safe_register(name):
    def decorator(cls):
        try:
            from verl.experimental.reward_loop.reward_manager.registry import (
                REWARD_LOOP_MANAGER_REGISTRY,
            )

            # 允许重复注册相同的类
            REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        except Exception:
            pass
        return cls

    return decorator


@safe_register("cot_rewriter_reward_manager")
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

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Make the manager callable like traditional reward managers.

        This method provides compatibility with the existing reward manager interface
        by wrapping the async run_single method in a synchronous call.

        Args:
            data: DataProto object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False: torch.Tensor of rewards
        """
        batch_size = len(data)

        # Process each item in the batch
        all_rewards = []
        all_extra_infos = {}

        for i in range(batch_size):
            data_item = data[
                i : i + 1
            ]  # Get single item as DataProto with batch_size=1

            # Run the async run_single method
            result = self.loop.run_until_complete(self.run_single(data_item))

            all_rewards.append(result["reward_score"])

            # Collect extra info
            for key, value in result.get("reward_extra_info", {}).items():
                if key not in all_extra_infos:
                    all_extra_infos[key] = []
                all_extra_infos[key].append(value)

        # Build reward_tensor
        response_length = data.batch["responses"].shape[1]
        device = data.batch["responses"].device
        reward_tensor = torch.zeros(batch_size, response_length, device=device)

        # Get actual sequence lengths
        response_mask = data.batch.get("response_mask", None)
        if response_mask is not None:
            seq_lengths = response_mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
        else:
            seq_lengths = torch.full(
                (batch_size,), response_length - 1, dtype=torch.long, device=device
            )

        # Assign reward to last valid token of each sequence
        for i in range(batch_size):
            reward_tensor[i, seq_lengths[i]] = all_rewards[i]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": all_extra_infos,
            }
        return reward_tensor

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

            # Copy extra info, but unwrap single-element lists to scalars
            # because compute_cot_rewriter_reward returns batch-level lists (batch_size=1)
            # and VERL's agent_loop.py:778 expects scalar values per sample
            for key, value in result.get("reward_extra_info", {}).items():
                # Unwrap single-element lists to scalars
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    reward_extra_info[key] = value[0]
                elif isinstance(value, (list, tuple)) and len(value) > 1:
                    # Skip variable-length lists (like token_log_probs)
                    # as they break numpy array conversion
                    continue
                else:
                    reward_extra_info[key] = value
        else:
            # Scalar result
            score = float(result)
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}
