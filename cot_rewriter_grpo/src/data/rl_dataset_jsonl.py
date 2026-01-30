"""
Custom RL Dataset with JSONL Support for CoT Rewriter GRPO

This module extends VERL's RLHFDataset to support JSONL file format.
It also handles plain text prompts by converting them to chat message format.
"""

import logging
from typing import Optional

import torch
import datasets
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

# Import verl's RLHFDataset as base class
try:
    from verl.utils.dataset.rl_dataset import RLHFDataset
except ImportError:
    # Fallback import path
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../verl"))
    from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class RLHFDatasetJSONL(RLHFDataset):
    """
    Extended RLHFDataset that supports JSONL file format.
    
    This class inherits from VERL's RLHFDataset and overrides the
    _read_files_and_tokenize method to add support for .jsonl files.
    
    Supported formats:
        - .parquet: Parquet files (original)
        - .json: JSON files (original)
        - .jsonl: JSON Lines files (NEW)
    
    Args:
        data_files (str or list): Path(s) to data file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Maximum number of samples to load. Default is -1 (all).
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Call parent init but skip the _read_files_and_tokenize call
        # We'll handle it ourselves
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def _read_files_and_tokenize(self):
        """
        Read data files and tokenize.
        
        Overrides parent method to add support for .jsonl files.
        """
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".jsonl"):
                # NEW: Support for JSONL files
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                import numpy as np
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _build_messages(self, example: dict):
        """
        Build chat messages from example.

        Overrides parent method to handle plain text prompts by converting them
        to chat message format: [{"role": "user", "content": "..."}]

        VERL's RLHFDataset expects prompt_key to point to a list of messages,
        but our data has plain text in the 'problem' field.

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages in chat format.
        """
        prompt_content = example[self.prompt_key]

        # If prompt is already a list (chat format), use parent's method
        if isinstance(prompt_content, list):
            return super()._build_messages(example)

        # If prompt is a plain string, convert to chat format
        if isinstance(prompt_content, str):
            messages = [{"role": "user", "content": prompt_content}]
            return messages

        raise ValueError(f"Unsupported prompt format: {type(prompt_content)}")

    def __getitem__(self, item):
        """
        Get a single item from the dataset.

        Overrides parent method to ensure proper handling of our data format
        and pass through all necessary fields for reward computation.
        """
        row_dict: dict = dict(self.dataframe[item])  # Make a copy

        # Build messages (handles plain text -> chat format conversion)
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # Dummy tensor required by VERL's DataProto
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # Ensure extra_info exists for metadata
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        # Set index if not present
        if "index" not in row_dict["extra_info"]:
            row_dict["extra_info"]["index"] = item
        row_dict["index"] = row_dict["extra_info"].get("index", item)

        # Build reward_model field from our data fields for reward function
        # This maps our field names to VERL's expected structure
        if "reward_model" not in row_dict or row_dict["reward_model"] is None:
            row_dict["reward_model"] = {}

        # Pass through important fields for reward computation
        # These will be accessible in the reward function via data.non_tensor_batch
        reward_model = row_dict["reward_model"]

        # Map ground truth (solution field)
        if "solution" in row_dict:
            reward_model["ground_truth"] = row_dict["solution"]

        # Map original CoT (thought field)
        if "thought" in row_dict:
            reward_model["original_cot"] = row_dict["thought"]

        # Map question
        if "question" in row_dict:
            reward_model["question"] = row_dict["question"]

        # Map FSF score if present
        if "fsf" in row_dict:
            reward_model["fsf"] = row_dict["fsf"]

        # Set data_source if not present (required by VERL)
        if "data_source" not in row_dict:
            row_dict["data_source"] = "cot_rewrite"

        return row_dict
