"""
Dataset Processing Module for CoT Rewriter GRPO

Provides dataset classes and utilities for preparing CoT rewriting training data.
The data format is designed to work with VERL's data pipeline.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset


# Prompt template for CoT rewriting
REWRITE_PROMPT_TEMPLATE = """You are an expert at improving reasoning processes. Your task is to rewrite the given Chain-of-Thought (CoT) reasoning to make it clearer, more logical, and easier to follow.

Requirements:
1. Keep the core reasoning steps intact
2. Remove unnecessary backtracking or failed attempts
3. Make the logical flow more explicit
4. Ensure the reasoning leads to the correct answer

Question: {question}

Original Reasoning:
{original_cot}

Correct Answer: {answer}

Please rewrite the reasoning process to be cleaner and more direct. Only output the improved reasoning, no explanations:"""


class CoTRewriterDataset:
    """
    CoT Rewriter Dataset

    Handles loading, processing, and formatting data for CoT rewriting training.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any = None,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data file (JSON, JSONL, or Parquet)
            tokenizer: Tokenizer for encoding (optional)
            max_prompt_length: Maximum prompt length
            max_response_length: Maximum response length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        suffix = self.data_path.suffix.lower()

        if suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif suffix == ".jsonl":
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
            data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return data

    def format_for_verl(self) -> List[Dict[str, Any]]:
        """
        Format data for VERL training.

        Returns:
            List of formatted examples with:
                - prompt: The formatted rewrite prompt
                - reward_model: Dictionary with ground_truth, original_cot, question
        """
        formatted = []

        for item in self.data:
            # Support multiple field names for compatibility
            question = (
                item.get("question", "")
                or item.get("problem", "")
                or item.get("input", "")
                or item.get("query", "")
            )
            original_cot = (
                item.get("original_cot", "")
                or item.get("reasoning", "")
                or item.get("cot", "")
                or item.get("chain_of_thought", "")
                or item.get("solution", "")
            )
            answer = (
                item.get("answer", "")
                or item.get("ground_truth", "")
                or item.get("target", "")
                or item.get("output", "")
                or item.get("label", "")
            )

            # Build prompt
            prompt = REWRITE_PROMPT_TEMPLATE.format(
                question=question,
                original_cot=original_cot,
                answer=answer,
            )

            formatted.append({
                "prompt": prompt,
                "reward_model": {
                    "question": question,
                    "original_cot": original_cot,
                    "ground_truth": answer,
                },
            })

        return formatted

    def to_parquet(self, output_path: Union[str, Path]) -> None:
        """
        Save formatted data to Parquet file.

        Args:
            output_path: Output file path
        """
        formatted = self.format_for_verl()

        # Convert to DataFrame-friendly format
        records = []
        for item in formatted:
            record = {
                "prompt": item["prompt"],
                "question": item["reward_model"]["question"],
                "original_cot": item["reward_model"]["original_cot"],
                "ground_truth": item["reward_model"]["ground_truth"],
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} examples to {output_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def prepare_dataset(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Prepare train/val splits from raw data.

    Args:
        input_path: Path to input data file
        output_dir: Directory to save processed files
        train_ratio: Ratio for training split
        seed: Random seed for splitting

    Returns:
        Dictionary with paths to train and val parquet files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = CoTRewriterDataset(input_path)
    formatted = dataset.format_for_verl()

    # Split
    import random
    random.seed(seed)
    random.shuffle(formatted)

    n_train = int(len(formatted) * train_ratio)
    train_data = formatted[:n_train]
    val_data = formatted[n_train:]

    # Save
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    # Convert to DataFrames
    def to_df(data):
        records = []
        for item in data:
            records.append({
                "prompt": item["prompt"],
                "question": item["reward_model"]["question"],
                "original_cot": item["reward_model"]["original_cot"],
                "ground_truth": item["reward_model"]["ground_truth"],
            })
        return pd.DataFrame(records)

    to_df(train_data).to_parquet(train_path, index=False)
    to_df(val_data).to_parquet(val_path, index=False)

    print(f"Created train split: {len(train_data)} examples -> {train_path}")
    print(f"Created val split: {len(val_data)} examples -> {val_path}")

    return {
        "train": str(train_path),
        "val": str(val_path),
    }


def create_sample_data(output_path: Union[str, Path], n_samples: int = 10) -> None:
    """
    Create sample data for testing.

    Args:
        output_path: Path to save sample data
        n_samples: Number of samples to create
    """
    samples = []

    # Math reasoning examples
    math_examples = [
        {
            "question": "What is 15 + 27?",
            "original_cot": "Let me think... 15 plus 27. First, I'll try 15 + 20 = 35. Wait, that's not right because I need to add 27, not 20. Let me start over. 15 + 27: I can break it down as 15 + 25 + 2 = 40 + 2 = 42. Yes, that's correct!",
            "answer": "42",
        },
        {
            "question": "If a train travels 60 miles in 2 hours, what is its average speed?",
            "original_cot": "Speed = distance / time. Hmm, wait, is it time / distance? No, it's distance / time. So 60 miles divided by 2 hours. Let me calculate: 60 / 2 = 30. So the speed is 30 miles per hour.",
            "answer": "30 miles per hour",
        },
        {
            "question": "What is 8 * 7?",
            "original_cot": "8 times 7. I know 8 * 5 = 40. And 8 * 2 = 16. So 8 * 7 = 8 * 5 + 8 * 2 = 40 + 16 = 56.",
            "answer": "56",
        },
    ]

    # Logic reasoning examples
    logic_examples = [
        {
            "question": "All cats are animals. Fluffy is a cat. Is Fluffy an animal?",
            "original_cot": "Let's see... All cats are animals - that's our first premise. Wait, let me think if there could be exceptions. No, the statement says 'all cats'. Fluffy is a cat - that's our second premise. So by logical deduction, since all cats are animals and Fluffy is a cat, Fluffy must be an animal.",
            "answer": "Yes, Fluffy is an animal.",
        },
        {
            "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "original_cot": "This is tricky. If it rains -> ground is wet. Ground is wet. Can I conclude it rained? Wait, this might be affirming the consequent. The ground could be wet for other reasons (sprinkler, etc.). So I cannot definitively say it rained. This is a logical fallacy if I conclude it rained.",
            "answer": "We cannot conclude definitively that it rained (affirming the consequent fallacy).",
        },
    ]

    # Code reasoning examples
    code_examples = [
        {
            "question": "What does this code output? for i in range(3): print(i)",
            "original_cot": "Let me trace through. range(3) gives 0, 1, 2. Wait, does it include 3? No, range(n) goes from 0 to n-1. So: i=0, print 0; i=1, print 1; i=2, print 2. The output is 0, 1, 2 on separate lines.",
            "answer": "0\n1\n2",
        },
    ]

    all_examples = math_examples + logic_examples + code_examples

    # Repeat to get desired number of samples
    while len(samples) < n_samples:
        for ex in all_examples:
            if len(samples) >= n_samples:
                break
            samples.append(ex.copy())

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Created {len(samples)} sample examples -> {output_path}")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("data/examples/sample_data.json", n_samples=20)

    # Process sample data
    prepare_dataset(
        input_path="data/examples/sample_data.json",
        output_dir="data",
        train_ratio=0.8,
    )
