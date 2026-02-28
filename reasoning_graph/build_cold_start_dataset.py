#!/usr/bin/env python3
"""
Build cold start dataset from UltraChat 200k or MetaMathQA-40K for SFT.

This script:
1. Reads parquet files from HuggingFaceH4/ultrachat_200k or JSON files from MetaMathQA-40K
2. Converts multi-turn messages to single-turn conversations
3. Samples 10k examples
4. Saves to jsonl format
5. Registers the dataset to LlamaFactory

Input format (from parquet - UltraChat):
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }

Input format (from JSON - MetaMathQA-40K):
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Output format (sharegpt):
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Usage:
    # Default usage (UltraChat)
    python build_cold_start_dataset.py

    # MetaMathQA-40K usage
    python build_cold_start_dataset.py \
        --input-pattern "/fs/fast/u20240075/hf_home/datasets/meta-math/MetaMathQA-40K/MetaMathQA-40K.json" \
        --output-dir "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/cold_start/metamath_10k" \
        --num-samples 10000 \
        --dataset-info "LlamaFactory/data/dataset_info.json"

    # Custom parameters
    python build_cold_start_dataset.py \
        --input-pattern "/path/to/ultrachat/*.parquet" \
        --output-dir "/path/to/output" \
        --num-samples 10000 \
        --dataset-info "LlamaFactory/data/dataset_info.json"
"""

import json
import argparse
import os
import random
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build cold start dataset from UltraChat 200k or MetaMathQA-40K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (UltraChat)
  python build_cold_start_dataset.py

  # MetaMathQA-40K usage
  python build_cold_start_dataset.py \
      --input-pattern "/fs/fast/u20240075/hf_home/datasets/meta-math/MetaMathQA-40K/MetaMathQA-40K.json" \
      --output-dir "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/cold_start/metamath_10k" \
      --num-samples 10000 \
      --dataset-info "LlamaFactory/data/dataset_info.json"

  # Custom parameters (UltraChat)
  python build_cold_start_dataset.py \
      --input-pattern "/fs/fast/u20240075/hf_home/datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-*.parquet" \
      --output-dir "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/cold_start/ultrachat_10k" \
      --num-samples 10000 \
      --dataset-info "LlamaFactory/data/dataset_info.json"
        """,
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="/fs/fast/u20240075/hf_home/datasets/HuggingFaceH4/ultrachat_200k/data/train_sft-0000*-of-00003-a3ecf92756993583.parquet",
        help="Glob pattern for input files (supports .parquet and .json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/cold_start/ultrachat_10k",
        help="Output directory path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to select (default: 10000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="Training set ratio (default: 0.98)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dataset-info",
        type=str,
        default="LlamaFactory/data/dataset_info.json",
        help="Path to dataset_info.json (default: LlamaFactory/data/dataset_info.json)",
    )

    return parser.parse_args()


def load_parquet_data(file_pattern: str) -> List[Dict[str, Any]]:
    """
    Load data from parquet files matching the given pattern.

    Args:
        file_pattern: Glob pattern for parquet files

    Returns:
        List of data items
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to read parquet files. Install with: pip install pandas")

    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    files.sort()
    logger.info(f"Found {len(files)} parquet file(s):")
    for f in files:
        logger.info(f"  - {f}")

    all_data = []
    for file_path in files:
        logger.info(f"Loading {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            # Convert to list of dicts
            data = df.to_dict('records')
            all_data.extend(data)
            logger.info(f"  Loaded {len(data)} records from {Path(file_path).name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

    logger.info(f"Total records loaded: {len(all_data)}")
    return all_data


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of data items
    """
    logger.info(f"Loading JSON file: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list format and dict format
        if isinstance(data, list):
            all_data = data
        elif isinstance(data, dict):
            # Try common keys that might contain the data list
            for key in ['data', 'items', 'conversations', 'examples']:
                if key in data and isinstance(data[key], list):
                    all_data = data[key]
                    break
            else:
                # If it's a single item, wrap in list
                all_data = [data]
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")
        
        logger.info(f"  Loaded {len(all_data)} records from {Path(file_path).name}")
        return all_data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def load_data(file_pattern: str) -> List[Dict[str, Any]]:
    """
    Load data from files matching the given pattern.
    Automatically detects file type (.parquet or .json) and uses appropriate loader.

    Args:
        file_pattern: Glob pattern for input files

    Returns:
        List of data items
    """
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    # Check if it's a JSON file pattern
    if len(files) == 1 and files[0].endswith('.json'):
        return load_json_data(files[0])
    
    # Default to parquet loader
    return load_parquet_data(file_pattern)


def convert_to_single_turn(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert multi-turn messages to single-turn conversation.
    Takes the first user message and the first assistant response.

    Supports two input formats:
    1. UltraChat format: item contains 'messages' field
    2. MetaMathQA-40K format: item contains 'conversations' field with query/response

    Args:
        item: Data item containing 'messages' or 'conversations' field

    Returns:
        Sharegpt format item with single-turn conversation, or None if invalid
    """
    # First, try to extract from 'messages' field (UltraChat format)
    messages = item.get("messages", [])
    # Convert numpy array to list if needed
    if hasattr(messages, 'tolist'):
        messages = messages.tolist()
    
    if messages and len(messages) >= 2:
        # Find first user message and first assistant response
        user_message = None
        assistant_message = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user" and user_message is None:
                user_message = content
            elif role == "assistant" and assistant_message is None:
                assistant_message = content

            # Break once we have both
            if user_message is not None and assistant_message is not None:
                break

        if user_message is not None and assistant_message is not None:
            # Build sharegpt format
            return {
                "conversations": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            }
    
    # Second, try to extract from 'conversations' field (MetaMathQA-40K format)
    conversations = item.get("conversations", [])
    # Convert numpy array to list if needed
    if hasattr(conversations, 'tolist'):
        conversations = conversations.tolist()
    
    if conversations and len(conversations) >= 2:
        # Get first user and first assistant from conversations
        user_message = None
        assistant_message = None
        
        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user" and user_message is None:
                user_message = content
            elif role == "assistant" and assistant_message is None:
                assistant_message = content
            
            # Break once we have both
            if user_message is not None and assistant_message is not None:
                break
        
        if user_message is not None and assistant_message is not None:
            # Already in sharegpt format, return as-is
            return {
                "conversations": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ]
            }
    
    # Third, try 'query' and 'response' fields (some datasets use this format)
    query = item.get("query") or item.get("question") or item.get("input")
    response = item.get("response") or item.get("answer") or item.get("output")
    
    if query is not None and response is not None:
        return {
            "conversations": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
        }
    
    return None


def process_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all data items and convert to sharegpt format.

    Args:
        data: List of raw data items

    Returns:
        List of processed sharegpt format items
    """
    processed = []
    skipped = 0

    for item in data:
        converted = convert_to_single_turn(item)
        if converted is not None:
            processed.append(converted)
        else:
            skipped += 1

    logger.info(f"Processed {len(processed)} valid conversations, skipped {skipped} invalid items")
    return processed


def sample_data(data: List[Dict[str, Any]], num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Randomly sample specified number of items from data.

    Args:
        data: List of data items
        num_samples: Number of samples to select
        seed: Random seed

    Returns:
        Sampled data items
    """
    random.seed(seed)

    if len(data) <= num_samples:
        logger.info(f"Data size ({len(data)}) <= requested samples ({num_samples}), using all data")
        return data

    sampled = random.sample(data, num_samples)
    logger.info(f"Sampled {len(sampled)} items from {len(data)} total")
    return sampled


def split_train_val(
    data: List[Dict[str, Any]], train_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training and validation sets.

    Args:
        data: All data items
        train_ratio: Training set ratio
        seed: Random seed

    Returns:
        (train_data, val_data) tuple
    """
    random.seed(seed)

    # Shuffle data
    shuffled = data.copy()
    random.shuffle(shuffled)

    # Split
    train_size = int(len(shuffled) * train_ratio)
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:]

    return train_data, val_data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """
    Save data to JSONL file.

    Args:
        data: List of data items
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} items to {output_path}")


def register_to_dataset_info(
    dataset_info_path: str, dataset_name: str, file_path: str
) -> bool:
    """
    Register dataset to dataset_info.json for LlamaFactory.

    Args:
        dataset_info_path: Path to dataset_info.json
        dataset_name: Dataset name
        file_path: Path to dataset file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read dataset_info.json
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        # Calculate relative path
        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            # Use absolute path if not in LlamaFactory/data
            relative_path = str(Path(file_path).resolve())
            logger.warning(f"File not in LlamaFactory/data directory, using absolute path: {relative_path}")

        # Add to dataset_info
        dataset_info[dataset_name] = {
            "file_name": relative_path,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }

        # Write back
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Dataset '{dataset_name}' registered to {dataset_info_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to register dataset: {e}")
        return False


def print_statistics(data: List[Dict[str, Any]], name: str = "Dataset"):
    """Print statistics about the dataset."""
    total = len(data)
    if total == 0:
        logger.info(f"{name}: No entries")
        return

    # Calculate average content lengths
    user_lengths = []
    assistant_lengths = []

    for item in data:
        conversations = item.get("conversations", [])
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", "")
            if role == "user":
                user_lengths.append(len(content))
            elif role == "assistant":
                assistant_lengths.append(len(content))

    avg_user_len = sum(user_lengths) / len(user_lengths) if user_lengths else 0
    avg_assistant_len = sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0

    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Total entries: {total}")
    logger.info(f"  Avg user content length: {avg_user_len:.0f} chars")
    logger.info(f"  Avg assistant content length: {avg_assistant_len:.0f} chars")


def main():
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("  Cold Start Dataset Builder")
    logger.info("=" * 70)
    logger.info(f"📁 Input pattern: {args.input_pattern}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📊 Number of samples: {args.num_samples}")
    logger.info(f"📊 Train ratio: {args.train_ratio}")
    logger.info(f"🎲 Random seed: {args.seed}")
    logger.info(f"📝 Dataset info: {args.dataset_info}")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n" + "-" * 60)
    logger.info("Step 1: Loading data...")
    logger.info("-" * 60)

    raw_data = load_data(args.input_pattern)

    if not raw_data:
        logger.error("No data loaded!")
        return 1

    # Step 2: Convert to single-turn sharegpt format
    logger.info("\n" + "-" * 60)
    logger.info("Step 2: Converting to single-turn sharegpt format...")
    logger.info("-" * 60)

    processed_data = process_data(raw_data)

    if not processed_data:
        logger.error("No valid data after processing!")
        return 1

    # Step 3: Sample data
    logger.info("\n" + "-" * 60)
    logger.info("Step 3: Sampling data...")
    logger.info("-" * 60)

    sampled_data = sample_data(processed_data, args.num_samples, args.seed)

    # Step 4: Split train/val
    logger.info("\n" + "-" * 60)
    logger.info("Step 4: Splitting train/val...")
    logger.info("-" * 60)

    train_data, val_data = split_train_val(sampled_data, args.train_ratio, args.seed)

    print_statistics(train_data, "Train Set")
    print_statistics(val_data, "Validation Set")

    # Step 5: Save data
    logger.info("\n" + "-" * 60)
    logger.info("Step 5: Saving data...")
    logger.info("-" * 60)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))

    # Step 6: Register to dataset_info.json
    logger.info("\n" + "-" * 60)
    logger.info("Step 6: Registering to dataset_info.json...")
    logger.info("-" * 60)

    dataset_base_name = output_path.name
    train_dataset_name = f"{dataset_base_name}_train"
    val_dataset_name = f"{dataset_base_name}_val"

    train_registered = register_to_dataset_info(
        args.dataset_info, train_dataset_name, str(train_path)
    )
    val_registered = register_to_dataset_info(
        args.dataset_info, val_dataset_name, str(val_path)
    )

    # Step 7: Save config and log
    logger.info("\n" + "-" * 60)
    logger.info("Step 7: Saving config and logs...")
    logger.info("-" * 60)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "input_pattern": args.input_pattern,
            "output_dir": str(output_path.absolute()),
            "num_samples": args.num_samples,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "dataset_info": args.dataset_info,
        },
        "counts": {
            "raw_total": len(raw_data),
            "processed": len(processed_data),
            "sampled": len(sampled_data),
            "train_count": len(train_data),
            "val_count": len(val_data),
        },
    }

    log_path = output_path / "build_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Log saved to: {log_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  Summary")
    logger.info("=" * 70)
    logger.info(f"✅ Output saved to: {output_path.absolute()}")
    logger.info(f"   Train: {train_path.name} ({len(train_data)} entries)")
    logger.info(f"   Val:   {val_path.name} ({len(val_data)} entries)")

    if train_registered and val_registered:
        logger.info(f"✅ Datasets registered successfully!")
        logger.info(f"   Train dataset name: {train_dataset_name}")
        logger.info(f"   Val dataset name:   {val_dataset_name}")
    else:
        logger.warning("⚠️  Some datasets failed to register")

    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())


"""
# Example usage:

# Default usage (UltraChat)
python build_cold_start_dataset.py

# MetaMathQA-40K usage
python build_cold_start_dataset.py \
    --input-pattern "/fs/fast/u20240075/hf_home/datasets/meta-math/MetaMathQA-40K/MetaMathQA-40K.json" \
    --output-dir "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/cold_start/metamath_40k" \
    --num-samples 40000 \
    --dataset-info "../LlamaFactory/data/dataset_info.json"

"""
