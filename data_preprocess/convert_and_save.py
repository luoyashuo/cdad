#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_and_save.py
Convert the filtered MATH data to HuggingFace dataset format.
"""

import os
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict


def load_jsonl(jsonl_path):
    """Load JSONL file"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
    return data


def convert_to_hf_format(jsonl_path, output_dir):
    """Convert JSONL to HuggingFace dataset format"""
    # Load the JSONL data
    print("Loading JSONL data...")
    data = []

    # Since the file might be corrupted, create a simple test version
    if os.path.exists(jsonl_path):
        data = load_jsonl(jsonl_path)

    if not data:
        print("No valid data found. Creating test data...")
        # Manually create 30k MATH samples from the parquet files
        create_math_dataset(jsonl_path)
        data = load_jsonl(jsonl_path)

    print(f"Loaded {len(data)} samples")

    # For now, just create a simple random split
    random.seed(42)
    random.shuffle(data)

    train_count = int(len(data) * 0.9)
    train_data = data[:train_count]
    test_data = data[train_count:]

    # Create datasets
    print("Creating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Save to disk
    print(f"Saving HuggingFace dataset to {output_dir}")
    dataset_dict.save_to_disk(output_dir)

    # Also save as separate JSONL files for easy access
    os.makedirs(output_dir, exist_ok=True)

    train_jsonl = os.path.join(output_dir, "train.jsonl")
    test_jsonl = os.path.join(output_dir, "test.jsonl")

    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(test_jsonl, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(train_data)} train samples to {train_jsonl}")
    print(f"Saved {len(test_data)} test samples to {test_jsonl}")

    # Save dataset info
    info = {
        "dataset": "OmniThought MATH Filtered",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "total_samples": len(data),
        "filter_criteria": "Contains '\\boxed' in reasoning solution",
        "format": "Original OmniThought format",
        "description": "MATH problems with \boxed answers from OmniThought dataset"
    }

    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def create_math_dataset(output_jsonl):
    """Create a dataset from OmniThought parquets with MATH problems"""
    import pyarrow.parquet as pq
    import numpy as np

    data_path = "/mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data"
    parquet_files = list(Path(data_path).glob("*.parquet"))

    print(f"Processing {len(parquet_files)} parquet files...")

    all_data = []
    count = 0

    # Randomly select subset to create 30000 samples
    random.seed(42)
    random.shuffle(parquet_files)

    for parquet_file in parquet_files:
        if count >= 30000:
            break

        try:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Process each row
            for idx, row in df.iterrows():
                if count >= 30000:
                    break

                # Check if MATH problem
                if 'reasoning' in row and row['reasoning'] is not None:
                    reasoning = row['reasoning']

                    # Get first reasoning item
                    if isinstance(reasoning, np.ndarray) and len(reasoning) > 0:
                        reasoning_item = reasoning[0]
                        if isinstance(reasoning_item, dict) and 'solution' in reasoning_item:
                            solution = reasoning_item['solution']
                            if solution and r'\boxed' in solution:
                                # This is a MATH problem
                                data_item = {
                                    'question': row.get('question', ''),
                                    'reasoning': [{
                                        'Cognitive_Difficulty': reasoning_item.get('Cognitive_Difficulty', {}),
                                        'Reasoning_Verbosity': reasoning_item.get('Reasoning_Verbosity', {}),
                                        'full_response': reasoning_item.get('full_response', ''),
                                        'solution': solution,
                                        'teacher': reasoning_item.get('teacher', ''),
                                        'thought': reasoning_item.get('thought', ''),
                                        'thought_correctness_verify': reasoning_item.get('thought_correctness_verify', '')
                                    }],
                                    'answer': row.get('answer', ''),
                                    'metadata': row.get('metadata', {})
                                }
                                all_data.append(data_item)
                                count += 1

        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue

        if count % 1000 == 0:
            print(f"Processed {count} MATH problems...")

    print(f"Total found: {len(all_data)} MATH problems")

    # Save to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(all_data)} samples to {output_jsonl}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_30k_v2/train.jsonl")
    parser.add_argument("--output_dir", default="/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_30k_final")
    parser.add_argument("--create_from_parquet", action="store_true", help="Create from parquet files directly")

    args = parser.parse_args()

    # Create the HuggingFace dataset from the JSONL
    convert_to_hf_format(args.input_jsonl, args.output_dir)


if __name__ == "__main__":
    main()