#!/usr/bin/env python3
"""
Add data_source field to raw JSONL data for VERL compatibility.

Usage:
    python scripts/preprocess/add_data_source.py \
        --input /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite_max_fsf/train.jsonl \
        --output data/processed_jsonl/train.jsonl

    python scripts/preprocess/add_data_source.py \
        --input /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite_max_fsf/val.jsonl \
        --output data/processed_jsonl/val.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


def add_data_source(
    input_path: str,
    output_path: str,
    data_source: str = "cot_rewriter",
    max_samples: Optional[int] = None,
):
    """
    Add data_source field to each record in the JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        data_source: Value for data_source field (default: cot_rewriter)
        max_samples: Maximum number of samples to process (optional)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count total lines first
    print(f"Counting lines in {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines: {total_lines}")
    
    # Process file
    processed = 0
    skipped = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            if max_samples and processed >= max_samples:
                print(f"Reached max_samples limit: {max_samples}")
                break
            
            line = line.strip()
            if not line:
                skipped += 1
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {line_num}: {e}")
                skipped += 1
                continue
            
            # Add data_source field if not present
            if "data_source" not in record:
                record["data_source"] = data_source
            
            # Write to output
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1
            
            if processed % 10000 == 0:
                print(f"Processed {processed} records...")
    
    print(f"\nDone!")
    print(f"  Processed: {processed} records")
    print(f"  Skipped: {skipped} records")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add data_source field to JSONL files for VERL training"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--data-source", "-s",
        default="cot_rewriter",
        help="Value for data_source field (default: cot_rewriter)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    add_data_source(
        input_path=args.input,
        output_path=args.output,
        data_source=args.data_source,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
