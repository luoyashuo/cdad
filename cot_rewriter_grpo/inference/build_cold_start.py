#!/usr/bin/env python3
"""
Build cold start dataset from rewrites.jsonl for instruction tuning.

This script:
1. Reads rewrites.jsonl containing original and rewritten reasoning
2. Builds ShareGPT format dataset for instruction tuning
3. Splits into train/val sets and saves to output directory

Input format (rewrites.jsonl):
    {
        "id": "...",
        "question": "...",
        "original_thought": "...",
        "answer": "...",
        "rewrite_results": [
            {
                "rewrite_reasoning": "...",
                "success": true
            }
        ]
    }

Output format (ShareGPT):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

Usage:
    python build_cold_start.py
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Input/Output paths
INPUT_PATH = "/data/luoys/data/cdad/rewritten_output/llama3_3b_train_top2000/ckpt/rewrites.jsonl"
OUTPUT_DIR = "/data/luoys/data/cdad/cold_start/llama3_3b_instruct_top2k"

# System prompt
SYSTEM_PROMPT = """\
You are an expert in math and code. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the correct path and present it clearly.

Follow these guidelines:
1. **Refine and Bridge**: Retain the core logic, including essential self-reflections and verification steps. Supplement logical leaps and clarify complex steps to ensure a smooth, coherent flow.
2. **Filter Invalid Paths**: Remove derivation branches within the 'Original Reasoning' that do not lead to the final answer (e.g., failed attempts, incorrect directions, or dead ends). Keep only the effective reasoning path.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

**Output Rules:**
- Directly output the rewritten reasoning process.
- Do NOT mention that you are rewriting another reasoning process."""


def build_user_prompt(question: str, original_thought: str) -> str:
    """Build user prompt from question and original thought."""
    return f"""\
## Problem
{question}

## Original Reasoning Process
{original_thought}"""


def build_sharegpt_entry(question: str, original_thought: str, rewrite_reasoning: str) -> Dict[str, Any]:
    """Build a single ShareGPT format entry."""
    user_prompt = build_user_prompt(question, original_thought)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"<rewrite_reasoning>{rewrite_reasoning.strip()}</rewrite_reasoning>"}
    ]

    return {"messages": messages}


def read_rewrites(input_path: str) -> List[Dict[str, Any]]:
    """
    Read rewrites.jsonl and extract valid entries.

    Returns list of entries with question, original_thought, rewrite_reasoning
    """
    entries = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading rewrites.jsonl"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON: {e}")
                continue

            question = data.get("question", "").strip()
            original_thought = data.get("original_thought", "").strip()
            rewrite_results = data.get("rewrite_results", [])

            # Skip if missing required fields
            if not question or not original_thought:
                continue

            # Extract successful rewrite reasoning
            rewrite_reasoning = None
            for result in rewrite_results:
                if result.get("success", False):
                    reasoning = result.get("rewrite_reasoning", "").strip()
                    if reasoning:
                        rewrite_reasoning = reasoning
                        break

            if rewrite_reasoning:
                entries.append({
                    "question": question,
                    "original_thought": original_thought,
                    "rewrite_reasoning": rewrite_reasoning
                })

    print(f"Loaded {len(entries)} valid entries from {input_path}")
    return entries


def convert_to_sharegpt(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert entries to ShareGPT format."""
    sharegpt_entries = []

    for entry in tqdm(entries, desc="Converting to ShareGPT format"):
        sharegpt_entry = build_sharegpt_entry(
            question=entry["question"],
            original_thought=entry["original_thought"],
            rewrite_reasoning=entry["rewrite_reasoning"]
        )
        sharegpt_entries.append(sharegpt_entry)

    return sharegpt_entries


def split_train_val(entries: List[Dict[str, Any]], train_ratio: float = 0.98, seed: int = 42):
    """Split entries into train and validation sets."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_entries = shuffled[:split_idx]
    val_entries = shuffled[split_idx:]

    return train_entries, val_entries


def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """Save entries to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(entries)} entries to {output_path}")


def print_statistics(entries: List[Dict[str, Any]], name: str = "Dataset"):
    """Print dataset statistics."""
    total = len(entries)
    if total == 0:
        print(f"\n{name}: 0 entries")
        return

    # Calculate average lengths
    total_user_len = 0
    total_assistant_len = 0

    for entry in entries:
        for msg in entry.get("messages", []):
            if msg.get("role") == "user":
                total_user_len += len(msg.get("content", ""))
            elif msg.get("role") == "assistant":
                total_assistant_len += len(msg.get("content", ""))

    avg_user_len = total_user_len / total
    avg_assistant_len = total_assistant_len / total

    print(f"\n{name} Statistics:")
    print(f"  Total entries: {total}")
    print(f"  Avg user message length: {avg_user_len:.0f} chars")
    print(f"  Avg assistant message length: {avg_assistant_len:.0f} chars")


def main():
    print("=" * 60)
    print("Building Cold Start Dataset from Rewrites")
    print("=" * 60)

    # Check input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found: {INPUT_PATH}")
        return

    # Read rewrites
    entries = read_rewrites(INPUT_PATH)

    if not entries:
        print("Error: No valid entries found!")
        return

    print_statistics(entries, "Raw Data")

    # Convert to ShareGPT format
    sharegpt_entries = convert_to_sharegpt(entries)
    print_statistics(sharegpt_entries, "ShareGPT Format")

    # Split into train/val
    train_ratio = 0.98
    seed = 42
    train_entries, val_entries = split_train_val(sharegpt_entries, train_ratio, seed)

    print_statistics(train_entries, "Train Set")
    print_statistics(val_entries, "Validation Set")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save train/val files
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    save_jsonl(train_entries, str(train_path))
    save_jsonl(val_entries, str(val_path))

    print(f"\n{'=' * 60}")
    print(f"Output saved to: {output_path.absolute()}")
    print(f"  Train: {train_path.name} ({len(train_entries)} entries)")
    print(f"  Val:   {val_path.name} ({len(val_entries)} entries)")


if __name__ == "__main__":
    main()
