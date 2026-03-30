#!/usr/bin/env python3
"""
Build COT rewrite training dataset from DeepMath format data with FSF filtering.

Input format (per line in JSONL from train_fsf.jsonl):
    {
        "question": "...",
        "reasoning": [
            {
                "thought": "...",
                "solution": "...",
                "fsf": 0.25,
                "failed_nodes": 1,
                "node_count": 4,
                ...
            },
            ...
        ]
    }

Features:
  - Filter by failed_nodes range (1-3)
  - Each reasoning becomes an independent sample
  - Train/val split is done at the question level

Output format:
    {
        "question": "...",
        "original_thought": "...",
        "solution": "...",
        "fsf": 0.25,
        "failed_nodes": 1,
        "node_count": 4,
        "system_prompt": "...",
        "user_prompt": "..."
    }
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools


# =============================================================================
# Prompt builder
# =============================================================================

def build_prompt(question: str, thought: str) -> Tuple[str, str]:
#     system_prompt = """\
# You are an expert in math and code. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the correct path and present it clearly.

# Follow these guidelines:
# 1. **Refine and Bridge**: Retain the core logic, including essential self-reflections and verification steps. Supplement logical leaps and clarify complex steps to ensure a smooth, coherent flow.
# 2. **Filter Invalid Paths**: Remove derivation branches within the 'Original Reasoning' that do not lead to the final answer (e.g., failed attempts, incorrect directions, or dead ends). Keep only the effective reasoning path.
# 3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

# **Output Rules:**
# - Directly output the rewritten reasoning process.
# - Do NOT mention that you are rewriting another reasoning process."""


    system_prompt = """\
You are an expert in math. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the logical core and transform it into a "Cognitive-Aligned" reasoning path.

Follow these guidelines:

1. **Refine and Bridge**: Retain the core logic, especially all self-reflection behaviors and verification steps. If the original process contains a moment where it realizes a mistake and pivots to a new strategy (e.g., "Wait, this is wrong because...", "Actually, let's try..."), YOU MUST PRESERVE THIS PIVOT. Supplement logical leaps to ensure a smooth, pedagogical flow.

2. **Preserve Cognitive Pivots**: Treat "trial -> error detection -> correction" sequences as essential cognitive bridges rather than invalid paths. Ensure the transition between an initial attempt, the realization of a flaw, and the subsequent correction is logically explicit and clear.

3. **Avoid Literal Repetition**: Rephrase the content in your own words to improve clarity and logical rigor. The final output should read like a master teacher's explanation, illustrating not just the correct answer, but the thinking process required to reach it.

**Output Rules:**
- Directly output the rewritten reasoning process.
- Do NOT mention that you are rewriting another reasoning process.
- Prioritize the completeness of the cognitive trajectory over absolute brevity."""

    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}

## Rewritten Reasoning Process
"""

    return system_prompt, user_prompt


# =============================================================================
# Data loading with FSF filtering
# =============================================================================

def _load_single_file_with_fsf(args: Tuple[str, int, int, int]) -> List[Dict[str, Any]]:
    """
    Load one FSF JSONL file with failed_nodes filtering.
    
    Args:
        args: (file_path, worker_id, min_failed_nodes, max_failed_nodes)
    
    Returns list of {"question": ..., "reasoning": [filtered_reasoning, ...]}
    """
    file_path, worker_id, min_failed_nodes, max_failed_nodes = args
    records = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                question = data.get("question", "").strip()
                reasoning_list = data.get("reasoning", [])
                
                if not question or not reasoning_list:
                    continue
                
                # Filter reasoning by failed_nodes (1-3)
                filtered_reasoning = []
                for r in reasoning_list:
                    # Check if it's a valid FSF result
                    if r.get("parse_failed") or r.get("fsf") is None:
                        continue
                    
                    failed_nodes = r.get("failed_nodes")
                    if failed_nodes is None:
                        continue
                    
                    # Filter: failed_nodes must be between min and max (inclusive)
                    if min_failed_nodes <= failed_nodes <= max_failed_nodes:
                        filtered_reasoning.append(r)
                
                # Only keep questions that have at least one valid reasoning
                if filtered_reasoning:
                    records.append({
                        "question": question,
                        "reasoning": filtered_reasoning
                    })
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading {file_path}: {e}")
    
    return records


def load_fsf_data(
    file_paths: List[str],
    min_failed_nodes: int = 1,
    max_failed_nodes: int = 3,
    num_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load FSF JSONL files with failed_nodes filtering."""
    if num_workers is None:
        num_workers = min(cpu_count(), max(1, len(file_paths)))

    print(f"  Loading {len(file_paths)} file(s) with {num_workers} workers...")
    print(f"  Filtering: failed_nodes between {min_failed_nodes} and {max_failed_nodes}")
    
    args_list = [(fp, i, min_failed_nodes, max_failed_nodes) for i, fp in enumerate(file_paths)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_load_single_file_with_fsf, args_list)

    records = list(itertools.chain.from_iterable(results))
    
    # Count total reasonings
    total_reasonings = sum(len(r["reasoning"]) for r in records)
    
    print(f"  Loaded {len(records)} questions with {total_reasonings} valid reasonings")
    print(f"  (filtered by failed_nodes in [{min_failed_nodes}, {max_failed_nodes}])")
    
    return records


# =============================================================================
# Entry building - each reasoning as independent sample
# =============================================================================

def build_entries_from_fsf(
    records: List[Dict[str, Any]],
    tokenizer=None,
    max_problem_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build entries where each reasoning is an independent sample.
    
    Each entry contains FSF metadata for analysis.
    """
    entries = []
    skipped = 0
    total = 0

    for rec in records:
        question = rec["question"]
        reasoning_list = rec["reasoning"]

        for r in reasoning_list:
            thought = r.get("thought", "").strip()
            if not thought:
                skipped += 1
                continue

            solution = r.get("solution", "")
            system_prompt, user_prompt = build_prompt(question, thought)

            if tokenizer is not None and max_problem_tokens is not None:
                try:
                    if len(tokenizer.encode(user_prompt)) > max_problem_tokens:
                        skipped += 1
                        continue
                except Exception:
                    pass

            entry = {
                "question": question,
                "original_thought": thought,
                "solution": solution,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                # FSF metadata
                "fsf": r.get("fsf"),
                "failed_nodes": r.get("failed_nodes"),
                "node_count": r.get("node_count"),
                "node_count_dict": r.get("node_count_dict"),
                "dot_graph": r.get("dot_graph"),
                "retry_count": r.get("retry_count", 0),
            }
            entries.append(entry)
            total += 1

    print(f"  Built {total} entries (skipped {skipped})")
    return entries


# =============================================================================
# Train / val split (no question overlap)
# =============================================================================

def split_by_question(
    records: List[Dict[str, Any]],
    train_ratio: float,
    seed: int,
    tokenizer=None,
    max_problem_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split questions into train/val first, then build entries.
    Guarantees no question overlap and shuffles both sets.
    """
    rng = random.Random(seed)

    # Shuffle questions
    shuffled_records = records.copy()
    rng.shuffle(shuffled_records)

    split_idx = int(len(shuffled_records) * train_ratio)
    train_records = shuffled_records[:split_idx]
    val_records = shuffled_records[split_idx:]

    print(f"  Questions: {len(train_records)} train / {len(val_records)} val")

    # Build entries - each reasoning becomes an independent sample
    train_entries = build_entries_from_fsf(train_records, tokenizer, max_problem_tokens)
    val_entries = build_entries_from_fsf(val_records, tokenizer, max_problem_tokens)

    # Shuffle entries
    rng.shuffle(train_entries)
    random.Random(seed + 1).shuffle(val_entries)

    return train_entries, val_entries


def sample_entries_by_failed_nodes(
    entries: List[Dict[str, Any]],
    nonzero_values: List[int] = [1, 2, 3],
    zero_ratio: float = 0.25,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample entries so that failed_nodes=0 accounts for `zero_ratio`,
    and the rest are sampled from `nonzero_values`.
    """
    rng = random.Random(seed)

    zero_entries = [e for e in entries if e.get("failed_nodes") == 0]
    nonzero_entries = [e for e in entries if e.get("failed_nodes") in nonzero_values]

    n_zero = len(zero_entries)
    n_nonzero = len(nonzero_entries)

    # target: zero / nonzero = zero_ratio / (1 - zero_ratio)
    target_zero = min(n_zero, int(n_nonzero * zero_ratio / (1 - zero_ratio)))
    target_nonzero = int(target_zero * (1 - zero_ratio) / zero_ratio)
    target_nonzero = min(target_nonzero, n_nonzero)

    if target_zero == 0 and target_nonzero == 0:
        return []

    sampled = []
    if target_zero > 0:
        sampled.extend(rng.sample(zero_entries, target_zero))
    if target_nonzero > 0:
        sampled.extend(rng.sample(nonzero_entries, target_nonzero))

    rng.shuffle(sampled)
    print(f"    Sampled {len(sampled)} entries ({target_zero} zero + {target_nonzero} nonzero)")
    return sampled


# =============================================================================
# Helpers
# =============================================================================

def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_statistics(entries: List[Dict[str, Any]], name: str = "Dataset"):
    total = len(entries)
    if total == 0:
        print(f"\n{name}: No entries")
        return
    
    avg_thought = sum(len(e.get("original_thought", "")) for e in entries) / total
    avg_sol = sum(len(e.get("solution", "")) for e in entries) / total
    
    # FSF statistics
    fsf_values = [e.get("fsf") for e in entries if e.get("fsf") is not None]
    failed_nodes_list = [e.get("failed_nodes") for e in entries if e.get("failed_nodes") is not None]
    
    print(f"\n{name} Statistics:")
    print(f"  Total entries:               {total}")
    print(f"  Avg original_thought length: {avg_thought:.0f} chars")
    print(f"  Avg solution length:         {avg_sol:.0f} chars")
    
    if fsf_values:
        print(f"  Avg FSF:                     {sum(fsf_values)/len(fsf_values):.4f}")
        print(f"  FSF range:                   {min(fsf_values):.4f} - {max(fsf_values):.4f}")
    
    if failed_nodes_list:
        failed_dist = {}
        for fn in failed_nodes_list:
            failed_dist[fn] = failed_dist.get(fn, 0) + 1
        print(f"  Failed nodes distribution:")
        for fn in sorted(failed_dist.keys()):
            print(f"    {fn} failed nodes: {failed_dist[fn]} entries")


def get_default_config() -> Dict[str, Any]:
    return {
        "input_paths": [
            "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl",
        ],
        "output_dir": "/data/luoys/data/cdad/cot_rewrite_deepmath_fsf",
        "min_failed_nodes": 0,      # Minimum failed_nodes to include
        "max_failed_nodes": 3,      # Maximum failed_nodes to include
        "target_zero_ratio": 0.25,  # Target ratio of failed_nodes=0 samples
        "nonzero_failed_nodes": [1, 2, 3],  # Allowed non-zero failed_nodes values
        "tokenizer_path": None,
        "max_problem_tokens": None,
        "train_ratio": 0.99,
        "random_seed": 42,
        "num_workers": None,
    }


# =============================================================================
# Main
# =============================================================================

def main(config: Optional[Dict[str, Any]] = None):
    if config is None:
        config = get_default_config()

    input_paths = config["input_paths"]
    output_dir = config["output_dir"]
    min_failed_nodes = config.get("min_failed_nodes", 0)
    max_failed_nodes = config.get("max_failed_nodes", 3)
    target_zero_ratio = config.get("target_zero_ratio", 0.25)
    nonzero_failed_nodes = config.get("nonzero_failed_nodes", [1, 2, 3])
    tokenizer_path = config.get("tokenizer_path")
    max_problem_tokens = config.get("max_problem_tokens")
    train_ratio = config.get("train_ratio", 0.99)
    random_seed = config.get("random_seed", 42)
    num_workers = config.get("num_workers")

    print("=" * 60)
    print("COT Rewrite Dataset Builder (DeepMath with FSF Filter)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  INPUT_PATHS:       {input_paths}")
    print(f"  OUTPUT_DIR:        {output_dir}")
    print(f"  MIN_FAILED_NODES:  {min_failed_nodes}")
    print(f"  MAX_FAILED_NODES:  {max_failed_nodes}")
    print(f"  TARGET_ZERO_RATIO: {target_zero_ratio}")
    print(f"  NONZERO_FAILED_NODES: {nonzero_failed_nodes}")
    print(f"  TRAIN_RATIO:       {train_ratio}")
    print(f"  RANDOM_SEED:       {random_seed}")
    print(f"  TOKENIZER_PATH:    {tokenizer_path}")
    print(f"  MAX_PROBLEM_TOKENS:{max_problem_tokens}")
    print(f"  NUM_WORKERS:       {num_workers or 'auto'}")

    # Tokenizer (optional)
    tokenizer = None
    if tokenizer_path and max_problem_tokens:
        print("\nLoading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"  Loaded: {tokenizer_path}")

    # Step 1: Load with FSF filtering
    print("\n" + "-" * 60)
    print("Step 1: Loading FSF data with failed_nodes filtering...")
    print("-" * 60)
    records = load_fsf_data(
        input_paths,
        min_failed_nodes=min_failed_nodes,
        max_failed_nodes=max_failed_nodes,
        num_workers=num_workers
    )

    # Step 2: Split & build entries
    print("\n" + "-" * 60)
    print("Step 2: Splitting by question and building entries...")
    print("  (Each reasoning becomes an independent sample)")
    print("-" * 60)
    train_entries, val_entries = split_by_question(
        records, train_ratio, random_seed, tokenizer, max_problem_tokens
    )

    # Step 2.5: Sample to control failed_nodes distribution
    print("\n" + "-" * 60)
    print("Step 2.5: Sampling entries to control failed_nodes distribution...")
    print(f"  Target: {target_zero_ratio*100:.0f}% failed_nodes=0, rest from {nonzero_failed_nodes}")
    print("-" * 60)
    train_entries = sample_entries_by_failed_nodes(
        train_entries, nonzero_failed_nodes, target_zero_ratio, random_seed
    )
    val_entries = sample_entries_by_failed_nodes(
        val_entries, nonzero_failed_nodes, target_zero_ratio, random_seed + 1
    )

    print_statistics(train_entries, "Train Set")
    print_statistics(val_entries, "Validation Set")

    # Step 3: Save
    print("\n" + "-" * 60)
    print("Step 3: Saving data...")
    print("-" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    save_jsonl(train_entries, str(train_path))
    save_jsonl(val_entries, str(val_path))

    print(f"\nOutput saved to: {output_path.absolute()}")
    print(f"  Train: {train_path.name} ({len(train_entries)} entries)")
    print(f"  Val:   {val_path.name} ({len(val_entries)} entries)")

    # Step 4: Save config & log
    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "counts": {
            "total_questions": len(records),
            "train_entries": len(train_entries),
            "val_entries": len(val_entries),
        },
    }
    log_path = output_path / "build_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build COT rewrite dataset from DeepMath FSF data"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--input", type=str, nargs="+", help="Input FSF JSONL file path(s)"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--min-failed-nodes",
        type=int,
        default=1,
        help="Minimum failed_nodes to include (default: 1)"
    )
    parser.add_argument(
        "--max-failed-nodes",
        type=int,
        default=3,
        help="Maximum failed_nodes to include (default: 3)"
    )
    parser.add_argument("--train-ratio", type=float, default=0.99, help="Train ratio (default: 0.99)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path for token-length filtering")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max user_prompt token length")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes (default: auto)")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = get_default_config()
        if args.input:
            config["input_paths"] = args.input
        if args.output_dir:
            config["output_dir"] = args.output_dir
        config["min_failed_nodes"] = args.min_failed_nodes
        config["max_failed_nodes"] = args.max_failed_nodes
        config["train_ratio"] = args.train_ratio
        config["random_seed"] = args.seed
        if args.tokenizer:
            config["tokenizer_path"] = args.tokenizer
        if args.max_tokens:
            config["max_problem_tokens"] = args.max_tokens
        if args.num_workers:
            config["num_workers"] = args.num_workers

    main(config)


"""
# Example usage:

# Default: failed_nodes in [1, 3]
python3 build_cot_rewrite_from_deepmath.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_fsf_1to3

# Custom failed_nodes range [2, 5]
python3 build_cot_rewrite_from_deepmath.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_fsf_2to5 \
    --min-failed-nodes 2 \
    --max-failed-nodes 5

# With token-length filtering
python3 build_cot_rewrite_from_deepmath.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl \
    --output-dir /data/luoys/data/cdad/cot_rewrite_deepmath_fsf \
    --tokenizer /data/luoys/modelscope_cache/models/Qwen/Qwen3-4B \
    --max-tokens 4500
"""
