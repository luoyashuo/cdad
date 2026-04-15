#!/usr/bin/env python3
"""
Build COT rewrite training dataset from DeepMath plain ShareGPT-format data.

Input format (per line in JSONL from deepmath_plain_30k/train.jsonl):
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<think>...</think> final response"}
        ]
    }

This script follows the prompt/output style of build_cot_rewrite_from_deepmath.py,
but replaces FSF loading with direct extraction from ShareGPT-format plain data.

Output format:
    {
        "question": "...",
        "original_thought": "...",
        "solution": "...",
        "system_prompt": "...",
        "user_prompt": "..."
    }
"""

import argparse
import json
import os
import random
import itertools
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Prompt builder
# =============================================================================

def build_prompt(question: str, thought: str) -> Tuple[str, str]:
    system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the correct path and present it clearly.

Follow these guidelines:
1. **Refine and Bridge**: Retain the core logic, including essential self-reflections and verification steps. Supplement logical leaps and clarify complex steps to ensure a smooth, coherent flow.
2. **Filter Invalid Paths**: Remove derivation branches within the 'Original Reasoning' that do not lead to the final answer (e.g., failed attempts, incorrect directions, or dead ends). Keep only the effective reasoning path.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

**Output Rules:**
- Directly output the rewritten reasoning process.
- Do NOT mention that you are rewriting another reasoning process."""

    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}

## Rewritten Reasoning Process
"""

    return system_prompt, user_prompt


# =============================================================================
# Data loading
# =============================================================================

def _get_message_text(message: Dict[str, Any]) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or message.get("value") or "").strip()


def extract_think_and_solution(content: str) -> Tuple[str, str]:
    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise ValueError("Missing valid <think>...</think> block")

    thought = content[start_idx + len(start_tag):end_idx].strip()
    solution = content[end_idx + len(end_tag):].strip()
    if not thought:
        raise ValueError("Empty thought extracted from <think> block")

    return thought, solution


def _load_single_plain_file(args: Tuple[str, int]) -> Dict[str, Any]:
    file_path, worker_id = args
    entries: List[Dict[str, Any]] = []
    stats = {
        "total_lines": 0,
        "valid_entries": 0,
        "skipped_invalid_json": 0,
        "skipped_missing_messages": 0,
        "skipped_missing_think": 0,
        "skipped_empty_question": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats["total_lines"] += 1

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped_invalid_json"] += 1
                    continue

                conversations = data.get("conversations", [])
                if not isinstance(conversations, list) or len(conversations) < 2:
                    stats["skipped_missing_messages"] += 1
                    continue

                question = _get_message_text(conversations[0])
                assistant_content = _get_message_text(conversations[1])

                if not question:
                    stats["skipped_empty_question"] += 1
                    continue

                try:
                    thought, solution = extract_think_and_solution(assistant_content)
                except ValueError:
                    stats["skipped_missing_think"] += 1
                    continue

                entries.append({
                    "question": question,
                    "original_thought": thought,
                    "solution": solution,
                })
                stats["valid_entries"] += 1
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading {file_path}: {e}")

    return {"entries": entries, "stats": stats}


def load_plain_data(
    file_paths: List[str],
    num_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if num_workers is None:
        num_workers = min(cpu_count(), max(1, len(file_paths)))

    print(f"  Loading {len(file_paths)} file(s) with {num_workers} workers...")
    args_list = [(fp, i) for i, fp in enumerate(file_paths)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_load_single_plain_file, args_list)

    entries = list(itertools.chain.from_iterable(result["entries"] for result in results))

    merged_stats: Dict[str, int] = defaultdict(int)
    for result in results:
        for key, value in result["stats"].items():
            merged_stats[key] += value

    print(f"  Loaded {len(entries)} valid entries from plain ShareGPT data")
    print(f"  Skipped invalid JSON:        {merged_stats['skipped_invalid_json']}")
    print(f"  Skipped missing messages:    {merged_stats['skipped_missing_messages']}")
    print(f"  Skipped missing think block: {merged_stats['skipped_missing_think']}")
    print(f"  Skipped empty question:      {merged_stats['skipped_empty_question']}")

    return entries, dict(merged_stats)


# =============================================================================
# Entry building / split
# =============================================================================

def build_output_entries(
    raw_entries: List[Dict[str, Any]],
    tokenizer=None,
    max_problem_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    entries = []
    skipped = 0

    for item in raw_entries:
        question = item["question"]
        thought = item["original_thought"]
        solution = item.get("solution", "")
        system_prompt, user_prompt = build_prompt(question, thought)

        if tokenizer is not None and max_problem_tokens is not None:
            try:
                if len(tokenizer.encode(user_prompt)) > max_problem_tokens:
                    skipped += 1
                    continue
            except Exception:
                pass

        entries.append({
            "question": question,
            "original_thought": thought,
            "solution": solution,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

    print(f"  Built {len(entries)} entries (skipped {skipped} for length)")
    return entries


def select_top_thought_length_entries(
    raw_entries: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if top_k is None or top_k <= 0:
        print(f"  Thought-length filtering disabled, using all {len(raw_entries)} entries")
        return raw_entries

    if len(raw_entries) <= top_k:
        print(f"  Entries count ({len(raw_entries)}) <= top_k ({top_k}), using all entries")
        return raw_entries

    sorted_entries = sorted(
        raw_entries,
        key=lambda x: len(x.get("original_thought", "")),
        reverse=True,
    )
    selected_entries = sorted_entries[:top_k]

    min_len = len(selected_entries[-1].get("original_thought", ""))
    max_len = len(selected_entries[0].get("original_thought", ""))
    print(f"  Selected top {len(selected_entries)} entries by thought length")
    print(f"  Thought length range in selected set: {min_len} - {max_len} chars")
    return selected_entries


def select_mixed_thought_length_entries(
    raw_entries: List[Dict[str, Any]],
    total_k: Optional[int],
    top_pool_k: int = 5000,
    top_ratio: float = 0.7,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if total_k is None or total_k <= 0:
        print(f"  Mixed selection disabled, using all {len(raw_entries)} entries")
        return raw_entries

    if not 0.0 <= top_ratio <= 1.0:
        raise ValueError(f"top_ratio must be in [0, 1], got {top_ratio}")

    if len(raw_entries) <= total_k:
        print(f"  Entries count ({len(raw_entries)}) <= total_k ({total_k}), using all entries")
        return raw_entries

    sorted_entries = sorted(
        raw_entries,
        key=lambda x: len(x.get("original_thought", "")),
        reverse=True,
    )

    top_pool = sorted_entries[:min(top_pool_k, len(sorted_entries))]
    rest_pool = sorted_entries[len(top_pool):]

    target_top = min(int(total_k * top_ratio), len(top_pool))
    target_random = total_k - target_top

    rng = random.Random(seed)

    selected_top = rng.sample(top_pool, target_top) if target_top < len(top_pool) else top_pool.copy()

    selected_random_count = min(target_random, len(rest_pool))
    selected_random = (
        rng.sample(rest_pool, selected_random_count)
        if selected_random_count < len(rest_pool)
        else rest_pool.copy()
    )

    if len(selected_top) + len(selected_random) < total_k:
        used_ids = {id(item) for item in selected_top}
        unused_top = [item for item in top_pool if id(item) not in used_ids]
        need = total_k - len(selected_top) - len(selected_random)
        filler = rng.sample(unused_top, need) if need < len(unused_top) else unused_top
        selected_top.extend(filler)

    selected_entries = selected_top + selected_random
    rng.shuffle(selected_entries)

    selected_lengths = [len(item.get("original_thought", "")) for item in selected_entries]
    print(f"  Mixed selection: {len(selected_top)} from top-{len(top_pool)} thought lengths + {len(selected_random)} random from remaining pool")
    print(f"  Selected total: {len(selected_entries)} entries")
    print(f"  Thought length range in selected set: {min(selected_lengths)} - {max(selected_lengths)} chars")
    return selected_entries


def split_by_question(
    raw_entries: List[Dict[str, Any]],
    train_ratio: float,
    seed: int,
    tokenizer=None,
    max_problem_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in raw_entries:
        grouped[item["question"]].append(item)

    questions = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(questions)

    split_idx = int(len(questions) * train_ratio)
    train_questions = set(questions[:split_idx])
    val_questions = set(questions[split_idx:])

    train_raw = []
    val_raw = []
    for question, items in grouped.items():
        if question in train_questions:
            train_raw.extend(items)
        else:
            val_raw.extend(items)

    print(f"  Questions: {len(train_questions)} train / {len(val_questions)} val")
    print(f"  Raw entries: {len(train_raw)} train / {len(val_raw)} val")

    train_entries = build_output_entries(train_raw, tokenizer, max_problem_tokens)
    val_entries = build_output_entries(val_raw, tokenizer, max_problem_tokens)

    rng.shuffle(train_entries)
    random.Random(seed + 1).shuffle(val_entries)

    split_stats = {
        "unique_questions": len(questions),
        "train_questions": len(train_questions),
        "val_questions": len(val_questions),
        "train_raw_entries": len(train_raw),
        "val_raw_entries": len(val_raw),
    }
    return train_entries, val_entries, split_stats


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

    print(f"\n{name} Statistics:")
    print(f"  Total entries:               {total}")
    print(f"  Avg original_thought length: {avg_thought:.0f} chars")
    print(f"  Avg solution length:         {avg_sol:.0f} chars")


def get_default_config() -> Dict[str, Any]:
    return {
        "input_paths": [
            "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl",
        ],
        "output_dir": "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_plain_30k",
        "top_thought_k": None,
        "mixed_total_k": None,
        "mixed_top_pool_k": 5000,
        "mixed_top_ratio": 0.7,
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
    top_thought_k = config.get("top_thought_k")
    mixed_total_k = config.get("mixed_total_k")
    mixed_top_pool_k = config.get("mixed_top_pool_k", 5000)
    mixed_top_ratio = config.get("mixed_top_ratio", 0.7)
    tokenizer_path = config.get("tokenizer_path")
    max_problem_tokens = config.get("max_problem_tokens")
    train_ratio = config.get("train_ratio", 0.99)
    random_seed = config.get("random_seed", 42)
    num_workers = config.get("num_workers")

    print("=" * 60)
    print("COT Rewrite Dataset Builder (DeepMath Plain)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  INPUT_PATHS:        {input_paths}")
    print(f"  OUTPUT_DIR:         {output_dir}")
    print(f"  TOP_THOUGHT_K:      {top_thought_k}")
    print(f"  MIXED_TOTAL_K:      {mixed_total_k}")
    print(f"  MIXED_TOP_POOL_K:   {mixed_top_pool_k}")
    print(f"  MIXED_TOP_RATIO:    {mixed_top_ratio}")
    print(f"  TRAIN_RATIO:        {train_ratio}")
    print(f"  RANDOM_SEED:        {random_seed}")
    print(f"  TOKENIZER_PATH:     {tokenizer_path}")
    print(f"  MAX_PROBLEM_TOKENS: {max_problem_tokens}")
    print(f"  NUM_WORKERS:        {num_workers or 'auto'}")

    tokenizer = None
    if tokenizer_path and max_problem_tokens:
        print("\nLoading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"  Loaded: {tokenizer_path}")

    print("\n" + "-" * 60)
    print("Step 1: Loading plain ShareGPT data...")
    print("-" * 60)
    raw_entries, load_stats = load_plain_data(input_paths, num_workers=num_workers)

    print("\n" + "-" * 60)
    print("Step 2: Selecting entries by thought length...")
    print("-" * 60)
    if mixed_total_k is not None:
        selected_raw_entries = select_mixed_thought_length_entries(
            raw_entries,
            total_k=mixed_total_k,
            top_pool_k=mixed_top_pool_k,
            top_ratio=mixed_top_ratio,
            seed=random_seed,
        )
    else:
        selected_raw_entries = select_top_thought_length_entries(
            raw_entries, top_k=top_thought_k
        )

    print("\n" + "-" * 60)
    print("Step 3: Splitting by question and building entries...")
    print("  (Each input sample becomes an independent sample)")
    print("-" * 60)
    train_entries, val_entries, split_stats = split_by_question(
        selected_raw_entries, train_ratio, random_seed, tokenizer, max_problem_tokens
    )

    print_statistics(train_entries, "Train Set")
    print_statistics(val_entries, "Validation Set")

    print("\n" + "-" * 60)
    print("Step 4: Saving data...")
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

    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "load_stats": load_stats,
        "selection_stats": {
            "raw_entries_before_selection": len(raw_entries),
            "raw_entries_after_selection": len(selected_raw_entries),
            "top_thought_k": top_thought_k,
            "mixed_total_k": mixed_total_k,
            "mixed_top_pool_k": mixed_top_pool_k,
            "mixed_top_ratio": mixed_top_ratio,
        },
        "split_stats": split_stats,
        "counts": {
            "total_valid_entries": len(raw_entries),
            "selected_entries": len(selected_raw_entries),
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
        description="Build COT rewrite dataset from DeepMath plain ShareGPT data"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--input", type=str, nargs="+", help="Input plain JSONL file path(s)"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--top-thought-k", type=int, default=None, help="Keep top K entries by original_thought length before split")
    parser.add_argument("--mixed-total-k", type=int, default=None, help="Build a mixed subset with this total size before split")
    parser.add_argument("--mixed-top-pool-k", type=int, default=5000, help="Top-length pool size used by mixed selection")
    parser.add_argument("--mixed-top-ratio", type=float, default=0.7, help="Ratio drawn from the top-length pool in mixed selection")
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
        if args.top_thought_k is not None:
            config["top_thought_k"] = args.top_thought_k
        if args.mixed_total_k is not None:
            config["mixed_total_k"] = args.mixed_total_k
        config["mixed_top_pool_k"] = args.mixed_top_pool_k
        config["mixed_top_ratio"] = args.mixed_top_ratio
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
python3 build_cot_rewrite_from_deepmath_plain.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_plain_30k \
    --train-ratio 0.99

python3 build_cot_rewrite_from_deepmath_plain.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_plain_top5k \
    --top-thought-k 5000 \
    --train-ratio 0.99

python3 build_cot_rewrite_from_deepmath_plain.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_plain_mix5k \
    --mixed-total-k 5000 \
    --mixed-top-pool-k 5000 \
    --mixed-top-ratio 0.7 \
    --train-ratio 0.99

python3 build_cot_rewrite_from_deepmath_plain.py \
    --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl \
    --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_plain_30k_qwen \
    --tokenizer /data/luoys/modelscope_cache/models/Qwen/Qwen3-4B \
    --max-tokens 4500
"""
