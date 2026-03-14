"""
CoT Rewrite - Step 5: Build ShareGPT Dataset

Reads fsf_tag_output.jsonl (from step 4), applies filters and selection
strategy, then writes ShareGPT-format JSONL for fine-tuning.

Usage:
    python -m inference.build_sharegpt --config inference/config_example.yaml
    python -m inference.build_sharegpt --config cfg.yaml --strategy priority_correct
    python -m inference.build_sharegpt --config cfg.yaml \\
        --filter-fsf-positive --filter-correct-only \\
        --input-file /path/to/fsf_tag_output.jsonl \\
        --output-file /path/to/train.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_best_rewrite(
    rewrites: List[Dict[str, Any]],
    filter_correct_only: bool,
    filter_fsf_positive: bool,
    strategy: str,
) -> Optional[Dict[str, Any]]:
    """
    Apply filters and select one rewrite according to strategy.

    Args:
        rewrites: list of rewrite dicts with keys rewrite_cot, is_correct, fsf
        filter_correct_only: keep only is_correct=True rewrites
        filter_fsf_positive: keep only fsf > 0 rewrites
        strategy: random / max_fsf / priority_correct

    Returns:
        selected rewrite dict, or None if no candidates remain
    """
    candidates = list(rewrites)

    if filter_correct_only:
        candidates = [r for r in candidates if r.get("is_correct") is True]

    if filter_fsf_positive:
        candidates = [r for r in candidates if (r.get("fsf") or 0) > 0]

    if not candidates:
        return None

    if strategy == "random":
        return random.choice(candidates)

    elif strategy == "max_fsf":
        # Sort descending by fsf (None treated as -inf)
        candidates.sort(key=lambda r: r.get("fsf") or 0.0, reverse=True)
        return candidates[0]

    elif strategy == "priority_correct":
        # Prefer correct, then by fsf descending
        correct_cands = [r for r in candidates if r.get("is_correct") is True]
        pool = correct_cands if correct_cands else candidates
        pool.sort(key=lambda r: r.get("fsf") or 0.0, reverse=True)
        return pool[0] if pool else None

    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")


def build_dataset(
    input_file: str,
    output_file: str,
    filter_correct_only: bool,
    filter_fsf_positive: bool,
    strategy: str,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Build ShareGPT dataset from fsf_tag_output.jsonl.

    Returns stats dict: total, selected, skipped.
    """
    random.seed(seed)
    records = _read_jsonl(input_file)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total = len(records)
    selected = 0
    skipped = 0

    with open(output_file, "w") as fout:
        for record in records:
            rewrites = record.get("rewrite_cots", [])
            best = select_best_rewrite(
                rewrites,
                filter_correct_only=filter_correct_only,
                filter_fsf_positive=filter_fsf_positive,
                strategy=strategy,
            )
            if best is None:
                skipped += 1
                continue

            question = record["question"]
            rewrite_cot = best["rewrite_cot"]
            solution = record.get("solution", "")

            # Build ShareGPT format
            assistant_value = f"<think>\n{rewrite_cot}\n</think>\n\n{solution}"
            out = {
                "conversations": [
                    {"role": "user", "value": question},
                    {"role": "assistant", "value": assistant_value},
                ]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            selected += 1

    return {"total": total, "selected": selected, "skipped": skipped}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ShareGPT dataset from CoT rewrite pipeline output (step 5)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--filter-fsf-positive",
        action="store_true",
        default=None,
        help="Only keep rewrites with FSF > 0 (overrides config)",
    )
    parser.add_argument(
        "--filter-correct-only",
        action="store_true",
        default=None,
        help="Only keep rewrites where is_correct=True (overrides config)",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "max_fsf", "priority_correct"],
        default=None,
        help="Selection strategy (overrides config)",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Override input file path (default: {output_dir}/intermediate/fsf_tag_output.jsonl)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Override output file path (default: {output_dir}/final/{output_filename})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Add verl to path
    verl_path = os.path.join(os.path.dirname(__file__), "..", "verl")
    if verl_path not in sys.path:
        sys.path.insert(0, os.path.abspath(verl_path))

    from inference.utils.config_loader import load_config
    from inference.utils.checkpoint import CheckpointManager

    config = load_config(args.config)
    exp_name = config["experiment_name"]
    output_dir = os.path.join(config["output_root"], exp_name)

    db_cfg = config.get("dataset_build", {})
    filters_cfg = db_cfg.get("filters", {})

    # Determine filter flags (CLI overrides config)
    filter_fsf_positive = (
        args.filter_fsf_positive if args.filter_fsf_positive is not None
        else filters_cfg.get("fsf_positive", True)
    )
    filter_correct_only = (
        args.filter_correct_only if args.filter_correct_only is not None
        else filters_cfg.get("correct_only", True)
    )
    strategy = args.strategy or db_cfg.get("selection_strategy", "priority_correct")

    # Determine input/output paths
    if args.input_file:
        input_file = args.input_file
    else:
        # Try checkpoint first
        ckpt_mgr = CheckpointManager(output_dir)
        ckpt_mgr.load()
        input_file = ckpt_mgr.get_step_output("fsf_tag")
        if not input_file or not os.path.exists(input_file):
            input_file = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        print("Run pipeline.py first (steps 1-4).", file=sys.stderr)
        sys.exit(1)

    if args.output_file:
        output_file = args.output_file
    else:
        final_dir = os.path.join(output_dir, "final")
        output_filename = db_cfg.get("output_filename", "train_sharegpt.jsonl")
        output_file = os.path.join(final_dir, output_filename)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Building ShareGPT dataset:")
    print(f"  Input:   {input_file}")
    print(f"  Output:  {output_file}")
    print(f"  Filters: correct_only={filter_correct_only}, fsf_positive={filter_fsf_positive}")
    print(f"  Strategy: {strategy}")

    stats = build_dataset(
        input_file=input_file,
        output_file=output_file,
        filter_correct_only=filter_correct_only,
        filter_fsf_positive=filter_fsf_positive,
        strategy=strategy,
        seed=config.get("seed", 42),
    )

    print(f"\nDone:")
    print(f"  Total records:  {stats['total']}")
    print(f"  Selected:       {stats['selected']}")
    print(f"  Skipped:        {stats['skipped']}")
    print(f"  Output:         {output_file}")


if __name__ == "__main__":
    main()
