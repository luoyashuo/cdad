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

    # With separate dataset build config (recommended):
    python -m inference.build_sharegpt \\
        --config inference/config_example.yaml \\
        --dataset-config inference/build_dataset_config_example.yaml
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
    filter_fsf_zero: bool,
    strategy: str,
) -> Optional[Dict[str, Any]]:
    """
    Apply filters and select one rewrite according to strategy.

    Args:
        rewrites: list of rewrite dicts with keys rewrite_cot, is_correct, fsf, ged
        filter_correct_only: keep only is_correct=True rewrites
        filter_fsf_positive: keep only fsf > 0 rewrites
        filter_fsf_zero: keep only fsf == 0 rewrites (mutually exclusive with filter_fsf_positive)
        strategy: random / max_fsf / priority_correct / min_ged

    Returns:
        selected rewrite dict, or None if no candidates remain
    """
    candidates = list(rewrites)

    if filter_correct_only:
        candidates = [r for r in candidates if r.get("is_correct") is True]

    if filter_fsf_positive:
        candidates = [r for r in candidates if (r.get("fsf") or 0) > 0]
    elif filter_fsf_zero:
        candidates = [r for r in candidates if (r.get("fsf") or 0) == 0]

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

    elif strategy == "min_ged":
        # Select the rewrite with the lowest GED to the original CoT.
        # Candidates with ged=None (parse/timeout failures) are deprioritized.
        valid = [r for r in candidates if r.get("ged") is not None]
        pool = valid if valid else candidates
        pool.sort(key=lambda r: r.get("ged") if r.get("ged") is not None else float("inf"))
        return pool[0]

    else:
        raise ValueError(f"Unknown selection strategy: {strategy!r}")


def print_tag_distribution(records: List[Dict[str, Any]], output_dir: str) -> None:
    """Print tag distribution stats and persist to {output_dir}/stats/."""
    from datetime import datetime

    total_samples  = len(records)
    total_rewrites = 0
    correct_count  = 0
    fsf_pos_count  = 0
    both_count     = 0
    fsf_values     = []
    per_sample     = []

    for rec in records:
        rewrites  = rec.get("rewrite_cots", [])
        n         = len(rewrites)
        n_correct = sum(1 for r in rewrites if r.get("is_correct") is True)
        n_fsf_pos = sum(1 for r in rewrites if (r.get("fsf") or 0) > 0)
        n_both    = sum(1 for r in rewrites if r.get("is_correct") is True and (r.get("fsf") or 0) > 0)
        total_rewrites += n
        correct_count  += n_correct
        fsf_pos_count  += n_fsf_pos
        both_count     += n_both
        fsf_values.extend((r.get("fsf") or 0) for r in rewrites)
        per_sample.append((n_correct, n_fsf_pos, n_both))

    samples_any_correct = sum(1 for c, _, _ in per_sample if c > 0)
    samples_any_fsf_pos = sum(1 for _, fp, _ in per_sample if fp > 0)
    samples_any_both    = sum(1 for _, _, b in per_sample if b > 0)
    fsf_pos_vals        = [v for v in fsf_values if v > 0]

    def pct(a, b):   return f"{100*a/b:.1f}%" if b > 0 else "N/A"
    def avg(lst):    return sum(lst) / len(lst) if lst else 0
    def med(lst):
        if not lst: return 0
        s = sorted(lst); n = len(s)
        return (s[n//2-1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]

    lines = [
        "=" * 60,
        "  Tag Distribution Report",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "[Samples]",
        f"  Total                    : {total_samples}",
        f"  Have >=1 correct rewrite : {samples_any_correct}  ({pct(samples_any_correct, total_samples)})",
        f"  Have >=1 fsf>0 rewrite   : {samples_any_fsf_pos}  ({pct(samples_any_fsf_pos, total_samples)})",
        f"  Have >=1 correct+fsf>0   : {samples_any_both}  ({pct(samples_any_both, total_samples)})",
        "",
        "[Rewrites]",
        f"  Total                    : {total_rewrites}",
        f"  is_correct=True          : {correct_count}  ({pct(correct_count, total_rewrites)})",
        f"  is_correct=False         : {total_rewrites - correct_count}  ({pct(total_rewrites - correct_count, total_rewrites)})",
        f"  fsf > 0                  : {fsf_pos_count}  ({pct(fsf_pos_count, total_rewrites)})",
        f"  fsf = 0                  : {total_rewrites - fsf_pos_count}  ({pct(total_rewrites - fsf_pos_count, total_rewrites)})",
        f"  correct AND fsf>0        : {both_count}  ({pct(both_count, total_rewrites)})",
        "",
        "[FSF Value Stats]",
        f"  mean   (all)             : {avg(fsf_values):.4f}",
        f"  median (all)             : {med(fsf_values):.4f}",
        f"  mean   (fsf>0 only)      : {avg(fsf_pos_vals):.4f}",
        f"  median (fsf>0 only)      : {med(fsf_pos_vals):.4f}",
        "",
        "[FSF Histogram (10 bins)]",
    ]

    if fsf_values:
        counts = [0] * 10
        for v in fsf_values:
            counts[min(int(v * 10), 9)] += 1
        max_cnt = max(counts) or 1
        for i in range(10):
            bar = "#" * int(counts[i] / max_cnt * 30)
            lines.append(f"  [{i/10:.1f}, {(i+1)/10:.1f})  {bar:<30}  {counts[i]}")

    lines += ["", "=" * 60, ""]
    report = "\n".join(lines)
    print(report)

    # Persist
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (
        os.path.join(stats_dir, f"tag_distribution_{ts}.txt"),
        os.path.join(stats_dir, "tag_distribution_latest.txt"),
    ):
        with open(path, "w") as f:
            f.write(report)
    print(f"[Stats saved] {stats_dir}/tag_distribution_{ts}.txt")
    print(f"[Stats saved] {stats_dir}/tag_distribution_latest.txt")


def _make_sharegpt_sample(question: str, cot: str, solution: str) -> Dict[str, Any]:
    """Return a single ShareGPT-format sample dict."""
    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n\n{solution}"},
        ]
    }


def build_dataset(
    records: List[Dict[str, Any]],
    output_file: str,
    filter_correct_only: bool,
    filter_fsf_positive: bool,
    filter_fsf_zero: bool,
    strategy: str,
    seed: int = 42,
    build_mode: str = "single",
    question_limit: Optional[int] = None,
    mix_rewrite_ratio: float = 0.5,
    target_count: Optional[int] = None,
) -> Dict[str, int]:
    """
    Build ShareGPT dataset from a list of records.

    Args:
        build_mode: "single" – one sample per question using rewrite_cot (original behavior).
                    "dual"   – two samples per question: original_cot and rewrite_cot.
                    "mix"    – target_count samples total. First fill floor(target_count *
                               mix_rewrite_ratio) slots from questions with valid rewrites
                               (after filtering); then fill the remainder with original_cot
                               from unused questions (no duplicate questions).
        question_limit: if set, randomly sample this many questions before building.
        mix_rewrite_ratio: (mix mode only) fraction of target_count that uses rewrite_cot.
        target_count: (mix mode only) desired total output samples. If None, uses all
                      questions with valid rewrites as rewrite samples (old behavior).

    Returns stats dict: total, selected, skipped, output_samples, mix_rewrite, mix_original.
    """
    random.seed(seed)

    if question_limit is not None and question_limit < len(records):
        records = random.sample(records, question_limit)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total = len(records)
    selected = 0
    skipped = 0
    output_samples = 0
    mix_rewrite = 0
    mix_original = 0

    if build_mode == "mix":
        # Pre-pass: split into valid-rewrite pool and no-rewrite pool
        valid_rewrite_pool: List[tuple] = []  # (record, best_rewrite)
        no_rewrite_pool: List[Dict] = []

        for record in records:
            rewrites = record.get("rewrite_cots", [])
            best = select_best_rewrite(
                rewrites,
                filter_correct_only=filter_correct_only,
                filter_fsf_positive=filter_fsf_positive,
                filter_fsf_zero=filter_fsf_zero,
                strategy=strategy,
            )
            if best is not None:
                valid_rewrite_pool.append((record, best))
            else:
                no_rewrite_pool.append(record)

        skipped = len(no_rewrite_pool)

        # Determine how many rewrite vs original samples
        n_total = target_count if target_count is not None else len(valid_rewrite_pool)
        n_rewrite = min(int(round(n_total * mix_rewrite_ratio)), len(valid_rewrite_pool))
        n_original = n_total - n_rewrite

        # Shuffle and partition valid_rewrite_pool
        random.shuffle(valid_rewrite_pool)
        rewrite_selected = valid_rewrite_pool[:n_rewrite]
        remaining_valid = valid_rewrite_pool[n_rewrite:]

        # Build original pool: remaining valid records + no-rewrite records (no duplicate questions)
        original_pool: List[Dict] = [r for r, _ in remaining_valid] + no_rewrite_pool
        random.shuffle(original_pool)
        original_selected = original_pool[:n_original]

        if len(original_selected) < n_original:
            print(
                f"WARNING: Not enough original samples to fill target. "
                f"Need {n_original}, have {len(original_selected)}. "
                f"Output will be {n_rewrite + len(original_selected)} samples."
            )

        selected = len(rewrite_selected)

        # Build all samples, then shuffle before writing
        all_samples = []
        for record, best in rewrite_selected:
            all_samples.append((_make_sharegpt_sample(record["question"], best["rewrite_cot"], record.get("solution", "")), "rewrite"))
        for record in original_selected:
            all_samples.append((_make_sharegpt_sample(record["question"], record.get("original_cot", ""), record.get("solution", "")), "original"))

        random.shuffle(all_samples)

        with open(output_file, "w") as fout:
            for sample, kind in all_samples:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if kind == "rewrite":
                    mix_rewrite += 1
                else:
                    mix_original += 1
                output_samples += 1

        return {
            "total": total,
            "selected": selected,
            "skipped": skipped,
            "output_samples": output_samples,
            "mix_rewrite": mix_rewrite,
            "mix_original": mix_original,
        }

    with open(output_file, "w") as fout:
        for record in records:
            question = record["question"]
            solution = record.get("solution", "")

            if build_mode == "dual":
                # original_cot is always written regardless of filters
                original_cot = record.get("original_cot", "")
                fout.write(json.dumps(
                    _make_sharegpt_sample(question, original_cot, solution),
                    ensure_ascii=False,
                ) + "\n")
                output_samples += 1

            rewrites = record.get("rewrite_cots", [])
            best = select_best_rewrite(
                rewrites,
                filter_correct_only=filter_correct_only,
                filter_fsf_positive=filter_fsf_positive,
                filter_fsf_zero=filter_fsf_zero,
                strategy=strategy,
            )
            if best is None:
                skipped += 1
                continue

            rewrite_cot = best["rewrite_cot"]

            if build_mode == "dual":
                fout.write(json.dumps(
                    _make_sharegpt_sample(question, rewrite_cot, solution),
                    ensure_ascii=False,
                ) + "\n")
                output_samples += 1
            else:
                fout.write(json.dumps(
                    _make_sharegpt_sample(question, rewrite_cot, solution),
                    ensure_ascii=False,
                ) + "\n")
                output_samples += 1

            selected += 1

    return {
        "total": total,
        "selected": selected,
        "skipped": skipped,
        "output_samples": output_samples,
        "mix_rewrite": mix_rewrite,
        "mix_original": mix_original,
    }


def register_to_llamafactory(
    dataset_info_path: str,
    dataset_name: str,
    output_file: str,
    use_absolute_path: bool = True,
) -> None:
    """Register (or overwrite) the dataset entry in LlamaFactory's dataset_info.json."""
    file_name = os.path.abspath(output_file) if use_absolute_path else output_file

    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[LlamaFactory] Registered '{dataset_name}' -> {file_name}")
    print(f"[LlamaFactory] Updated: {dataset_info_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ShareGPT dataset from CoT rewrite pipeline output (step 5)"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config file")
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Path to dataset build config YAML (overrides dataset_build section in --config)",
    )
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
        choices=["random", "max_fsf", "priority_correct", "min_ged"],
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


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


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

    # Load dataset build config: separate file takes precedence over inline dataset_build section
    if args.dataset_config:
        db_cfg = _load_yaml(args.dataset_config)
        print(f"[Config] Loaded dataset build config: {args.dataset_config}")
    else:
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
    filter_fsf_zero = filters_cfg.get("fsf_zero", False)
    strategy = args.strategy or db_cfg.get("selection_strategy", "priority_correct")
    build_mode = db_cfg.get("build_mode", "single")
    question_limit = db_cfg.get("question_limit", None)

    if build_mode not in ("single", "dual", "mix"):
        print(f"ERROR: Unknown build_mode '{build_mode}'. Use 'single', 'dual', or 'mix'.", file=sys.stderr)
        sys.exit(1)

    mix_rewrite_ratio = float(db_cfg.get("mix_rewrite_ratio", 0.5))
    target_count = db_cfg.get("target_count", None)
    if target_count is not None:
        target_count = int(target_count)

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

    # Print & persist tag distribution before building
    records = _read_jsonl(input_file)
    print_tag_distribution(records, output_dir)

    if args.output_file:
        output_file = args.output_file
    else:
        final_dir = os.path.join(output_dir, "final")
        output_filename = db_cfg.get("output_filename", "train_sharegpt.jsonl")
        output_file = os.path.join(final_dir, output_filename)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Building ShareGPT dataset:")
    print(f"  Input:          {input_file}")
    print(f"  Output:         {output_file}")
    print(f"  Build mode:     {build_mode}")
    print(f"  Question limit: {question_limit if question_limit is not None else 'all'}")
    print(f"  Filters:        correct_only={filter_correct_only}, fsf_positive={filter_fsf_positive}, fsf_zero={filter_fsf_zero}")
    print(f"  Strategy:       {strategy}")
    if build_mode == "mix":
        print(f"  Mix ratio:      rewrite={mix_rewrite_ratio:.2f}, original={1-mix_rewrite_ratio:.2f}")
        print(f"  Target count:   {target_count if target_count is not None else 'all valid rewrites'}")

    stats = build_dataset(
        records=records,
        output_file=output_file,
        filter_correct_only=filter_correct_only,
        filter_fsf_positive=filter_fsf_positive,
        filter_fsf_zero=filter_fsf_zero,
        strategy=strategy,
        seed=config.get("seed", 42),
        build_mode=build_mode,
        question_limit=question_limit,
        mix_rewrite_ratio=mix_rewrite_ratio,
        target_count=target_count,
    )

    print(f"\nDone:")
    print(f"  Total questions:  {stats['total']}")
    print(f"  Selected:         {stats['selected']}")
    print(f"  Skipped:          {stats['skipped']}")
    print(f"  Output samples:   {stats['output_samples']}")
    if build_mode == "mix":
        print(f"  Mix rewrite:      {stats['mix_rewrite']}")
        print(f"  Mix original:     {stats['mix_original']}")
    print(f"  Output file:      {output_file}")

    # LlamaFactory registration
    lf_cfg = db_cfg.get("llamafactory", {})
    if lf_cfg.get("enabled", False):
        dataset_info_path = lf_cfg.get("dataset_info_path")
        dataset_name = lf_cfg.get("dataset_name")
        if not dataset_info_path or not dataset_name:
            print("WARNING: llamafactory.dataset_info_path and dataset_name must both be set.", file=sys.stderr)
        elif not os.path.exists(dataset_info_path):
            print(f"WARNING: LlamaFactory dataset_info.json not found: {dataset_info_path}", file=sys.stderr)
        else:
            register_to_llamafactory(
                dataset_info_path=dataset_info_path,
                dataset_name=dataset_name,
                output_file=output_file,
                use_absolute_path=lf_cfg.get("use_absolute_path", True),
            )


if __name__ == "__main__":
    main()
