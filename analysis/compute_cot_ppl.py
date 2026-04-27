#!/usr/bin/env python3
"""
Step 1: compute assistant-only PPL for original/rewrite CoTs with pure forward.

Prompt format:
    user: {question}, assistant: <think>\n\n</think>{cot}

Scored text:
    " <think>\n\n</think>{cot}"

Implementation notes:
- teacher-forcing only, no generation
- labels before assistant start are masked to -100
- multi-GPU data parallel via one worker process per GPU
- dynamic task queue and resume from worker result shards
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_INPUT = (
    "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/"
    "qwen2_5_7b_grpo_8x40g/intermediate/rollout_output.jsonl"
)
DEFAULT_MODEL = "/data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = (
    "/home/luoys/cdad/analysis/results/"
    "cot_ppl_qwen2_5_7b_instruct_rollout_output.jsonl"
)


@dataclass
class PromptSpec:
    sample_idx: int
    variant_type: str
    rewrite_idx: int
    prompt: str
    prompt_token_len: int
    assistant_start_idx: int
    question_char_len: int
    cot_char_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute assistant-only PPL for original/rewrite CoTs with teacher-forcing forward."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=4096,
        help="Approx upper bound on padded tokens per forward batch: batch_size * max_seq_len.",
    )
    parser.add_argument(
        "--max-samples-per-batch",
        type=int,
        default=4,
        help="Hard cap on samples per forward batch.",
    )
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_queue-dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_work-dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_total-items", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_data(path: str, limit_samples: Optional[int] = None) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit_samples is not None and idx >= limit_samples:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_available_gpus() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]


def recover_queue(queue_dir: Path) -> None:
    for path in sorted(queue_dir.glob("chunk_*.claiming_*")):
        base = path.name.rsplit(".claiming_", 1)[0]
        target = queue_dir / f"{base}.json"
        if target.exists():
            path.unlink()
        else:
            path.rename(target)


def claim_next_chunk(queue_dir: Path, gpu_id: int) -> tuple[Optional[list[int]], Optional[Path]]:
    for chunk_file in sorted(queue_dir.glob("chunk_*.json")):
        claimed = chunk_file.with_name(f"{chunk_file.stem}.claiming_{gpu_id}")
        try:
            chunk_file.rename(claimed)
            with open(claimed, "r", encoding="utf-8") as f:
                items = json.load(f)
            return items, claimed
        except (FileNotFoundError, OSError):
            continue
    return None, None


def get_worker_result_files(work_dir: Path) -> list[Path]:
    return sorted(work_dir.glob("worker_*.results.jsonl"))


def collect_done_samples(work_dir: Path) -> set[int]:
    counts: collections.Counter[int] = collections.Counter()
    for result_file in get_worker_result_files(work_dir):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                counts[int(row["sample_idx"])] += 1
    return {sample_idx for sample_idx, count in counts.items() if count >= 6}


def build_prompt(question: str, cot: str) -> tuple[str, str]:
    prefix = f"user: {question}, assistant:"
    assistant_text = f" <think>\n\n</think>{cot}"
    return prefix, prefix + assistant_text


def build_prompt_specs(tokenizer, sample_idx: int, item: dict[str, Any], max_model_len: int) -> list[PromptSpec]:
    question = item["question"]
    out: list[PromptSpec] = []

    def add_one(variant_type: str, rewrite_idx: int, cot: str) -> None:
        prefix, full_prompt = build_prompt(question, cot)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(full_ids)
        if prompt_len > max_model_len:
            raise ValueError(
                f"sample_idx={sample_idx} variant={variant_type} rewrite_idx={rewrite_idx} "
                f"prompt_len={prompt_len} exceeds max_model_len={max_model_len}"
            )
        out.append(
            PromptSpec(
                sample_idx=sample_idx,
                variant_type=variant_type,
                rewrite_idx=rewrite_idx,
                prompt=full_prompt,
                prompt_token_len=prompt_len,
                assistant_start_idx=len(prefix_ids),
                question_char_len=len(question),
                cot_char_len=len(cot),
            )
        )

    add_one("original", -1, item["original_cot"])
    for rewrite_idx, rewrite_item in enumerate(item["rewrite_cots"]):
        add_one("rewrite", rewrite_idx, rewrite_item["rewrite_cot"])
    return out


def make_micro_batches(
    specs: list[PromptSpec],
    max_batch_tokens: int,
    max_samples_per_batch: int,
) -> list[list[PromptSpec]]:
    ordered = sorted(specs, key=lambda x: x.prompt_token_len, reverse=True)
    batches: list[list[PromptSpec]] = []
    current: list[PromptSpec] = []
    current_max_len = 0

    for spec in ordered:
        next_max_len = max(current_max_len, spec.prompt_token_len)
        next_size = len(current) + 1
        would_exceed = next_size * next_max_len > max_batch_tokens
        if current and (would_exceed or len(current) >= max_samples_per_batch):
            batches.append(current)
            current = [spec]
            current_max_len = spec.prompt_token_len
        else:
            current.append(spec)
            current_max_len = next_max_len

    if current:
        batches.append(current)
    return batches


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ppl_values = np.array([row["ppl"] for row in rows if math.isfinite(row["ppl"])], dtype=float)
    original_values = np.array(
        [row["ppl"] for row in rows if row["variant_type"] == "original" and math.isfinite(row["ppl"])],
        dtype=float,
    )
    rewrite_values = np.array(
        [row["ppl"] for row in rows if row["variant_type"] == "rewrite" and math.isfinite(row["ppl"])],
        dtype=float,
    )

    def stats(arr: np.ndarray) -> dict[str, Any]:
        if arr.size == 0:
            return {}
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "all": stats(ppl_values),
        "original": stats(original_values),
        "rewrite": stats(rewrite_values),
    }


def merge_result_shards(
    work_dir: Path,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    merged: dict[tuple[int, str, int], dict[str, Any]] = {}
    for result_file in get_worker_result_files(work_dir):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (
                    int(row["sample_idx"]),
                    str(row["variant_type"]),
                    int(row.get("rewrite_idx", -1)),
                )
                merged[key] = row

    rows = [merged[key] for key in sorted(merged.keys())]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata_path = output_path.with_suffix(".meta.json")
    metadata = dict(metadata)
    metadata["result_row_count"] = len(rows)
    metadata["summary"] = summarize_rows(rows)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def score_batch(model, tokenizer, batch_specs: list[PromptSpec], model_name: str, model_path: str) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    prompts = [spec.prompt for spec in batch_specs]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device, non_blocking=True)
    attention_mask = encoded["attention_mask"].to(model.device, non_blocking=True)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    for row_idx, spec in enumerate(batch_specs):
        labels[row_idx, : spec.assistant_start_idx] = -100

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_losses = flat_losses.view(shift_labels.shape)
        valid_mask = shift_labels.ne(-100)
        token_counts = valid_mask.sum(dim=1)
        loss_sums = (token_losses * valid_mask).sum(dim=1)
        mean_nlls = loss_sums / token_counts.clamp_min(1)
        ppls = torch.exp(mean_nlls.clamp(max=100))

    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(batch_specs):
        token_count = int(token_counts[idx].item())
        mean_nll = float(mean_nlls[idx].item()) if token_count > 0 else float("inf")
        ppl = float(ppls[idx].item()) if token_count > 0 else float("inf")
        rows.append(
            {
                "sample_idx": spec.sample_idx,
                "variant_type": spec.variant_type,
                "rewrite_idx": spec.rewrite_idx,
                "model_name": model_name,
                "model_path": model_path,
                "ppl": ppl,
                "mean_nll": mean_nll,
                "assistant_token_count": token_count,
                "assistant_start_idx": spec.assistant_start_idx,
                "total_prompt_token_count": spec.prompt_token_len,
                "missing_positions": 0,
                "question_char_len": spec.question_char_len,
                "cot_char_len": spec.cot_char_len,
            }
        )

    del outputs, logits, shift_logits, shift_labels, flat_losses, token_losses, valid_mask
    del input_ids, attention_mask, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def worker_main(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = args._gpu_id
    queue_dir = Path(args._queue_dir)
    work_dir = Path(args._work_dir)
    result_file = Path(args._result_file)
    model_name = args.model_name or Path(args.model).name

    done_set = collect_done_samples(work_dir)
    print(f"[GPU {gpu_id}] resume: {len(done_set)} samples already complete")

    data = load_data(args.input, args.limit_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    result_file.parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(
        total=args._total_items,
        initial=len(done_set),
        desc=f"[GPU {gpu_id}]",
        unit="sample",
        position=0,
        leave=True,
    )

    processed = 0
    while True:
        chunk_items, claimed_path = claim_next_chunk(queue_dir, gpu_id)
        if chunk_items is None:
            break

        remaining = [sample_idx for sample_idx in chunk_items if sample_idx not in done_set]
        if not remaining:
            claimed_path.rename(claimed_path.with_name(f"{claimed_path.stem.rsplit(f'.claiming_{gpu_id}', 1)[0]}.done"))
            continue

        prompt_specs: list[PromptSpec] = []
        for sample_idx in remaining:
            prompt_specs.extend(build_prompt_specs(tokenizer, sample_idx, data[sample_idx], args.max_model_len))

        micro_batches = make_micro_batches(
            prompt_specs,
            max_batch_tokens=args.max_batch_tokens,
            max_samples_per_batch=args.max_samples_per_batch,
        )

        with open(result_file, "a", encoding="utf-8") as f:
            for batch_specs in micro_batches:
                rows = score_batch(model, tokenizer, batch_specs, model_name, args.model)
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for sample_idx in remaining:
            done_set.add(sample_idx)
        processed += len(remaining)
        pbar.update(len(remaining))

        done_name = f"{claimed_path.stem.rsplit(f'.claiming_{gpu_id}', 1)[0]}.done"
        claimed_path.rename(claimed_path.with_name(done_name))

    pbar.close()
    print(f"[GPU {gpu_id}] finished, processed {processed} samples")


def launcher_main(args: argparse.Namespace) -> None:
    model_name = args.model_name or Path(args.model).name
    output_path = Path(args.output)
    work_dir = output_path.parent / f".cot_ppl_{output_path.stem}_{model_name}"
    queue_dir = work_dir / "queue"

    if args.force and work_dir.exists():
        shutil.rmtree(work_dir)
    if args.force and output_path.exists():
        output_path.unlink()
    if args.force and output_path.with_suffix(".meta.json").exists():
        output_path.with_suffix(".meta.json").unlink()

    work_dir.mkdir(parents=True, exist_ok=True)
    queue_dir.mkdir(parents=True, exist_ok=True)
    recover_queue(queue_dir)

    data = load_data(args.input, args.limit_samples)
    print(f"loaded {len(data)} samples from {args.input}")

    all_gpus = get_available_gpus()
    num_gpus = args.num_gpus or len(all_gpus)
    gpu_ids = all_gpus[:num_gpus]
    print(f"using GPUs: {gpu_ids}")

    done_set = collect_done_samples(work_dir)
    need_compute = [sample_idx for sample_idx in range(len(data)) if sample_idx not in done_set]
    print(f"already complete: {len(done_set)} samples")
    print(f"remaining: {len(need_compute)} samples")

    pending_chunks = list(queue_dir.glob("chunk_*.json"))
    if not pending_chunks:
        for old_done in queue_dir.glob("chunk_*.done"):
            old_done.unlink()
        for chunk_id, start in enumerate(range(0, len(need_compute), args.chunk_size)):
            chunk = need_compute[start : start + args.chunk_size]
            with open(queue_dir / f"chunk_{chunk_id:06d}.json", "w", encoding="utf-8") as f:
                json.dump(chunk, f)
        chunk_count = math.ceil(len(need_compute) / max(args.chunk_size, 1)) if need_compute else 0
        print(f"created {chunk_count} queue chunks")
    else:
        print(f"resume with {len(pending_chunks)} pending chunks")

    result_files = [work_dir / f"worker_{i}.results.jsonl" for i in range(num_gpus)]
    procs: list[subprocess.Popen[Any]] = []
    for worker_idx in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[worker_idx])
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--input",
            args.input,
            "--model",
            args.model,
            "--output",
            args.output,
            "--chunk-size",
            str(args.chunk_size),
            "--max-model-len",
            str(args.max_model_len),
            "--max-batch-tokens",
            str(args.max_batch_tokens),
            "--max-samples-per-batch",
            str(args.max_samples_per_batch),
            "--_worker",
            "--_gpu-id",
            str(gpu_ids[worker_idx]),
            "--_queue-dir",
            str(queue_dir),
            "--_work-dir",
            str(work_dir),
            "--_result-file",
            str(result_files[worker_idx]),
            "--_total-items",
            str(len(data)),
        ]
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        if args.limit_samples is not None:
            cmd.extend(["--limit-samples", str(args.limit_samples)])
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        print(f"worker {worker_idx}: pid={proc.pid}, gpu={gpu_ids[worker_idx]}")

    failed: list[tuple[int, int, int]] = []
    for worker_idx, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed.append((worker_idx, gpu_ids[worker_idx], ret))

    if failed:
        print(f"warning: {len(failed)} workers failed: {failed}")
    else:
        print("all workers finished successfully")

    metadata = {
        "input_path": args.input,
        "model_name": model_name,
        "model_path": args.model,
        "output_path": str(output_path),
        "prompt_format": "user: {question}, assistant: <think>\\n\\n</think>{cot}",
        "assistant_scored_text": " <think>\\n\\n</think>{cot}",
        "total_samples": len(data),
        "expected_rows": len(data) * 6,
        "num_gpus": len(gpu_ids),
        "chunk_size": args.chunk_size,
        "max_model_len": args.max_model_len,
        "max_batch_tokens": args.max_batch_tokens,
        "max_samples_per_batch": args.max_samples_per_batch,
        "engine": "transformers_forward",
    }
    merge_result_shards(work_dir, output_path, metadata)
    print(f"merged results -> {output_path}")
    print(f"metadata -> {output_path.with_suffix('.meta.json')}")

    if not failed:
        shutil.rmtree(queue_dir, ignore_errors=True)
        print(f"temporary queue cleaned: {queue_dir}")
    else:
        print(f"temporary queue kept for resume: {queue_dir}")


def main() -> None:
    args = parse_args()
    if args._worker:
        worker_main(args)
    else:
        launcher_main(args)


if __name__ == "__main__":
    main()
