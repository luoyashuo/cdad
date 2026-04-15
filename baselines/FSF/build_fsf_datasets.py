#!/usr/bin/env python3
"""
Build FSF-tagged datasets from DeepMath SFT data.

This tool keeps FSF recognition exactly aligned with
`verl.experimental.reward.cot_rewriter.sglang_fsf_reward.SGLangFSFClient`.

Pipeline:
1. Read `/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl`
2. Compute FSF for every `reasoning[].thought`
3. Keep only questions that contain at least one `fsf > 0` thought
4. Build two 10k datasets:
   - pure `fsf == 0`
   - pure `fsf > 0`

Outputs:
- `scored_records.jsonl`: original question-level schema with FSF annotations
- `fsf_zero_10k/raw_samples.jsonl`
- `fsf_zero_10k/sharegpt.jsonl`
- `fsf_positive_10k/raw_samples.jsonl`
- `fsf_positive_10k/sharegpt.jsonl`
- `summary.json`
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ray
from tqdm import tqdm


LOGGER = logging.getLogger("build_fsf_datasets")

DEFAULT_INPUT = "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl"
DEFAULT_MODEL = "/data/hf_home/models/Yaseal-qwen3_4b_fsf_grm"
DEFAULT_OUTPUT = "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits"
DEFAULT_RAY_TEMP = "/home/luoys/.ray_tmp_fsf"
REASONING_PER_QUESTION = 3


def _setup_repo_imports() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    repo_root = base_dir / "cot_rewriter_grpo"
    verl_root = repo_root / "verl"
    for path in (str(verl_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root


REPO_ROOT = _setup_repo_imports()


@dataclass
class Sample:
    question_idx: int
    reasoning_idx: int
    question: str
    thought: str
    solution: str
    fsf: float
    failed_nodes: Optional[int]
    total_nodes: Optional[int]
    question_id: str

    def to_raw_dict(self) -> Dict[str, Any]:
        return {
            "question_idx": self.question_idx,
            "reasoning_idx": self.reasoning_idx,
            "question_id": self.question_id,
            "question": self.question,
            "thought": self.thought,
            "solution": self.solution,
            "fsf": self.fsf,
            "failed_nodes": self.failed_nodes,
            "total_nodes": self.total_nodes,
        }

    def to_sharegpt_dict(self) -> Dict[str, Any]:
        return {
            "id": f"{self.question_id}_{self.reasoning_idx}",
            "conversations": [
                {"role": "user", "content": self.question},
                {
                    "role": "assistant",
                    "content": f"<think>\n{self.thought}\n</think>\n\n{self.solution}",
                },
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FSF for DeepMath thoughts and build 10k splits.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input DeepMath JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument(
        "--phase",
        choices=["all", "score", "build"],
        default="all",
        help="Run FSF scoring, dataset building, or both.",
    )
    parser.add_argument(
        "--scored-input",
        default=None,
        help="Use an existing scored_records.jsonl for build phase. Defaults to {output_dir}/scored_records.jsonl.",
    )
    parser.add_argument("--target-size", type=int, default=10_000, help="Samples per output dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume scoring from existing scored_records.jsonl.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="FSF judge model path.")
    parser.add_argument("--batch-size", type=int, default=256, help="Total thought batch size per FSF pass.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="TP size per SGLang worker.")
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. Default: use all visible GPUs.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.9, help="SGLang memory fraction.")
    parser.add_argument("--dtype", default="float16", help="Model dtype.")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length.")
    parser.add_argument("--temperature", type=float, default=0.3, help="FSF judge temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=8000, help="FSF judge max new tokens.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-worker micro batch size. Defaults to batch_size / num_workers.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        default=DEFAULT_RAY_TEMP,
        help="Ray temp directory. Use a short path to avoid AF_UNIX socket length limits.",
    )
    parser.add_argument(
        "--selection",
        choices=["random", "max_fsf", "min_fsf"],
        default="random",
        help="How to pick one matching thought per question.",
    )
    parser.add_argument(
        "--pair-on-question",
        action="store_true",
        help="Use the same questions for zero and positive datasets; requires questions that have both zero and positive thoughts.",
    )
    parser.add_argument(
        "--no-question-binding",
        action="store_true",
        help="V2: allow zero and positive datasets to be sampled from completely disjoint question pools. "
             "Zero candidates come from ALL questions with fsf=0 thoughts, not only questions that also have fsf>0.",
    )
    parser.add_argument(
        "--save-dot-outputs",
        action="store_true",
        help="Persist raw FSF model DOT outputs into scored_records.jsonl. Disabled by default to save disk.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of input questions. Useful for smoke tests.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_gpu_ids(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpu_ids:
        return None
    return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path, skip: int = 0, limit: Optional[int] = None) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < skip:
                continue
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def stable_question_id(question: str) -> str:
    return md5(question.encode("utf-8")).hexdigest()


def choose_reasoning(
    reasoning_list: Sequence[Tuple[int, Dict[str, Any]]],
    selection: str,
    rng: random.Random,
) -> Tuple[int, Dict[str, Any]]:
    candidates = list(reasoning_list)
    if not candidates:
        raise ValueError("No candidates available for choose_reasoning().")
    if selection == "random":
        return rng.choice(candidates)
    if selection == "max_fsf":
        return max(candidates, key=lambda item: float(item[1].get("fsf") or 0.0))
    if selection == "min_fsf":
        return min(candidates, key=lambda item: float(item[1].get("fsf") or 0.0))
    raise ValueError(f"Unknown selection mode: {selection}")


def build_client(args: argparse.Namespace):
    from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient  # noqa: E402
    devices = parse_gpu_ids(args.gpu_ids)
    if devices is not None:
        num_gpus = len(devices)
    else:
        import torch

        num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA device is available for FSF scoring.")
    if num_gpus % args.tensor_parallel_size != 0:
        raise ValueError(
            f"num_gpus ({num_gpus}) must be divisible by tensor_parallel_size ({args.tensor_parallel_size})."
        )
    num_workers = num_gpus // args.tensor_parallel_size
    micro_batch_size = args.micro_batch_size or max(1, args.batch_size // max(1, num_workers))
    LOGGER.info(
        "Initializing FSF client: model=%s, num_gpus=%s, tp=%s, num_workers=%s, batch_size=%s, micro_batch_size=%s",
        args.model_path,
        num_gpus,
        args.tensor_parallel_size,
        num_workers,
        args.batch_size,
        micro_batch_size,
    )
    return SGLangFSFClient(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        mem_fraction_static=args.mem_fraction_static,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        devices=devices,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        micro_batch_size=micro_batch_size,
    )


def annotate_batch(
    records: List[Dict[str, Any]],
    client: SGLangFSFClient,
    save_dot_outputs: bool,
) -> List[Dict[str, Any]]:
    thoughts: List[str] = []
    mapping: List[Tuple[int, int]] = []
    for record_idx, record in enumerate(records):
        reasoning_list = record.get("reasoning") or []
        for reasoning_idx, reasoning in enumerate(reasoning_list):
            thoughts.append((reasoning.get("thought") or "").strip())
            mapping.append((record_idx, reasoning_idx))

    result = client.compute_rewrite_fsf_only(thoughts, return_dot_outputs=save_dot_outputs)
    fsfs = result.get("rewrite_fsfs", [])
    failed_nodes = result.get("rewrite_failed_nodes", [])
    total_nodes = result.get("rewrite_total_nodes", [])
    dot_outputs = result.get("rewrite_dot_outputs", []) if save_dot_outputs else []

    for flat_idx, (record_idx, reasoning_idx) in enumerate(mapping):
        reasoning = records[record_idx]["reasoning"][reasoning_idx]
        fsf_value = fsfs[flat_idx]
        reasoning["fsf"] = fsf_value
        reasoning["fsf_failed_nodes"] = failed_nodes[flat_idx]
        reasoning["fsf_total_nodes"] = total_nodes[flat_idx]
        if save_dot_outputs:
            dot_output = dot_outputs[flat_idx]
            reasoning["fsf_model_output"] = dot_output[:32000] if dot_output else dot_output

    for record in records:
        reasoning_list = record.get("reasoning") or []
        fsf_values = [item.get("fsf") for item in reasoning_list]
        pos_count = sum(1 for value in fsf_values if value is not None and value > 0)
        zero_count = sum(1 for value in fsf_values if value == 0)
        missing_count = sum(1 for value in fsf_values if value is None)
        record["question_has_fsf_positive"] = pos_count > 0
        record["question_fsf_positive_count"] = pos_count
        record["question_fsf_zero_count"] = zero_count
        record["question_fsf_missing_count"] = missing_count
    return records


def process_scoring_batch(
    batch: List[Tuple[int, Dict[str, Any]]],
    client: SGLangFSFClient,
    output_handle: Any,
    save_dot_outputs: bool,
) -> int:
    if not batch:
        return 0
    record_indices = [item[0] for item in batch]
    records = [item[1] for item in batch]
    annotated = annotate_batch(records, client=client, save_dot_outputs=save_dot_outputs)
    for source_idx, record in zip(record_indices, annotated):
        record["_source_question_idx"] = source_idx
        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_handle.flush()
    return len(annotated)


def score_records(args: argparse.Namespace, output_dir: Path) -> Path:
    input_path = Path(args.input)
    scored_path = output_dir / "scored_records.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    already_done = 0
    write_mode = "w"
    if args.resume and scored_path.exists():
        already_done = count_lines(scored_path)
        write_mode = "a"
        LOGGER.info("Resuming scoring from %s records: %s", already_done, scored_path)

    input_limit = args.max_records
    actual_total_records = count_lines(input_path)
    total_records = min(input_limit, actual_total_records) if input_limit is not None else actual_total_records
    if total_records < already_done:
        raise ValueError(
            f"Existing scored file has {already_done} lines, but current input limit only allows {total_records}."
        )
    remaining_records = total_records - already_done
    if remaining_records <= 0:
        LOGGER.info("Scoring already complete: %s", scored_path)
        return scored_path

    if not ray.is_initialized():
        ray_temp_dir = Path(args.ray_temp_dir)
        ray_temp_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Initializing Ray with temp dir: %s", ray_temp_dir)
        ray.init(ignore_reinit_error=True, _temp_dir=str(ray_temp_dir))

    client = build_client(args)
    completed = already_done
    batch: List[Tuple[int, Dict[str, Any]]] = []
    thoughts_in_batch = 0

    try:
        with scored_path.open(write_mode, encoding="utf-8") as output_handle:
            with tqdm(total=remaining_records, desc="FSF scoring", unit="question") as pbar:
                for source_idx, record in iter_jsonl(input_path, skip=already_done, limit=input_limit):
                    batch.append((source_idx, record))
                    thoughts_in_batch += len(record.get("reasoning") or [])
                    if thoughts_in_batch >= args.batch_size:
                        completed += process_scoring_batch(
                            batch=batch,
                            client=client,
                            output_handle=output_handle,
                            save_dot_outputs=args.save_dot_outputs,
                        )
                        pbar.update(len(batch))
                        batch = []
                        thoughts_in_batch = 0
                if batch:
                    completed += process_scoring_batch(
                        batch=batch,
                        client=client,
                        output_handle=output_handle,
                        save_dot_outputs=args.save_dot_outputs,
                    )
                    pbar.update(len(batch))
    finally:
        client.shutdown()

    LOGGER.info("Scoring complete: %s records -> %s", completed, scored_path)
    return scored_path


def make_sample(
    question_idx: int,
    reasoning_idx: int,
    question: str,
    reasoning: Dict[str, Any],
) -> Sample:
    return Sample(
        question_idx=question_idx,
        reasoning_idx=reasoning_idx,
        question=question,
        thought=(reasoning.get("thought") or "").strip(),
        solution=(reasoning.get("solution") or "").strip(),
        fsf=float(reasoning.get("fsf")),
        failed_nodes=reasoning.get("fsf_failed_nodes"),
        total_nodes=reasoning.get("fsf_total_nodes"),
        question_id=stable_question_id(question),
    )


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_datasets(args: argparse.Namespace, scored_path: Path, output_dir: Path) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    scored_records = load_jsonl(scored_path, limit=args.max_records)
    LOGGER.info("Loaded %s scored records from %s", len(scored_records), scored_path)

    total_thoughts = sum(len(record.get("reasoning") or []) for record in scored_records)
    if total_thoughts != len(scored_records) * REASONING_PER_QUESTION:
        LOGGER.warning(
            "Dataset no longer looks like fixed 3-thought DeepMath: total_records=%s total_thoughts=%s",
            len(scored_records),
            total_thoughts,
        )

    positive_question_entries: List[Tuple[int, str, List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]] = []
    parse_missing_questions = 0
    for question_idx, record in enumerate(scored_records):
        source_question_idx = record.get("_source_question_idx", question_idx)
        question = record["question"]
        positives: List[Tuple[int, Dict[str, Any]]] = []
        zeros: List[Tuple[int, Dict[str, Any]]] = []
        for reasoning_idx, reasoning in enumerate(record.get("reasoning") or []):
            fsf_value = reasoning.get("fsf")
            if fsf_value is None:
                continue
            if fsf_value > 0:
                positives.append((reasoning_idx, reasoning))
            elif fsf_value == 0:
                zeros.append((reasoning_idx, reasoning))
        if not positives and not zeros:
            parse_missing_questions += 1
            continue
        if positives:
            positive_question_entries.append((source_question_idx, question, positives, zeros))

    if not positive_question_entries:
        raise RuntimeError("No question contains an FSF-positive thought. Cannot build datasets.")

    all_question_entries: List[Tuple[int, str, List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]] = []
    for question_idx, record in enumerate(scored_records):
        source_question_idx = record.get("_source_question_idx", question_idx)
        question = record["question"]
        positives: List[Tuple[int, Dict[str, Any]]] = []
        zeros: List[Tuple[int, Dict[str, Any]]] = []
        for reasoning_idx, reasoning in enumerate(record.get("reasoning") or []):
            fsf_value = reasoning.get("fsf")
            if fsf_value is None:
                continue
            if fsf_value > 0:
                positives.append((reasoning_idx, reasoning))
            elif fsf_value == 0:
                zeros.append((reasoning_idx, reasoning))
        if positives or zeros:
            all_question_entries.append((source_question_idx, question, positives, zeros))

    if args.pair_on_question:
        base_entries = [entry for entry in all_question_entries if entry[2] and entry[3]]
        if len(base_entries) < args.target_size:
            raise RuntimeError(
                f"Only {len(base_entries)} questions have both fsf>0 and fsf=0 thoughts, "
                f"smaller than target_size={args.target_size}."
            )
        rng.shuffle(base_entries)
        chosen_entries = base_entries[: args.target_size]
        zero_samples = []
        positive_samples = []
        for question_idx, question, positives, zeros in chosen_entries:
            zero_reasoning_idx, zero_reasoning = choose_reasoning(zeros, args.selection, rng)
            positive_reasoning_idx, positive_reasoning = choose_reasoning(positives, args.selection, rng)
            zero_samples.append(make_sample(question_idx, zero_reasoning_idx, question, zero_reasoning))
            positive_samples.append(make_sample(question_idx, positive_reasoning_idx, question, positive_reasoning))
    else:
        if args.no_question_binding:
            zero_candidate_entries = [entry for entry in all_question_entries if entry[3]]
            positive_candidate_entries = [entry for entry in all_question_entries if entry[2]]
        else:
            positive_question_entries = [entry for entry in all_question_entries if entry[2]]
            zero_candidate_entries = [entry for entry in positive_question_entries if entry[3]]
            positive_candidate_entries = list(positive_question_entries)

        if len(zero_candidate_entries) < args.target_size:
            raise RuntimeError(
                f"Only {len(zero_candidate_entries)} eligible questions have at least one fsf=0 thought, "
                f"smaller than target_size={args.target_size}."
            )
        if len(positive_candidate_entries) < args.target_size:
            raise RuntimeError(
                f"Only {len(positive_candidate_entries)} eligible questions have at least one fsf>0 thought, "
                f"smaller than target_size={args.target_size}."
            )

        rng.shuffle(zero_candidate_entries)
        rng.shuffle(positive_candidate_entries)

        zero_samples = []
        for question_idx, question, _positives, zeros in zero_candidate_entries[: args.target_size]:
            reasoning_idx, reasoning = choose_reasoning(zeros, args.selection, rng)
            zero_samples.append(make_sample(question_idx, reasoning_idx, question, reasoning))

        positive_samples = []
        for question_idx, question, positives, _zeros in positive_candidate_entries[: args.target_size]:
            reasoning_idx, reasoning = choose_reasoning(positives, args.selection, rng)
            positive_samples.append(make_sample(question_idx, reasoning_idx, question, reasoning))

    zero_dir = output_dir / f"fsf_zero_{args.target_size}"
    positive_dir = output_dir / f"fsf_positive_{args.target_size}"
    zero_dir.mkdir(parents=True, exist_ok=True)
    positive_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(zero_dir / "raw_samples.jsonl", (sample.to_raw_dict() for sample in zero_samples))
    write_jsonl(zero_dir / "sharegpt.jsonl", (sample.to_sharegpt_dict() for sample in zero_samples))
    write_jsonl(positive_dir / "raw_samples.jsonl", (sample.to_raw_dict() for sample in positive_samples))
    write_jsonl(positive_dir / "sharegpt.jsonl", (sample.to_sharegpt_dict() for sample in positive_samples))

    positive_thought_count = sum(
        1 for record in scored_records for reasoning in (record.get("reasoning") or []) if (reasoning.get("fsf") or 0) > 0
    )
    zero_thought_count = sum(
        1 for record in scored_records for reasoning in (record.get("reasoning") or []) if reasoning.get("fsf") == 0
    )
    missing_thought_count = sum(
        1 for record in scored_records for reasoning in (record.get("reasoning") or []) if reasoning.get("fsf") is None
    )

    summary = {
        "input": str(Path(args.input)),
        "scored_input": str(scored_path),
        "output_dir": str(output_dir),
        "repo_root": str(REPO_ROOT),
        "model_path": args.model_path,
        "seed": args.seed,
        "target_size": args.target_size,
        "selection": args.selection,
        "pair_on_question": args.pair_on_question,
        "no_question_binding": args.no_question_binding,
        "save_dot_outputs": args.save_dot_outputs,
        "max_records": args.max_records,
        "stats": {
            "total_questions": len(scored_records),
            "total_thoughts": total_thoughts,
            "questions_with_positive_thought": len(positive_question_entries),
            "questions_with_positive_and_zero_thought": sum(1 for _idx, _q, pos, zero in positive_question_entries if pos and zero),
            "questions_parse_all_failed": parse_missing_questions,
            "positive_thoughts": positive_thought_count,
            "zero_thoughts": zero_thought_count,
            "missing_thoughts": missing_thought_count,
            "zero_dataset_size": len(zero_samples),
            "positive_dataset_size": len(positive_samples),
        },
        "outputs": {
            "scored_records": str(scored_path),
            "zero_raw": str(zero_dir / "raw_samples.jsonl"),
            "zero_sharegpt": str(zero_dir / "sharegpt.jsonl"),
            "positive_raw": str(positive_dir / "raw_samples.jsonl"),
            "positive_sharegpt": str(positive_dir / "sharegpt.jsonl"),
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Dataset build complete. Summary written to %s", summary_path)
    return summary


def main() -> None:
    configure_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)
    scored_path = Path(args.scored_input) if args.scored_input else output_dir / "scored_records.jsonl"

    if args.phase in {"all", "score"}:
        scored_path = score_records(args, output_dir=output_dir)

    if args.phase in {"all", "build"}:
        if not scored_path.exists():
            raise FileNotFoundError(f"Scored input does not exist: {scored_path}")
        summary = build_datasets(args, scored_path=scored_path, output_dir=output_dir)
        LOGGER.info("Summary stats: %s", json.dumps(summary["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
