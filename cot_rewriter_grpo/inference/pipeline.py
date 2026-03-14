"""
CoT Rewrite Inference Pipeline - Steps 1-4

Steps:
  1. Model loading / FSDP→HF format conversion
  2. Rollout: batch CoT rewriting with vLLM (multi-GPU data parallel via Ray)
  3. Correct Tag: judge whether each rewrite leads to the correct solution
  4. FSF Tag: compute Failure Step Fraction for each rewrite

Usage:
    python -m inference.pipeline --config inference/config_example.yaml
    python -m inference.pipeline --config inference/config_example.yaml --step correct
    python -m inference.pipeline --config inference/config_example.yaml --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt builder (copied from reasoning_graph/build_cot_rewrite_from_deepmath.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_data(input_path: str) -> List[Dict[str, Any]]:
    """Load ShareGPT-format JSONL and extract question / original_cot / solution."""
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            convs = data.get("conversations", [])
            if len(convs) < 2:
                continue
            question = convs[0].get("value", convs[0].get("content", "")).strip()
            assistant_content = convs[1].get("value", convs[1].get("content", ""))
            if "<think>" in assistant_content:
                original_cot = assistant_content.split("<think>")[-1].split("</think>")[0].strip()
                solution = assistant_content.split("</think>")[-1].strip()
            else:
                original_cot = assistant_content.strip()
                solution = ""
            records.append({
                "question": question,
                "original_cot": original_cot,
                "solution": solution,
            })
    return records


# ---------------------------------------------------------------------------
# Step 1: Model loading / format detection + conversion
# ---------------------------------------------------------------------------

def detect_checkpoint_format(checkpoint_path: str) -> str:
    """Detect the format of a model checkpoint."""
    # Check for HF format: config.json in root
    if os.path.exists(os.path.join(checkpoint_path, "config.json")):
        return "hf"
    # Check for LoRA adapter
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        return "lora"
    # Check for FSDP format in actor/ subdirectory
    actor_dir = os.path.join(checkpoint_path, "actor")
    if os.path.isdir(actor_dir):
        for fname in os.listdir(actor_dir):
            if fname.startswith("model_world_size") or fname == "fsdp_config.json":
                return "fsdp"
        # Fallback: if there are .pt files inside, likely FSDP
        pt_files = [f for f in os.listdir(actor_dir) if f.endswith(".pt") or f.endswith(".bin")]
        if pt_files:
            return "fsdp"
    raise ValueError(
        f"Cannot detect checkpoint format at: {checkpoint_path}\n"
        "Expected one of: HF (config.json), LoRA (adapter_config.json), "
        "FSDP (actor/ directory with weight shards)"
    )


def step_model_loading(
    config: Dict[str, Any],
    ckpt_mgr,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    """
    Step 1: Ensure model exists in HuggingFace format.

    Returns the path to the HF model directory.
    """
    checkpoint_path = config["model"]["checkpoint_path"]
    hf_path = os.path.join(checkpoint_path, "actor", "huggingface")

    # Already merged?
    if os.path.exists(os.path.join(hf_path, "config.json")):
        logger.info(f"[Step1] HF model already exists at {hf_path}")
        ckpt_mgr.mark_step_complete("model", hf_path, 0)
        return hf_path

    fmt = config["model"].get("format", "auto")
    if fmt == "auto":
        fmt = detect_checkpoint_format(checkpoint_path)
    logger.info(f"[Step1] Detected checkpoint format: {fmt}")

    if dry_run:
        logger.info(f"[Step1] DRY-RUN: would convert {fmt} → HF at {hf_path}")
        return hf_path

    if fmt == "hf":
        # checkpoint_path itself is already HF
        logger.info(f"[Step1] Checkpoint is already HF format: {checkpoint_path}")
        ckpt_mgr.mark_step_complete("model", checkpoint_path, 0)
        return checkpoint_path

    elif fmt == "fsdp":
        actor_dir = os.path.join(checkpoint_path, "actor")
        logger.info(f"[Step1] Merging FSDP checkpoint: {actor_dir} → {hf_path}")
        os.makedirs(hf_path, exist_ok=True)
        cmd = [
            sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", actor_dir,
            "--target_dir", hf_path,
        ]
        logger.info(f"[Step1] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    elif fmt == "lora":
        logger.info(f"[Step1] Merging LoRA checkpoint: {checkpoint_path} → {hf_path}")
        os.makedirs(hf_path, exist_ok=True)
        cmd = [
            sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "lora",
            "--local_dir", checkpoint_path,
            "--target_dir", hf_path,
        ]
        logger.info(f"[Step1] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    else:
        raise ValueError(f"Unsupported checkpoint format: {fmt}")

    logger.info(f"[Step1] Model ready at {hf_path}")
    ckpt_mgr.mark_step_complete("model", hf_path, 0)
    return hf_path


# ---------------------------------------------------------------------------
# Step 2: Rollout - vLLM multi-GPU data parallel via Ray
# ---------------------------------------------------------------------------

import ray


@ray.remote
class VLLMRolloutWorker:
    """Single vLLM worker for data-parallel rollout."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],
        tp_size: int,
        dtype: str,
        max_model_len: int,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

        from vllm import LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=False,
        )

    def generate_batch(
        self,
        prompt_texts: List[str],  # already formatted (system+user merged)
        num_rollouts: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> List[List[str]]:
        """
        Generate `num_rollouts` completions per prompt.
        Returns List[List[str]] shape [n_prompts, num_rollouts].
        """
        from vllm import SamplingParams
        sp = SamplingParams(
            n=num_rollouts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        # Repeat prompts for vLLM (n= handles it natively)
        outputs = self.llm.generate(prompt_texts, sampling_params=sp, use_tqdm=False)
        results = []
        for out in outputs:
            completions = [c.text for c in out.outputs]
            results.append(completions)
        return results


def _build_vllm_prompt(tokenizer, question: str, original_cot: str) -> str:
    """Build the full vLLM prompt string using the tokenizer's chat template."""
    system_prompt, user_prompt = build_prompt(question, original_cot)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: plain concatenation
        text = f"{system_prompt}\n\n{user_prompt}"
    return text


def step_rollout(
    config: Dict[str, Any],
    ckpt_mgr,
    hf_model_path: str,
    records: List[Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    from tqdm import tqdm

    output_file = os.path.join(output_dir, "intermediate", "rollout_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    already_done = ckpt_mgr.get_step_completed_count("rollout")
    total = len(records)

    if already_done >= total and os.path.exists(output_file):
        logger.info(f"[Step2] Rollout already complete ({already_done}/{total}), skipping")
        return output_file

    logger.info(f"[Step2] Rollout: {total} samples, resuming from {already_done}")

    if dry_run:
        logger.info("[Step2] DRY-RUN: skipping rollout generation")
        return output_file

    rollout_cfg = config["rollout"]
    gen_cfg = rollout_cfg["generation"]
    parallel_cfg = rollout_cfg["parallel"]
    num_rollouts = rollout_cfg["num_rollouts"]
    tp_size = parallel_cfg.get("tensor_parallel_size", 1)

    # Determine GPU allocation
    gpu_ids_cfg = parallel_cfg.get("gpu_ids")
    if gpu_ids_cfg:
        all_gpu_ids = list(gpu_ids_cfg)
    else:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        all_gpu_ids = list(range(n))

    if not all_gpu_ids:
        raise RuntimeError("No GPUs available for rollout. Set rollout.parallel.gpu_ids in config.")

    num_workers = len(all_gpu_ids) // tp_size
    if num_workers == 0:
        raise RuntimeError(f"gpu_ids={all_gpu_ids} too few for tp_size={tp_size}")

    logger.info(f"[Step2] Workers={num_workers}, tp_size={tp_size}, gpu_ids={all_gpu_ids}")

    # Load tokenizer for prompt building
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    # Initialize Ray workers
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    workers = []
    for w in range(num_workers):
        w_gpus = all_gpu_ids[w * tp_size: (w + 1) * tp_size]
        worker = VLLMRolloutWorker.options(num_gpus=0, num_cpus=1).remote(
            model_path=hf_model_path,
            gpu_ids=w_gpus,
            tp_size=tp_size,
            dtype=config["model"]["dtype"],
            max_model_len=config["model"]["max_model_len"],
        )
        workers.append(worker)

    # Split unfinished records across workers
    remaining = records[already_done:]
    chunk_size = max(1, len(remaining) // num_workers)
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = len(remaining) if i == num_workers - 1 else min(start + chunk_size, len(remaining))
        chunks.append((already_done + start, remaining[start:end]))

    # Build prompts per chunk
    def build_prompts_for_chunk(chunk_records):
        return [
            _build_vllm_prompt(tokenizer, r["question"], r["original_cot"])
            for r in chunk_records
        ]

    # Submit all chunks
    futures = []
    for w_idx, (start_idx, chunk) in enumerate(chunks):
        if not chunk:
            continue
        prompts = build_prompts_for_chunk(chunk)
        fut = workers[w_idx].generate_batch.remote(
            prompts,
            num_rollouts,
            gen_cfg["temperature"],
            gen_cfg["top_p"],
            gen_cfg["max_new_tokens"],
        )
        futures.append((w_idx, start_idx, chunk, fut))

    # Collect results and write JSONL
    mode = "a" if already_done > 0 else "w"
    completed = already_done
    with open(output_file, mode) as fout:
        for w_idx, start_idx, chunk, fut in tqdm(futures, desc="Rollout workers", unit="worker"):
            batch_results = ray.get(fut)
            for i, (record, completions) in enumerate(zip(chunk, batch_results)):
                out_record = {
                    "question": record["question"],
                    "original_cot": record["original_cot"],
                    "solution": record["solution"],
                    "rewrite_cots": [{"rewrite_cot": c} for c in completions],
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                completed += 1

            ckpt_mgr.update_step_progress("rollout", completed, output_file)

    ckpt_mgr.mark_step_complete("rollout", output_file, completed)
    logger.info(f"[Step2] Rollout complete: {completed} samples → {output_file}")

    # Shutdown workers
    for w in workers:
        ray.kill(w, no_restart=True)

    return output_file


# ---------------------------------------------------------------------------
# Step 3: Correct Tag
# ---------------------------------------------------------------------------

def step_correct_tag(
    config: Dict[str, Any],
    ckpt_mgr,
    rollout_file: str,
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    output_file = os.path.join(output_dir, "intermediate", "correct_tag_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cj_cfg = config["correct_judge"]
    if not cj_cfg.get("enabled", True):
        logger.info("[Step3] Correct judge disabled, copying rollout output")
        shutil.copy2(rollout_file, output_file)
        return output_file

    # Read input
    records = _read_jsonl(rollout_file)
    total_records = len(records)
    already_done = ckpt_mgr.get_step_completed_count("correct_tag")

    if already_done >= total_records and os.path.exists(output_file):
        logger.info(f"[Step3] Correct tag already complete ({already_done}/{total_records}), skipping")
        return output_file

    logger.info(f"[Step3] Correct tag: {total_records} records, resuming from {already_done}")

    if dry_run:
        logger.info("[Step3] DRY-RUN: skipping correct judge")
        return output_file

    model_path = cj_cfg.get("model_path")
    if not model_path:
        raise ValueError("correct_judge.model_path is required when correct_judge.enabled=true")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from verl.experimental.reward.cot_rewriter.sglang_correct_judge import SGLangCorrectJudgeClient

    gen = cj_cfg["generation"]
    devices = _get_judge_devices(cj_cfg)
    client = SGLangCorrectJudgeClient(
        model_path=model_path,
        tensor_parallel_size=cj_cfg.get("tensor_parallel_size", 1),
        mem_fraction_static=cj_cfg.get("mem_fraction_static", 0.4),
        dtype=cj_cfg.get("dtype", "float16"),
        max_model_len=cj_cfg.get("max_model_len", 8192),
        devices=devices,
        temperature=gen.get("temperature", 0.0),
        max_new_tokens=gen.get("max_new_tokens", 50),
        micro_batch_size=cj_cfg.get("batch_size", 8),
    )

    try:
        mode = "a" if already_done > 0 else "w"
        completed = already_done
        batch_size = cj_cfg.get("batch_size", 8)

        pending_records = records[already_done:]

        # Flatten all rewrite_cots with record/rewrite indices
        flat_questions = []
        flat_rewrite_cots = []
        flat_solutions = []
        flat_indices = []  # (record_idx, rewrite_idx)

        for rec_idx, record in enumerate(pending_records):
            for rw_idx, rw in enumerate(record.get("rewrite_cots", [])):
                flat_questions.append(record["question"])
                flat_rewrite_cots.append(rw["rewrite_cot"])
                flat_solutions.append(record["solution"])
                flat_indices.append((rec_idx, rw_idx))

        # Run judge in batches
        from tqdm import tqdm
        is_correct_results = [None] * len(flat_questions)
        for i in tqdm(range(0, len(flat_questions), batch_size * client.num_actors),
                      desc="Correct judge", unit="batch"):
            end = min(i + batch_size * client.num_actors, len(flat_questions))
            batch_q = flat_questions[i:end]
            batch_r = flat_rewrite_cots[i:end]
            batch_s = flat_solutions[i:end]
            result = client.compute_correct_judge(batch_q, batch_r, batch_s)
            for j, is_corr in enumerate(result["is_correct"]):
                is_correct_results[i + j] = is_corr

        # Fill results back into records
        for (rec_idx, rw_idx), is_corr in zip(flat_indices, is_correct_results):
            pending_records[rec_idx]["rewrite_cots"][rw_idx]["is_correct"] = is_corr

        # Write output
        with open(output_file, mode) as fout:
            for record in pending_records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
            ckpt_mgr.update_step_progress("correct_tag", completed, output_file)

    finally:
        client.shutdown()

    ckpt_mgr.mark_step_complete("correct_tag", output_file, completed)
    logger.info(f"[Step3] Correct tag complete: {completed} records → {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Step 4: FSF Tag
# ---------------------------------------------------------------------------

def step_fsf_tag(
    config: Dict[str, Any],
    ckpt_mgr,
    correct_file: str,
    output_dir: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> str:
    output_file = os.path.join(output_dir, "intermediate", "fsf_tag_output.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fsf_cfg = config["fsf_judge"]
    if not fsf_cfg.get("enabled", True):
        logger.info("[Step4] FSF judge disabled, copying correct_tag output")
        shutil.copy2(correct_file, output_file)
        return output_file

    records = _read_jsonl(correct_file)
    total_records = len(records)
    already_done = ckpt_mgr.get_step_completed_count("fsf_tag")

    if already_done >= total_records and os.path.exists(output_file):
        logger.info(f"[Step4] FSF tag already complete ({already_done}/{total_records}), skipping")
        return output_file

    logger.info(f"[Step4] FSF tag: {total_records} records, resuming from {already_done}")

    if dry_run:
        logger.info("[Step4] DRY-RUN: skipping FSF judge")
        return output_file

    model_path = fsf_cfg.get("model_path")
    if not model_path:
        raise ValueError("fsf_judge.model_path is required when fsf_judge.enabled=true")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient

    gen = fsf_cfg["generation"]
    devices = _get_judge_devices(fsf_cfg)
    client = SGLangFSFClient(
        model_path=model_path,
        tensor_parallel_size=fsf_cfg.get("tensor_parallel_size", 1),
        mem_fraction_static=fsf_cfg.get("mem_fraction_static", 0.4),
        dtype=fsf_cfg.get("dtype", "float16"),
        max_model_len=fsf_cfg.get("max_model_len", 8192),
        devices=devices,
        temperature=gen.get("temperature", 0.6),
        max_new_tokens=gen.get("max_new_tokens", 8000),
        micro_batch_size=fsf_cfg.get("batch_size", 4),
    )

    try:
        mode = "a" if already_done > 0 else "w"
        completed = already_done
        batch_size = fsf_cfg.get("batch_size", 4)

        pending_records = records[already_done:]

        # Flatten
        flat_rewrite_cots = []
        flat_indices = []
        for rec_idx, record in enumerate(pending_records):
            for rw_idx, rw in enumerate(record.get("rewrite_cots", [])):
                flat_rewrite_cots.append(rw["rewrite_cot"])
                flat_indices.append((rec_idx, rw_idx))

        # Run FSF in batches
        from tqdm import tqdm
        fsf_results = [None] * len(flat_rewrite_cots)
        dot_results = [None] * len(flat_rewrite_cots)

        for i in tqdm(range(0, len(flat_rewrite_cots), batch_size * client.num_actors),
                      desc="FSF judge", unit="batch"):
            end = min(i + batch_size * client.num_actors, len(flat_rewrite_cots))
            batch_cots = flat_rewrite_cots[i:end]
            result = client.compute_rewrite_fsf_only(batch_cots, return_dot_outputs=True)
            for j, (fsf_val, dot_out) in enumerate(zip(
                result["rewrite_fsfs"], result.get("rewrite_dot_outputs", [None] * len(batch_cots))
            )):
                fsf_results[i + j] = fsf_val
                dot_results[i + j] = dot_out

        # Fill results back
        for (rec_idx, rw_idx), fsf_val, dot_out in zip(flat_indices, fsf_results, dot_results):
            pending_records[rec_idx]["rewrite_cots"][rw_idx]["fsf"] = fsf_val
            pending_records[rec_idx]["rewrite_cots"][rw_idx]["fsf_model_output"] = dot_out

        # Add fsf_tag at record level: True if any rewrite has fsf > 0
        for record in pending_records:
            record["fsf_tag"] = any(
                (rw.get("fsf") or 0) > 0
                for rw in record.get("rewrite_cots", [])
            )

        # Write output
        with open(output_file, mode) as fout:
            for record in pending_records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
            ckpt_mgr.update_step_progress("fsf_tag", completed, output_file)

    finally:
        client.shutdown()

    ckpt_mgr.mark_step_complete("fsf_tag", output_file, completed)
    logger.info(f"[Step4] FSF tag complete: {completed} records → {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_judge_devices(judge_cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Return device list from judge config, or None for auto-detect."""
    gpu_ids = judge_cfg.get("gpu_ids")
    if gpu_ids:
        return [f"cuda:{i}" for i in gpu_ids]
    return None


STEP_ORDER = ["model", "rollout", "correct", "fsf"]


def determine_start_step(
    args: argparse.Namespace,
    ckpt_mgr,
    config: Dict[str, Any],
) -> str:
    if args.step:
        step_map = {"model": "model", "rollout": "rollout", "correct": "correct", "fsf": "fsf"}
        return step_map.get(args.step, "model")

    if args.resume or config.get("resume", True):
        state = ckpt_mgr.load()
        if state:
            completed = state["progress"]["completed_steps"]
            for step in STEP_ORDER:
                if step not in completed:
                    return step
            return "done"

    return "model"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CoT Rewrite Inference Pipeline (steps 1-4)"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--step",
        choices=["model", "rollout", "correct", "fsf"],
        help="Start from this step (overrides resume logic)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Force resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check config and paths without running anything",
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
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir, "pipeline")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output dir: {output_dir}")

    # Copy config to output dir (for reference)
    config_copy = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(config_copy):
        shutil.copy2(args.config, config_copy)

    ckpt_mgr = CheckpointManager(output_dir)

    if args.dry_run:
        logger.info("DRY-RUN mode enabled - will check paths without executing")

    start_step = determine_start_step(args, ckpt_mgr, config)

    if start_step == "done":
        logger.info("All steps already complete.")
        return

    logger.info(f"Starting from step: {start_step}")

    hf_model_path = None
    rollout_file = None
    correct_file = None

    # Step 1: Model loading
    if start_step == "model":
        hf_model_path = step_model_loading(config, ckpt_mgr, logger, dry_run=args.dry_run)
        start_step = "rollout"
    else:
        # Recover hf_model_path from checkpoint or detect
        saved_path = ckpt_mgr.get_step_output("model")
        if saved_path and os.path.exists(saved_path):
            hf_model_path = saved_path
        else:
            # Try default path
            cp = config["model"]["checkpoint_path"]
            candidate = os.path.join(cp, "actor", "huggingface")
            if os.path.exists(os.path.join(candidate, "config.json")):
                hf_model_path = candidate
            elif os.path.exists(os.path.join(cp, "config.json")):
                hf_model_path = cp
            else:
                logger.warning(f"Cannot find HF model path, will try to detect from checkpoint")
                hf_model_path = step_model_loading(config, ckpt_mgr, logger, dry_run=args.dry_run)

    # Step 2: Rollout
    if start_step == "rollout":
        records = load_input_data(config["rollout"]["input_data"])
        logger.info(f"Loaded {len(records)} input records")
        ckpt_mgr.init_state(
            {"model_path": hf_model_path, "input_data": config["rollout"]["input_data"]},
            len(records),
        )
        rollout_file = step_rollout(
            config, ckpt_mgr, hf_model_path, records, output_dir, logger, dry_run=args.dry_run
        )
        start_step = "correct"
    else:
        rollout_file = ckpt_mgr.get_step_output("rollout")
        if not rollout_file or not os.path.exists(rollout_file):
            raise FileNotFoundError(
                f"Rollout output not found. Run with --step rollout first.\n"
                f"Expected: {rollout_file}"
            )

    # Step 3: Correct Tag
    if start_step == "correct":
        correct_file = step_correct_tag(
            config, ckpt_mgr, rollout_file, output_dir, logger, dry_run=args.dry_run
        )
        start_step = "fsf"
    else:
        correct_file = ckpt_mgr.get_step_output("correct_tag")
        if not correct_file or not os.path.exists(correct_file):
            raise FileNotFoundError(
                f"Correct tag output not found. Run with --step correct first.\n"
                f"Expected: {correct_file}"
            )

    # Step 4: FSF Tag
    if start_step == "fsf":
        step_fsf_tag(
            config, ckpt_mgr, correct_file, output_dir, logger, dry_run=args.dry_run
        )

    logger.info("Pipeline steps 1-4 complete.")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
