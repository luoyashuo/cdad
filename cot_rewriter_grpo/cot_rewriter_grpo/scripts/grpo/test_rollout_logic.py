#!/usr/bin/env python3
"""
测试FSDP-vLLM rollout逻辑，模拟训练时的调用流程。

用途：
1. 验证FSDP到vLLM的权重同步是否正确
2. 测试不同配置下的格式正确率
3. 调试rollout调用链路

运行方式：
    CUDA_VISIBLE_DEVICES=0 python scripts/grpo/test_rollout_logic.py --log-level INFO
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from contextlib import nullcontext

# Add verl to path
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo/verl')

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.device_mesh import init_device_mesh

# VERL imports
from verl.protocol import DataProto


def parse_args():
    parser = argparse.ArgumentParser(description="Test FSDP-vLLM rollout logic")
    parser.add_argument("--model", type=str,
                        default="/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B")
    parser.add_argument("--data-path", type=str,
                        default="/data/luoys/data/cdad/cot_rewrite_max_fsf_v3/train.jsonl")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--max-prompt-length", type=int, default=5500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"])
    parser.add_argument("--load-format", type=str, default="dummy_dtensor",
                        choices=["dummy_dtensor", "safetensors", "dummy_hf"],
                        help="vLLM load format (dummy_dtensor simulates training)")
    parser.add_argument("--check-weight-sync", action="store_true",
                        help="Compare FSDP and vLLM weights after sync")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup logging with VERL format."""
    import logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Set VERL logger level
    os.environ["VERL_LOGGING_LEVEL"] = log_level
    logger = logging.getLogger(__name__)
    return logger


def check_format(response: str) -> tuple[bool, str]:
    """Check if response has valid <rewrite_reasoning> tags."""
    import re
    pattern = r'<rewrite_reasoning>\s*(.*?)\s*</rewrite_reasoning>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True, match.group(1).strip()
    return False, response


def load_sample_data(data_path: str, sample_idx: int = 0):
    """Load a sample from training data."""
    with open(data_path) as f:
        lines = f.readlines()
        # Find sample with "Task B-2.2" if requested
        if sample_idx == -1:
            for line in lines:
                data = json.loads(line)
                if "Task B-2.2" in data.get("question", ""):
                    return data
            sample_idx = 0
        data = json.loads(lines[sample_idx])
    return data


def create_simple_rollout_config(args):
    """Create minimal rollout config for testing."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "name": "vllm",
        "mode": "sync",
        "load_format": args.load_format,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_model_parallel_size": 1,
        "max_model_len": 15500,
        "trust_remote_code": True,
        "enforce_eager": False,
        "skip_tokenizer_init": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": -1,
        "max_tokens": args.max_tokens,
        "n": args.num_samples,
        "logprobs": 0,
        "detokenize": False,
        "free_cache_engine": True,
        "sleep_level": 2,
        # Additional required parameters
        "log_prob_micro_batch_size": 1,
        "log_prob_micro_batch_size_per_gpu": 1,
        "log_prob_max_token_len_per_gpu": 2048,
        "log_prob_use_dynamic_bsz": False,
        "prompt_length": args.max_prompt_length,
        "response_length": args.max_tokens,
    })
    return config


def create_actor_config(args):
    """Create minimal actor config for testing."""
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "strategy": "fsdp2",
        "fsdp_config": {
            "fsdp_size": -1,
            "sharding_strategy": "FULL_SHARD",
            "mixed_precision": "bf16",
            "param_offload": False,
            "optimizer_offload": False,
            "offload_policy": None,
        },
        "use_remove_padding": False,
        "use_fused_kernels": False,
    })
    return config


def test_direct_vllm(args, logger):
    """Test direct vLLM inference (baseline)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Direct vLLM (baseline, no FSDP)")
    logger.info("=" * 80)

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = load_sample_data(args.data_path, sample_idx=-1)
    prompt_content = data["problem"]

    messages = [{"role": "user", "content": prompt_content}]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    logger.info(f"Prompt length: {len(tokenizer.encode(prompt_str))} tokens")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    start_time = time.time()
    outputs = llm.generate([prompt_str], sampling_params)
    elapsed = time.time() - start_time

    valid_count = 0
    for i, out in enumerate(outputs[0].outputs):
        is_valid, content = check_format(out.text)
        if is_valid:
            valid_count += 1
        if args.verbose or not is_valid:
            logger.info(f"Sample {i+1}: {'✓ VALID' if is_valid else '✗ INVALID'} (finish: {out.finish_reason})")

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Valid format: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.0f}%)")

    return valid_count


def test_vllm_with_token_ids(args, logger):
    """Test vLLM with prompt_token_ids (simulates training)."""
    logger.info("=" * 80)
    logger.info("TEST 2: vLLM with prompt_token_ids")
    logger.info("=" * 80)

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = load_sample_data(args.data_path, sample_idx=-1)
    prompt_content = data["problem"]

    messages = [{"role": "user", "content": prompt_content}]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get token ids
    raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    logger.info(f"Prompt length: {len(raw_prompt_ids)} tokens")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    start_time = time.time()
    # Use prompt_token_ids like training does
    outputs = llm.generate([{"prompt_token_ids": raw_prompt_ids}], sampling_params)
    elapsed = time.time() - start_time

    valid_count = 0
    for i, out in enumerate(outputs[0].outputs):
        is_valid, content = check_format(out.text)
        if is_valid:
            valid_count += 1
        if args.verbose or not is_valid:
            logger.info(f"Sample {i+1}: {'✓ VALID' if is_valid else '✗ INVALID'} (finish: {out.finish_reason})")

    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"Valid format: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.0f}%)")

    return valid_count


def test_fsdp_vllm_rollout(args, logger):
    """Test FSDP-vLLM rollout (simulates actual training)."""
    logger.info("=" * 80)
    logger.info("TEST 3: FSDP-vLLM Hybrid Engine (load_format=%s)", args.load_format)
    logger.info("=" * 80)

    # Initialize distributed for single GPU
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Only init if not already initialized
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )

    from omegaconf import OmegaConf

    # Create minimal config
    rollout_config = create_simple_rollout_config(args)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load sample data
    data = load_sample_data(args.data_path, sample_idx=-1)
    prompt_content = data["problem"]

    messages = [{"role": "user", "content": prompt_content}]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    logger.info(f"Prompt length: {len(tokenizer.encode(prompt_str))} tokens")

    # Create FSDP worker with rollout
    logger.info("Initializing FSDP worker with vLLM rollout...")

    # Build worker config
    worker_config = OmegaConf.create({
        "model": {
            "path": args.model,
            "trust_remote_code": True,
            "use_remove_padding": False,
        },
        "rollout": rollout_config,
        "actor": create_actor_config(args),
        "ref": OmegaConf.create({"fsdp_config": {}}),
        "reward_model": OmegaConf.create({"fsdp_config": {}}),
    })
    worker_config.profiler = None  # Disable profiler

    # Create FSDP rollout worker
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    worker = ActorRolloutRefWorker(
        config=worker_config,
        role="rollout",
    )

    # Initialize model
    logger.info("Calling init_model()...")
    worker.init_model()

    # Prepare input DataProto
    logger.info("Preparing input DataProto...")

    # Tokenize
    model_inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]

    # Pad to max_prompt_length
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    max_length = args.max_prompt_length

    if len(input_ids) < max_length:
        pad_length = max_length - len(input_ids)
        # Left padding like training does
        input_ids = torch.cat([
            torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype),
            input_ids
        ])
        attention_mask = torch.cat([
            torch.zeros(pad_length, dtype=attention_mask.dtype),
            attention_mask
        ])

    # Create position_ids
    position_ids = torch.cumsum(attention_mask, dim=0) - 1
    position_ids = position_ids.clamp(min=0)

    # Create DataProto
    prompts_batch = DataProto.from_dict({
        "prompts": input_ids.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
    })

    # Set meta_info
    prompts_batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_token_id,
        "do_sample": True,
    }

    # Set raw_prompt_ids (for vLLM)
    raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompts_batch.non_tensor_batch = {"raw_prompt_ids": [raw_prompt_ids]}

    logger.info("Calling generate_sequences()...")
    start_time = time.time()

    # Generate sequences
    output = worker.generate_sequences(prompts_batch)

    elapsed = time.time() - start_time
    logger.info(f"Generation time: {elapsed:.2f}s")

    # Decode outputs
    responses = output.batch["responses"]
    logger.info(f"Generated {len(responses)} responses")

    # Check format
    valid_count = 0
    for i, response_ids in enumerate(responses):
        # Decode response
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        is_valid, content = check_format(response_text)
        if is_valid:
            valid_count += 1
        if args.verbose or not is_valid:
            logger.info(f"Sample {i+1}: {'✓ VALID' if is_valid else '✗ INVALID'}")
            if not is_valid and args.verbose:
                logger.info(f"  Response preview: {response_text[:200]}...")

    logger.info(f"Valid format: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.0f}%)")

    return valid_count


def compare_weights(args, logger):
    """Compare FSDP and vLLM weights after sync."""
    logger.info("=" * 80)
    logger.info("TEST 4: Weight Comparison (FSDP vs vLLM)")
    logger.info("=" * 80)

    if not args.check_weight_sync:
        logger.info("Skipping weight comparison (use --check-weight-sync to enable)")
        return

    # Initialize distributed
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )

    from omegaconf import OmegaConf
    import torch.nn.functional as F

    # Create configs
    rollout_config = create_simple_rollout_config(args)
    worker_config = OmegaConf.create({
        "model": {
            "path": args.model,
            "trust_remote_code": True,
        },
        "rollout": rollout_config,
        "actor": create_actor_config(args),
        "ref": OmegaConf.create({"fsdp_config": {}}),
        "reward_model": OmegaConf.create({"fsdp_config": {}}),
    })
    worker_config.profiler = None  # Disable profiler

    # Create worker
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    worker = ActorRolloutRefWorker(
        config=worker_config,
        role="rollout",
    )
    worker.init_model()

    # Get FSDP model weights
    logger.info("Extracting FSDP weights...")
    fsdp_weights = {}
    for name, param in worker.actor_module_fsdp.named_parameters():
        if "_fsdp_wrapped_module" in name:
            # Get the wrapped module name
            name = name.replace("_fsdp_wrapped_module.", "")
        fsdp_weights[name] = param.detach().cpu()

    logger.info(f"FSDP has {len(fsdp_weights)} parameters")

    # Get vLLM model weights
    logger.info("Extracting vLLM weights...")
    vllm_model = worker.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    vllm_weights = {}
    for name, param in vllm_model.named_parameters():
        vllm_weights[name] = param.detach().cpu()

    logger.info(f"vLLM has {len(vllm_weights)} parameters")

    # Compare weights
    logger.info("Comparing weights...")
    matched = 0
    mismatched = 0
    not_found_in_vllm = 0

    for fsdp_name, fsdp_weight in fsdp_weights.items():
        if fsdp_name in vllm_weights:
            vllm_weight = vllm_weights[fsdp_name]
            if fsdp_weight.shape != vllm_weight.shape:
                logger.warning(f"Shape mismatch: {fsdp_name} {fsdp_weight.shape} vs {vllm_weight.shape}")
                mismatched += 1
            else:
                diff = torch.abs(fsdp_weight - vllm_weight).max().item()
                if diff < 1e-5:
                    matched += 1
                else:
                    logger.warning(f"Weight mismatch: {fsdp_name} max_diff={diff}")
                    mismatched += 1
        else:
            not_found_in_vllm += 1

    logger.info(f"Matched: {matched}, Mismatched: {mismatched}, Not found in vLLM: {not_found_in_vllm}")

    if mismatched > 0:
        logger.warning("⚠ Weight sync issues detected!")
    else:
        logger.info("✓ Weights appear to be synchronized correctly")


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("FSDP-vLLM Rollout Logic Test")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Load format: {args.load_format}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 80)

    results = {}

    try:
        # Test 1: Direct vLLM (baseline)
        results["direct_vllm"] = test_direct_vllm(args, logger)
        logger.info("")

        # Test 2: vLLM with prompt_token_ids
        results["vllm_token_ids"] = test_vllm_with_token_ids(args, logger)
        logger.info("")

        # Test 3: FSDP-vLLM rollout
        results["fsdp_vllm"] = test_fsdp_vllm_rollout(args, logger)
        logger.info("")

        # Test 4: Weight comparison
        compare_weights(args, logger)

    except Exception as e:
        logger.exception(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for test_name, valid_count in results.items():
        logger.info(f"  {test_name}: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.0f}%)")

    # Compare results
    if results.get("direct_vllm", 0) != results.get("fsdp_vllm", 0):
        logger.warning("⚠ Different results between direct vLLM and FSDP-vLLM!")
    else:
        logger.info("✓ Results are consistent across tests")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
