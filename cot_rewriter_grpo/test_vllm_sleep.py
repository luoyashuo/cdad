#!/usr/bin/env python3
"""
Test vLLM Sleep Mode for GPU Memory Release

This script tests the vLLM sleep mode functionality to verify that GPU memory
is properly released when calling sleep(level=1) or sleep(level=2).

Usage:
    python test_vllm_sleep.py --model /path/to/model --gpu 0 --gpu-memory-utilization 0.7

Sleep Levels:
    - level=1: Offload model weights to CPU memory, discard KV cache
               Wake up is fast (~0.1-0.8s) but requires sufficient CPU RAM
    - level=2: Completely discard weights and KV cache
               Wake up requires reloading from disk (~0.8-2.6s) but uses minimal CPU RAM

Reference: https://docs.vllm.ai/en/latest/features/sleep_mode/
"""

import argparse
import gc
import os
import subprocess
import time

import torch


def get_gpu_memory_nvidia_smi(gpu_id: int) -> dict:
    """Get GPU memory using nvidia-smi (works across all processes)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        total, used, free = map(float, result.stdout.strip().split(", "))
        return {
            "total_gb": round(total / 1024, 2),
            "used_gb": round(used / 1024, 2),
            "free_gb": round(free / 1024, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpu_memory_pynvml(gpu_id: int) -> dict:
    """Get GPU memory using pynvml (works across all processes)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "total_gb": round(info.total / (1024**3), 2),
            "used_gb": round(info.used / (1024**3), 2),
            "free_gb": round(info.free / (1024**3), 2),
        }
    except ImportError:
        return get_gpu_memory_nvidia_smi(gpu_id)
    except Exception as e:
        return get_gpu_memory_nvidia_smi(gpu_id)


def print_gpu_memory(gpu_id: int, stage: str):
    """Print GPU memory status using nvidia-smi (sees all processes)."""
    info = get_gpu_memory_pynvml(gpu_id)
    print(f"\n{'='*60}")
    print(f"[{stage}] GPU {gpu_id} Memory Status (nvidia-smi):")
    print(f"  Total:     {info['total_gb']:.2f} GB")
    print(f"  Used:      {info['used_gb']:.2f} GB")
    print(f"  Free:      {info['free_gb']:.2f} GB")
    print(f"{'='*60}")
    return info


def test_sleep_mode(model_path: str, gpu_id: int, gpu_memory_utilization: float):
    """Test vLLM sleep mode functionality."""

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force V0 engine for simpler memory tracking (optional)
    # os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM, SamplingParams

    print(f"\n{'#'*60}")
    print(f"# vLLM Sleep Mode Test")
    print(f"# Model: {model_path}")
    print(f"# GPU: {gpu_id}")
    print(f"# GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"{'#'*60}")

    # Initial GPU memory
    info_initial = print_gpu_memory(gpu_id, "Initial (before loading)")

    # ========== Phase 1: Initialize vLLM with sleep mode enabled ==========
    print("\n[Phase 1] Initializing vLLM with enable_sleep_mode=True...")
    start_time = time.time()

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_sleep_mode=True,  # IMPORTANT: Enable sleep mode
        enforce_eager=True,  # Disable CUDA graph for easier memory tracking
        trust_remote_code=True,
        dtype="float16",
    )

    load_time = time.time() - start_time
    print(f"  vLLM initialized in {load_time:.2f}s")

    # Wait a moment for memory to stabilize
    time.sleep(1)
    info_after_load = print_gpu_memory(gpu_id, "After vLLM initialization")

    # ========== Phase 2: Run a simple inference ==========
    print("\n[Phase 2] Running inference...")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=100,
    )

    prompts = ["Hello, how are you?", "What is the capital of France?"]

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - start_time

    print(f"  Inference completed in {inference_time:.2f}s")
    for i, output in enumerate(outputs):
        print(f"  Prompt {i}: {output.prompt[:50]}...")
        print(f"  Output {i}: {output.outputs[0].text[:100]}...")

    time.sleep(1)
    info_after_inference = print_gpu_memory(gpu_id, "After inference")

    # ========== Phase 3: Test sleep(level=1) ==========
    print("\n[Phase 3] Testing sleep(level=1) - Offload to CPU...")
    start_time = time.time()

    llm.sleep(level=1)

    sleep1_time = time.time() - start_time
    print(f"  sleep(level=1) completed in {sleep1_time:.2f}s")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # Wait for memory to be released

    info_after_sleep1 = print_gpu_memory(gpu_id, "After sleep(level=1)")

    memory_released_1 = info_after_inference['used_gb'] - info_after_sleep1['used_gb']
    print(f"\n  >>> Memory released by sleep(level=1): {memory_released_1:.2f} GB <<<")

    # ========== Phase 4: Wake up from level 1 ==========
    print("\n[Phase 4] Testing wake_up() from level 1...")
    start_time = time.time()

    llm.wake_up()

    wakeup1_time = time.time() - start_time
    print(f"  wake_up() from level 1 completed in {wakeup1_time:.2f}s")

    time.sleep(1)
    info_after_wakeup1 = print_gpu_memory(gpu_id, "After wake_up from level 1")

    # Verify inference still works
    print("\n  Verifying inference after wake_up...")
    outputs = llm.generate(["Test after wake up"], sampling_params)
    print(f"  Output: {outputs[0].outputs[0].text[:100]}...")

    time.sleep(1)
    info_after_inference2 = print_gpu_memory(gpu_id, "After inference (post wake_up)")

    # ========== Phase 5: Test sleep(level=2) ==========
    print("\n[Phase 5] Testing sleep(level=2) - Discard weights...")
    start_time = time.time()

    llm.sleep(level=2)

    sleep2_time = time.time() - start_time
    print(f"  sleep(level=2) completed in {sleep2_time:.2f}s")

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)  # Wait for memory to be released

    info_after_sleep2 = print_gpu_memory(gpu_id, "After sleep(level=2)")

    memory_released_2 = info_after_inference2['used_gb'] - info_after_sleep2['used_gb']
    print(f"\n  >>> Memory released by sleep(level=2): {memory_released_2:.2f} GB <<<")

    # ========== Phase 6: Wake up from level 2 ==========
    print("\n[Phase 6] Testing wake_up() from level 2 (reload from disk)...")
    start_time = time.time()

    # For level 2, we need to reload weights
    # Method 1: Two-stage wake_up for better memory management
    try:
        llm.wake_up(tags=["weights"])
        llm.wake_up(tags=["kv_cache"])
    except TypeError:
        # Fallback for older vLLM versions
        llm.wake_up()

    wakeup2_time = time.time() - start_time
    print(f"  wake_up() from level 2 completed in {wakeup2_time:.2f}s")

    time.sleep(1)
    info_after_wakeup2 = print_gpu_memory(gpu_id, "After wake_up from level 2")

    # Verify inference still works
    print("\n  Verifying inference after wake_up from level 2...")
    outputs = llm.generate(["Test after level 2 wake up: What is 2+2?"], sampling_params)
    print(f"  Output: {outputs[0].outputs[0].text[:100]}...")

    # ========== Summary ==========
    print(f"\n{'#'*60}")
    print("# Summary")
    print(f"{'#'*60}")
    print(f"\n  Timing:")
    print(f"    Model load time:           {load_time:.2f}s")
    print(f"    sleep(level=1) time:       {sleep1_time:.2f}s")
    print(f"    wake_up from level 1 time: {wakeup1_time:.2f}s")
    print(f"    sleep(level=2) time:       {sleep2_time:.2f}s")
    print(f"    wake_up from level 2 time: {wakeup2_time:.2f}s")
    print(f"\n  Memory Usage (GPU {gpu_id}):")
    print(f"    Initial:            {info_initial['used_gb']:.2f} GB")
    print(f"    After load:         {info_after_load['used_gb']:.2f} GB")
    print(f"    After inference:    {info_after_inference['used_gb']:.2f} GB")
    print(f"    After sleep(1):     {info_after_sleep1['used_gb']:.2f} GB")
    print(f"    After wake_up(1):   {info_after_wakeup1['used_gb']:.2f} GB")
    print(f"    After sleep(2):     {info_after_sleep2['used_gb']:.2f} GB")
    print(f"    After wake_up(2):   {info_after_wakeup2['used_gb']:.2f} GB")
    print(f"\n  Memory Released:")
    print(f"    By sleep(level=1):  {memory_released_1:.2f} GB")
    print(f"    By sleep(level=2):  {memory_released_2:.2f} GB")
    print(f"{'#'*60}")

    # Conclusion
    print(f"\n{'#'*60}")
    print("# Conclusion")
    print(f"{'#'*60}")
    if memory_released_1 > 1.0 or memory_released_2 > 1.0:
        print("  SUCCESS: Sleep mode is working correctly!")
        print(f"  sleep(level=1) released {memory_released_1:.2f} GB")
        print(f"  sleep(level=2) released {memory_released_2:.2f} GB")
    else:
        print("  WARNING: Sleep mode may not be releasing memory as expected.")
        print("  Check vLLM logs above for 'Sleep mode freed X GiB memory' messages.")
    print(f"{'#'*60}")

    # Cleanup
    print("\n[Cleanup] Deleting vLLM instance...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    print_gpu_memory(gpu_id, "Final (after cleanup)")

    return {
        "load_time": load_time,
        "sleep1_time": sleep1_time,
        "wakeup1_time": wakeup1_time,
        "sleep2_time": sleep2_time,
        "wakeup2_time": wakeup2_time,
        "memory_released_1": memory_released_1,
        "memory_released_2": memory_released_2,
    }


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Sleep Mode")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (default: 0.7)",
    )

    args = parser.parse_args()

    test_sleep_mode(
        model_path=args.model,
        gpu_id=args.gpu,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
