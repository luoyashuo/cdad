#!/usr/bin/env python3
"""
Test vLLM Sleep Mode: V0 vs V1 Engine Comparison

This script compares the sleep mode memory release behavior between
vLLM V0 and V1 engines.

Usage:
    # Test both V0 and V1
    python test_vllm_sleep_v0_v1.py --model /path/to/model --gpu 0

    # Test only V0
    python test_vllm_sleep_v0_v1.py --model /path/to/model --gpu 0 --engine v0

    # Test only V1
    python test_vllm_sleep_v0_v1.py --model /path/to/model --gpu 0 --engine v1
"""

import argparse
import gc
import os
import subprocess
import sys
import time


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
    except Exception:
        return get_gpu_memory_nvidia_smi(gpu_id)


def get_gpu_memory_nvidia_smi(gpu_id: int) -> dict:
    """Get GPU memory using nvidia-smi (fallback)."""
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
        return {"error": str(e), "total_gb": 0, "used_gb": 0, "free_gb": 0}


def print_gpu_memory(gpu_id: int, stage: str) -> dict:
    """Print GPU memory status."""
    info = get_gpu_memory_pynvml(gpu_id)
    print(f"  [{stage}] GPU {gpu_id}: Used={info['used_gb']:.2f}GB, Free={info['free_gb']:.2f}GB")
    return info


def test_engine(
    model_path: str,
    gpu_id: int,
    gpu_memory_utilization: float,
    use_v1: bool,
    sleep_level: int = 2,
) -> dict:
    """Test sleep mode for a specific engine version."""
    import torch

    engine_name = "V1" if use_v1 else "V0"
    print(f"\n{'='*60}")
    print(f"Testing vLLM {engine_name} Engine (sleep level={sleep_level})")
    print(f"{'='*60}")

    # Set environment variable for engine version
    if use_v1:
        os.environ.pop("VLLM_USE_V1", None)  # Remove to use V1 (default in newer vLLM)
    else:
        os.environ["VLLM_USE_V1"] = "0"  # Force V0

    # Clear GPU memory before test
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    info_initial = print_gpu_memory(gpu_id, "Initial")

    # Import vLLM after setting env var
    # We need to reload the module to pick up the new env var
    if "vllm" in sys.modules:
        # Force reimport
        import importlib
        import vllm
        importlib.reload(vllm)

    from vllm import LLM, SamplingParams

    results = {
        "engine": engine_name,
        "sleep_level": sleep_level,
        "initial_used_gb": info_initial["used_gb"],
    }

    try:
        # Initialize vLLM
        print(f"\n  Initializing vLLM {engine_name}...")
        start_time = time.time()

        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=True,
            enforce_eager=True,  # Disable CUDA graph
            trust_remote_code=True,
            dtype="float16",
        )

        load_time = time.time() - start_time
        print(f"  Initialized in {load_time:.2f}s")
        results["load_time"] = load_time

        time.sleep(1)
        info_after_load = print_gpu_memory(gpu_id, "After load")
        results["after_load_used_gb"] = info_after_load["used_gb"]

        # Run inference
        print(f"\n  Running inference...")
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
        outputs = llm.generate(["Hello, how are you?"], sampling_params)
        print(f"  Output: {outputs[0].outputs[0].text[:50]}...")

        time.sleep(1)
        info_after_inference = print_gpu_memory(gpu_id, "After inference")
        results["after_inference_used_gb"] = info_after_inference["used_gb"]

        # Test sleep
        print(f"\n  Calling sleep(level={sleep_level})...")
        start_time = time.time()
        llm.sleep(level=sleep_level)
        sleep_time = time.time() - start_time
        print(f"  sleep() completed in {sleep_time:.2f}s")
        results["sleep_time"] = sleep_time

        # Wait and clear cache
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

        info_after_sleep = print_gpu_memory(gpu_id, "After sleep")
        results["after_sleep_used_gb"] = info_after_sleep["used_gb"]

        # Calculate memory released
        memory_released = info_after_inference["used_gb"] - info_after_sleep["used_gb"]
        results["memory_released_gb"] = memory_released
        print(f"\n  >>> Memory released by sleep: {memory_released:.2f} GB <<<")

        # Test wake_up
        print(f"\n  Calling wake_up()...")
        start_time = time.time()
        try:
            llm.wake_up(tags=["weights"])
            llm.wake_up(tags=["kv_cache"])
        except TypeError:
            llm.wake_up()
        wakeup_time = time.time() - start_time
        print(f"  wake_up() completed in {wakeup_time:.2f}s")
        results["wakeup_time"] = wakeup_time

        time.sleep(1)
        info_after_wakeup = print_gpu_memory(gpu_id, "After wake_up")
        results["after_wakeup_used_gb"] = info_after_wakeup["used_gb"]

        # Verify inference still works
        print(f"\n  Verifying inference after wake_up...")
        outputs = llm.generate(["Test after wake up"], sampling_params)
        print(f"  Output: {outputs[0].outputs[0].text[:50]}...")

        results["success"] = True

    except Exception as e:
        print(f"\n  ERROR: {e}")
        results["success"] = False
        results["error"] = str(e)

    finally:
        # Cleanup
        print(f"\n  Cleaning up...")
        try:
            del llm
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        info_final = print_gpu_memory(gpu_id, "Final")
        results["final_used_gb"] = info_final["used_gb"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Sleep Mode: V0 vs V1")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["v0", "v1", "both"],
        default="both",
        help="Which engine to test",
    )
    parser.add_argument(
        "--sleep-level",
        type=int,
        choices=[1, 2],
        default=2,
        help="Sleep level to test",
    )

    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Use the first visible GPU (index 0) after CUDA_VISIBLE_DEVICES is set
    gpu_id_for_query = args.gpu  # For nvidia-smi query

    print(f"\n{'#'*60}")
    print(f"# vLLM Sleep Mode Test: V0 vs V1 Engine Comparison")
    print(f"{'#'*60}")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Sleep Level: {args.sleep_level}")
    print(f"Engine(s) to test: {args.engine}")

    results = {}

    # Test V0 engine
    if args.engine in ["v0", "both"]:
        results["v0"] = test_engine(
            model_path=args.model,
            gpu_id=gpu_id_for_query,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_v1=False,
            sleep_level=args.sleep_level,
        )

        # Need to restart Python to properly switch engines
        # For now, we'll just note this limitation
        if args.engine == "both":
            print("\n" + "="*60)
            print("NOTE: Testing V1 after V0 in same process may not be accurate.")
            print("For accurate comparison, run with --engine v0 and --engine v1 separately.")
            print("="*60)

    # Test V1 engine
    if args.engine in ["v1", "both"]:
        results["v1"] = test_engine(
            model_path=args.model,
            gpu_id=gpu_id_for_query,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_v1=True,
            sleep_level=args.sleep_level,
        )

    # Print comparison summary
    print(f"\n{'#'*60}")
    print(f"# Summary")
    print(f"{'#'*60}")

    for engine, result in results.items():
        print(f"\n{engine.upper()} Engine:")
        if result.get("success"):
            print(f"  Load time:        {result.get('load_time', 'N/A'):.2f}s")
            print(f"  Sleep time:       {result.get('sleep_time', 'N/A'):.2f}s")
            print(f"  Wake up time:     {result.get('wakeup_time', 'N/A'):.2f}s")
            print(f"  Memory after load:      {result.get('after_load_used_gb', 'N/A'):.2f} GB")
            print(f"  Memory after inference: {result.get('after_inference_used_gb', 'N/A'):.2f} GB")
            print(f"  Memory after sleep:     {result.get('after_sleep_used_gb', 'N/A'):.2f} GB")
            print(f"  Memory RELEASED:        {result.get('memory_released_gb', 'N/A'):.2f} GB")

            if result.get("memory_released_gb", 0) > 1.0:
                print(f"  Status: SUCCESS - Sleep mode is working!")
            else:
                print(f"  Status: WARNING - Sleep mode may not be releasing memory properly")
        else:
            print(f"  Status: FAILED - {result.get('error', 'Unknown error')}")

    # Comparison
    if "v0" in results and "v1" in results:
        print(f"\n{'#'*60}")
        print(f"# Comparison: V0 vs V1")
        print(f"{'#'*60}")

        v0_released = results["v0"].get("memory_released_gb", 0)
        v1_released = results["v1"].get("memory_released_gb", 0)

        print(f"\nMemory Released:")
        print(f"  V0: {v0_released:.2f} GB")
        print(f"  V1: {v1_released:.2f} GB")

        if v1_released > v0_released + 0.5:
            print(f"\n  Conclusion: V1 engine releases MORE memory than V0")
            print(f"  Recommendation: Use V1 engine for better sleep mode support")
        elif v0_released > v1_released + 0.5:
            print(f"\n  Conclusion: V0 engine releases MORE memory than V1")
        else:
            print(f"\n  Conclusion: Both engines release similar amounts of memory")

    print(f"\n{'#'*60}")


if __name__ == "__main__":
    main()
