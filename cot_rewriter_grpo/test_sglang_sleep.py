#!/usr/bin/env python3
"""
Test script to verify SGLang sleep/release_memory functionality.

Usage:
    python test_sglang_sleep.py --model /path/to/model

This script:
1. Creates a SGLang Engine instance
2. Checks GPU memory usage
3. Tests memory release methods
4. Checks if GPU memory is released
"""

import argparse
import gc
import time
import torch


def get_gpu_memory_info(device_id=0):
    """Get GPU memory info in GB."""
    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    return free / 1024**3, total / 1024**3


def print_gpu_memory(label, device_id=0):
    """Print GPU memory usage."""
    free, total = get_gpu_memory_info(device_id)
    if free is not None:
        used = total - free
        print(f"[{label}] GPU {device_id}: {free:.2f}GB free / {total:.2f}GB total (used: {used:.2f}GB)")
    return free


def main():
    parser = argparse.ArgumentParser(description="Test SGLang memory release")
    parser.add_argument("--model", type=str, default="/fs/fast/u20240075/hf_home/models/Qwen/Qwen3-4B",
                        help="Path to the model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--mem-fraction", type=float, default=0.7,
                        help="GPU memory fraction for SGLang")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("SGLang Memory Release Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu}")
    print(f"Memory Fraction: {args.mem_fraction}")
    print()

    # Check initial memory
    free_initial = print_gpu_memory("Initial", 0)

    # Import SGLang
    print("\n[1] Importing SGLang...")
    try:
        import sglang as sgl
        print(f"    SGLang version: {sgl.__version__}")
    except ImportError as e:
        print(f"    Failed to import sglang: {e}")
        print("    Install with: pip install sglang[all]")
        return

    # Create Engine instance
    print("\n[2] Creating SGLang Engine...")
    start_time = time.time()

    try:
        # SGLang Engine API
        engine = sgl.Engine(
            model_path=args.model,
            tp_size=1,
            mem_fraction_static=args.mem_fraction,
            dtype="float16",
            trust_remote_code=True,
        )
        print(f"    Engine created in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"    Failed to create engine: {e}")
        import traceback
        traceback.print_exc()
        return

    free_after_load = print_gpu_memory("After Engine load", 0)
    memory_used_by_model = free_initial - free_after_load
    print(f"    Memory used by model: {memory_used_by_model:.2f}GB")

    # Test SGLang's actual memory release methods
    print("\n[3] Testing memory release methods...")

    # Method 1: flush_cache
    print("\n    [3a] Trying engine.flush_cache()...")
    try:
        if hasattr(engine, 'flush_cache'):
            engine.flush_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            free_after_flush = print_gpu_memory("After flush_cache()", 0)
            released = free_after_flush - free_after_load
            print(f"        Released by flush_cache(): {released:.2f}GB")
        else:
            print("        flush_cache() method not available")
            free_after_flush = free_after_load
    except Exception as e:
        print(f"        flush_cache() failed: {e}")
        import traceback
        traceback.print_exc()
        free_after_flush = free_after_load

    # Method 2: release_memory_occupation (SGLang's actual API)
    print("\n    [3b] Trying engine.release_memory_occupation()...")
    try:
        if hasattr(engine, 'release_memory_occupation'):
            engine.release_memory_occupation()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            free_after_release = print_gpu_memory("After release_memory_occupation()", 0)
            released = free_after_release - free_after_load
            print(f"        Released by release_memory_occupation(): {released:.2f}GB")
        else:
            print("        release_memory_occupation() method not available")
            free_after_release = free_after_load
    except Exception as e:
        print(f"        release_memory_occupation() failed: {e}")
        import traceback
        traceback.print_exc()
        free_after_release = free_after_load

    # Method 3: resume_memory_occupation (restore after release)
    print("\n    [3c] Trying engine.resume_memory_occupation()...")
    try:
        if hasattr(engine, 'resume_memory_occupation'):
            engine.resume_memory_occupation()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            free_after_resume = print_gpu_memory("After resume_memory_occupation()", 0)
            reoccupied = free_after_release - free_after_resume
            print(f"        Re-occupied by resume_memory_occupation(): {reoccupied:.2f}GB")
        else:
            print("        resume_memory_occupation() method not available")
    except Exception as e:
        print(f"        resume_memory_occupation() failed: {e}")
        import traceback
        traceback.print_exc()

    free_after_offload = free_after_release  # For summary compatibility

    # List all available methods
    print("\n[4] Available engine methods (memory related):")
    for attr in dir(engine):
        if not attr.startswith('_'):
            lower = attr.lower()
            if any(kw in lower for kw in ['sleep', 'wake', 'release', 'offload', 'memory', 'free', 'cache']):
                print(f"    - {attr}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial free memory:        {free_initial:.2f}GB")
    print(f"After model load:           {free_after_load:.2f}GB (used {memory_used_by_model:.2f}GB)")
    print(f"After flush_cache():        {free_after_flush:.2f}GB (released {free_after_flush - free_after_load:.2f}GB)")
    print(f"After release_memory_occupation(): {free_after_release:.2f}GB (released {free_after_release - free_after_load:.2f}GB)")
    print()

    best_free = max(free_after_flush, free_after_release)
    if best_free > free_initial - 2:
        print("RESULT: Memory release WORKS correctly!")
    else:
        print("RESULT: Memory release does NOT work properly!")
        print(f"        Expected ~{free_initial:.2f}GB free, got {best_free:.2f}GB free")

    # Cleanup
    print("\n[5] Cleaning up...")
    try:
        engine.shutdown()
    except:
        pass
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("After cleanup", 0)


if __name__ == "__main__":
    main()
