#!/usr/bin/env python3
"""
FSF Judge Evaluation Pipeline

This script combines inference and PR curve analysis into a single pipeline.

Usage:
    python scripts/fsf_judge_evaluate.py \
        --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
        --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
        --max_tokens 5 \
        --batch_size 32

Optional arguments:
    --skip_inference: Skip inference if results already exist
    --skip_analysis: Skip PR curve analysis
    --tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="FSF Judge Evaluation Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained FSF Judge model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the validation dataset (jsonl file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If None, will use {model_path}/fsf_judge_val",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism size for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model inference",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference if results already exist",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip PR curve analysis",
    )
    parser.add_argument(
        "--num_thresholds",
        type=int,
        default=100,
        help="Number of threshold points for PR curve",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process. Process all if not set.",
    )
    return parser.parse_args()


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n{description} completed successfully!")


def main():
    args = parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "fsf_judge_val")
    
    inference_results_path = os.path.join(args.output_dir, "inference_results.jsonl")
    
    # Step 1: Inference
    if not args.skip_inference:
        inference_cmd = [
            sys.executable,
            "fsf_judge/fsf_judge_inference.py",
            "--model_path", args.model_path,
            "--data_path", args.data_path,
            "--output_dir", args.output_dir,
            "--max_tokens", str(args.max_tokens),
            "--batch_size", str(args.batch_size),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--dtype", args.dtype,
        ]
        if args.resume:
            inference_cmd.append("--resume")
        if args.max_samples is not None:
            inference_cmd.extend(["--max_samples", str(args.max_samples)])
        run_command(inference_cmd, "Inference")
    else:
        print(f"\nSkipping inference (using existing results from: {inference_results_path})")
        if not os.path.exists(inference_results_path):
            print(f"Error: Inference results not found at {inference_results_path}")
            sys.exit(1)
    
    # Step 2: PR Curve Analysis
    if not args.skip_analysis:
        analysis_cmd = [
            sys.executable,
            "fsf_judge/fsf_judge_analyze_pr.py",
            "--input_path", inference_results_path,
            "--output_dir", args.output_dir,
            "--num_thresholds", str(args.num_thresholds),
        ]
        run_command(analysis_cmd, "PR Curve Analysis")
    else:
        print("\nSkipping PR curve analysis")
    
    print("\n" + "="*70)
    print("Evaluation Pipeline Completed Successfully!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Inference results: {inference_results_path}")
    print(f"  - Metrics: {os.path.join(args.output_dir, 'metrics.json')}")
    print(f"  - PR curve data: {os.path.join(args.output_dir, 'pr_curve_data.csv')}")
    print(f"  - Precision-Recall-Threshold: {os.path.join(args.output_dir, 'precision_recall_threshold.csv')}")
    print(f"  - PR curve plot: {os.path.join(args.output_dir, 'pr_curve.png')}")
    print(f"  - Score distribution plot: {os.path.join(args.output_dir, 'score_distribution.png')}")


if __name__ == "__main__":
    main()
