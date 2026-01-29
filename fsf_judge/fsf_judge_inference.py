#!/usr/bin/env python3
"""
FSF Judge Inference Script using vLLM

This script loads a trained FSF Judge model using vLLM and runs inference
on a validation dataset. It records the softmax probabilities for the first
token (N/Y) as the probability of containing "failed branches".

Usage:
    python scripts/fsf_judge_inference.py \
        --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
        --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
        --output_dir /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val \
        --max_tokens 5 \
        --batch_size 32
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Disable vLLM internal tqdm progress bars
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'

from tqdm import tqdm
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="FSF Judge Inference with vLLM")
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
        help="Output directory for inference results. If None, will use {model_path}/fsf_judge_val",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (also used as save interval for checkpoint)",
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
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for models like Qwen3 (default: disabled)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process. Process all if not set.",
    )
    return parser.parse_args()


def load_dataset(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load dataset from jsonl file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                if max_samples is not None and len(data) >= max_samples:
                    break
    return data


def prepare_messages(data_item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract system and user messages from data item (exclude assistant/label for inference)."""
    conversations = data_item.get("conversations", [])
    messages = []
    for conv in conversations:
        role = conv.get("role", conv.get("from", "user"))
        content = conv.get("content", conv.get("value", ""))
        # Include system and user messages, exclude assistant (which is the label)
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role in ["user"]:
            messages.append({"role": "user", "content": content})
    return messages


def get_label_content(data_item: Dict[str, Any]) -> str:
    """Extract assistant response (label) from data item."""
    conversations = data_item.get("conversations", [])
    for conv in conversations:
        role = conv.get("role", conv.get("from", ""))
        if role in ["assistant", "gpt", "model"]:
            return conv.get("content", conv.get("value", ""))
    return None


def get_target_label(data_item: Dict[str, Any]) -> str:
    """Extract target label (Y/N) from data item."""
    # First, try to get label from the 'label' field
    label = data_item.get("label")
    if label and isinstance(label, str):
        first_char = label.strip()[0].upper()
        if first_char in ["Y", "N"]:
            return first_char
    
    # Fallback: extract from assistant's response in conversations
    conversations = data_item.get("conversations", [])
    for conv in conversations:
        role = conv.get("role", conv.get("from", ""))
        if role in ["assistant", "gpt", "model"]:
            content = conv.get("content", conv.get("value", ""))
            if content:
                first_char = content.strip()[0].upper()
                if first_char in ["Y", "N"]:
                    return first_char
    return None


def get_existing_results_count(output_path: str) -> int:
    """Count how many results already exist in the output file."""
    if not os.path.exists(output_path):
        return 0
    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def append_results(results: List[Dict[str, Any]], output_path: str):
    """Append results to jsonl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_inference_vllm(
    model_path: str,
    data: List[Dict[str, Any]],
    max_tokens: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    output_path: str,
    start_idx: int = 0,
) -> List[Dict[str, Any]]:
    """Run inference using vLLM and return results with softmax probabilities."""
    
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: vllm is not installed. Please install it first:")
        print("  pip install vllm")
        sys.exit(1)
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    print(f"Loading model from: {model_path}")
    print(f"  - dtype: {dtype}")
    print(f"  - tensor_parallel_size: {tensor_parallel_size}")
    print(f"  - gpu_memory_utilization: {gpu_memory_utilization}")
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model with vLLM (disable tqdm progress bars)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch_dtype,
        trust_remote_code=True,
        enable_lora=False,
        disable_log_stats=True,
    )
    
    # Set up sampling params - we need logits so set temperature to 0
    sampling_params_kwargs = {
        "temperature": 0.0,  # Greedy decoding for deterministic results
        "max_tokens": max_tokens,
        "logprobs": 20,  # Get top 20 logprobs to ensure Y/N are included
        "prompt_logprobs": None,
    }
    
    # Check if SamplingParams supports enable_thinking (for Qwen3 models)
    import inspect
    if "enable_thinking" in inspect.signature(SamplingParams).parameters:
        sampling_params_kwargs["enable_thinking"] = False
        print("  - SamplingParams thinking mode: disabled")
    else:
        print("  - SamplingParams does not support enable_thinking")
    
    sampling_params = SamplingParams(**sampling_params_kwargs)
    
    results = []
    total_samples = len(data)
    
    # Process data one by one (vLLM batch_size=1) with progress bar showing total
    with tqdm(total=total_samples, desc="Running inference", initial=start_idx) as pbar:
        for i, item in enumerate(data):
            # Prepare prompt with thinking disabled for Qwen3
            # Note: prepare_messages now returns system and user messages, assistant is extracted as label
            messages = prepare_messages(item)
            label_content = get_label_content(item)
            apply_chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "tokenizer_kwargs": {},
            }
            # Check if tokenizer supports enable_thinking (for Qwen3 models)
            # For Qwen3, enable_thinking should be passed via tokenizer_kwargs
            try:
                # Test if tokenizer supports enable_thinking
                test_output = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "test"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    tokenizer_kwargs={"enable_thinking": False}
                )
                apply_chat_template_kwargs["tokenizer_kwargs"] = {"enable_thinking": False}
                if i == 0:  # Only print once
                    print("  - Tokenizer enable_thinking: False (via tokenizer_kwargs)")
            except Exception:
                if i == 0:
                    print("  - Tokenizer does not support enable_thinking")
            
            prompt = tokenizer.apply_chat_template(
                messages,
                **apply_chat_template_kwargs
            )
            
            # Run inference with batch_size=1
            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0]
            
            # Get generated text and token information
            generated_text = output.outputs[0].text.strip()
            token_ids = output.outputs[0].token_ids
            
            # Get token IDs for N, Y, and think tags
            n_token_id = tokenizer.encode("N", add_special_tokens=False)
            y_token_id = tokenizer.encode("Y", add_special_tokens=False)
            think_start_id = tokenizer.encode("<think>", add_special_tokens=False)
            think_end_id = tokenizer.encode("</think>", add_special_tokens=False)
            newline_id = tokenizer.encode("\n", add_special_tokens=False)
            
            if isinstance(n_token_id, list) and len(n_token_id) > 0:
                n_token_id = n_token_id[0]
            if isinstance(y_token_id, list) and len(y_token_id) > 0:
                y_token_id = y_token_id[0]
            if isinstance(think_start_id, list) and len(think_start_id) > 0:
                think_start_id = think_start_id[0]
            if isinstance(think_end_id, list) and len(think_end_id) > 0:
                think_end_id = think_end_id[0]
            if isinstance(newline_id, list) and len(newline_id) > 0:
                newline_id = newline_id[0]
            
            # Extract target label from ground truth
            target_label = get_target_label(item)
            
            # Build detailed token information for all generated tokens
            all_tokens_info = []
            target_token_idx = None  # Index of the first Y/N token
            target_token_logprobs = None
            target_token_id = None
            is_anomaly = True  # Default to anomaly until we find Y or N
            
            if token_ids and output.outputs[0].logprobs:
                for token_idx, (token_id, token_logprob_dict) in enumerate(zip(token_ids, output.outputs[0].logprobs)):
                    # Decode the token
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                    
                    # Check if this is Y or N, or think tags
                    is_yn_token = (token_id == n_token_id or token_id == y_token_id)
                    is_think_start = (token_id == think_start_id)
                    is_think_end = (token_id == think_end_id)
                    is_newline = (token_id == newline_id)
                    
                    if is_yn_token and target_token_idx is None:
                        target_token_idx = token_idx
                        target_token_logprobs = token_logprob_dict
                        target_token_id = token_id
                        is_anomaly = False  # Found Y or N, not anomaly
                    
                    # Get logprobs for Y and N from top-20 (default to 0.0 if not found)
                    n_logprob = 0.0
                    y_logprob = 0.0
                    for tid, logprob_obj in token_logprob_dict.items():
                        if tid == n_token_id:
                            n_logprob = logprob_obj.logprob
                        elif tid == y_token_id:
                            y_logprob = logprob_obj.logprob
                    
                    token_info = {
                        "token_idx": token_idx,
                        "token_id": int(token_id),
                        "token_text": token_text,
                        "is_yn_token": is_yn_token,
                        "is_think_start": is_think_start,
                        "is_think_end": is_think_end,
                        "is_newline": is_newline,
                        "logprob_N": float(n_logprob),
                        "logprob_Y": float(y_logprob),
                    }
                    all_tokens_info.append(token_info)
            
            # Calculate final probabilities and label based on the first Y/N token
            # Get raw logprobs for Y and N from the target token (default to 0.0 if not found)
            n_logprob = 0.0
            y_logprob = 0.0
            n_prob = 0.0
            y_prob = 0.0
            failed_branch_prob = 0.0
            
            if target_token_logprobs is not None and target_token_id is not None:
                for tid, logprob_obj in target_token_logprobs.items():
                    if tid == n_token_id:
                        n_logprob = logprob_obj.logprob
                    elif tid == y_token_id:
                        y_logprob = logprob_obj.logprob
                
                # Use exp(logprob_Y) directly as failed_branch_prob (not softmax)
                if y_logprob != 0.0:
                    failed_branch_prob = float(np.exp(y_logprob))
                else:
                    failed_branch_prob = 0.0
                
                if target_token_id == n_token_id:
                    predicted_label = "N"
                    actual_predicted_token = "N"
                elif target_token_id == y_token_id:
                    predicted_label = "Y"
                    actual_predicted_token = "Y"
            else:
                # Fallback: use generated text to find first Y or N
                if generated_text:
                    for char in generated_text:
                        if char.upper() == 'Y':
                            predicted_label = 'Y'
                            break
                        elif char.upper() == 'N':
                            predicted_label = 'N'
                            break
            
            result = {
                "index": start_idx + i,
                "messages": messages,  # User messages only (input to model)
                "label_content": label_content,  # Assistant response (ground truth)
                "target_label": target_label,  # Y/N extracted from label
                "generated_text": generated_text,
                "predicted_label": predicted_label,
                "logprob_N": float(n_logprob),
                "logprob_Y": float(y_logprob),
                "prob_N": float(n_prob),
                "prob_Y": float(y_prob),
                "failed_branch_prob": failed_branch_prob,
                "is_anomaly": is_anomaly,
                "first_yn_token_idx": target_token_idx,
                "actual_predicted_token": actual_predicted_token,
                "all_tokens": all_tokens_info,  # Detailed info for each token
            }
            results.append(result)
            pbar.update(1)
            
            # Save checkpoint every batch_size samples
            if (i + 1) % batch_size == 0 or (i + 1) == total_samples:
                append_results(results, output_path)
                print(f"  Checkpoint saved: {start_idx + i + 1}/{start_idx + total_samples} samples processed")
                results = []  # Clear results after saving
    
    return results


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to jsonl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to: {output_path}")


def compute_metrics_from_file(output_path: str) -> Dict[str, Any]:
    """Compute basic metrics from existing results file."""
    total = 0
    correct = 0
    has_prob = 0
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                total += 1
                if r["target_label"] and r["predicted_label"]:
                    if r["target_label"] == r["predicted_label"]:
                        correct += 1
                if r["failed_branch_prob"] is not None:
                    has_prob += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "samples_with_probability": has_prob,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute basic metrics."""
    total = len(results)
    correct = 0
    has_prob = 0
    
    for r in results:
        if r["target_label"] and r["predicted_label"]:
            if r["target_label"] == r["predicted_label"]:
                correct += 1
        if r["failed_branch_prob"] is not None:
            has_prob += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "samples_with_probability": has_prob,
    }


def main():
    args = parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "fsf_judge_val")
    
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    data = load_dataset(args.data_path, max_samples=args.max_samples)
    print(f"Loaded {len(data)} samples")
    if args.max_samples is not None:
        print(f"  - max_samples limit: {args.max_samples}")
    
    output_path = os.path.join(args.output_dir, "inference_results.jsonl")
    
    # Check for existing results if resume is enabled
    start_idx = 0
    if args.resume and os.path.exists(output_path):
        start_idx = get_existing_results_count(output_path)
        if start_idx > 0:
            print(f"Resuming from checkpoint: {start_idx} samples already processed")
            if start_idx >= len(data):
                print("All samples already processed. Skipping inference.")
                # Compute and display metrics from existing results
                metrics = compute_metrics_from_file(output_path)
                print(f"\nMetrics (from existing results):")
                print(f"  - Total samples: {metrics['total_samples']}")
                print(f"  - Correct predictions: {metrics['correct_predictions']}")
                print(f"  - Accuracy: {metrics['accuracy']:.4f}")
                print(f"  - Samples with probability: {metrics['samples_with_probability']}")
                
                # Save metrics
                metrics_path = os.path.join(args.output_dir, "metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Metrics saved to: {metrics_path}")
                return
            data = data[start_idx:]
            print(f"Remaining samples to process: {len(data)}")
    
    # Run inference
    print(f"\nRunning inference with vLLM...")
    print(f"  - Save interval (batch_size): {args.batch_size}")
    print(f"  - vLLM inference batch_size: 1")
    run_inference_vllm(
        model_path=args.model_path,
        data=data,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        output_path=output_path,
        start_idx=start_idx,
    )
    
    # Compute metrics from file (to include all results)
    metrics = compute_metrics_from_file(output_path)
    print(f"\nMetrics:")
    print(f"  - Total samples: {metrics['total_samples']}")
    print(f"  - Correct predictions: {metrics['correct_predictions']}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Samples with probability: {metrics['samples_with_probability']}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\nInference completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
