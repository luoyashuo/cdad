#!/usr/bin/env python3
"""
简化版测试：只测试 vLLM 直接调用 vs prompt_token_ids
"""

import argparse
import os
import sys
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def check_format(response: str) -> tuple[bool, str]:
    """Check if response has valid <rewrite_reasoning> tags."""
    import re
    pattern = r'<rewrite_reasoning>\s*(.*?)\s*</rewrite_reasoning>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True, match.group(1).strip()
    return False, response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load sample data
    with open('/data/luoys/data/cdad/cot_rewrite_max_fsf_v3/train.jsonl') as f:
        for line in f:
            data = json.loads(line)
            if "Task B-2.2" in data.get("question", ""):
                prompt_content = data["problem"]
                break
        else:
            with open('/data/luoys/data/cdad/cot_rewrite_max_fsf_v3/train.jsonl') as f:
                data = json.loads(f.readline())
                prompt_content = data["problem"]

    messages = [{"role": "user", "content": prompt_content}]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"Prompt length: {len(tokenizer.encode(prompt_str))} tokens")

    # TEST 1: Direct vLLM with string prompt
    print("\n" + "="*60)
    print("TEST 1: vLLM with string prompt")
    print("="*60)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=0.4,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=10000,
    )

    outputs = llm.generate([prompt_str], sampling_params)
    valid1 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid1}/{args.num_samples}")
    for i, out in enumerate(outputs[0].outputs):
        is_valid, _ = check_format(out.text)
        print(f"  Sample {i+1}: {'✓' if is_valid else '✗'}")

    # TEST 2: vLLM with prompt_token_ids
    print("\n" + "="*60)
    print("TEST 2: vLLM with prompt_token_ids")
    print("="*60)

    raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    outputs = llm.generate([{"prompt_token_ids": raw_prompt_ids}], sampling_params)
    valid2 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid2}/{args.num_samples}")
    for i, out in enumerate(outputs[0].outputs):
        is_valid, _ = check_format(out.text)
        print(f"  Sample {i+1}: {'✓' if is_valid else '✗'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"String prompt:     {valid1}/{args.num_samples} ({100*valid1/args.num_samples:.0f}%)")
    print(f"Prompt token IDs:  {valid2}/{args.num_samples} ({100*valid2/args.num_samples:.0f}%)")
    print("="*60)

    if valid1 != valid2:
        print("⚠ Different results - prompt_token_ids has issues!")
    else:
        print("✓ Results are consistent")

if __name__ == "__main__":
    main()
