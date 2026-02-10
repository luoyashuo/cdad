#!/usr/bin/env python3
"""
测试padding和position_ids对格式正确率的影响
"""

import argparse
import os
import sys
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def check_format(response: str) -> tuple[bool, str]:
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

    # Get raw token ids
    raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    print(f"Original prompt length: {len(raw_prompt_ids)} tokens")
    print(f"Pad token ID: {tokenizer.pad_token_id}")

    # 模拟训练时的padding处理
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    max_length = 5500  # 训练时的max_prompt_length

    # Left padding like training does
    pad_length = max_length - len(raw_prompt_ids)
    padded_ids = torch.cat([
        torch.full((pad_length,), pad_token_id, dtype=torch.long),
        torch.tensor(raw_prompt_ids, dtype=torch.long)
    ])

    print(f"Padded prompt length: {len(padded_ids)} tokens")
    print(f"Pad tokens at start: {padded_ids[:5].tolist()}")
    print(f"First 5 actual tokens: {padded_ids[pad_length:pad_length+5].tolist()}")

    # 检查是否有BOS/EOS token
    print(f"\nBOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"First token is BOS: {padded_ids[pad_length].item() == tokenizer.bos_token_id}")

    # TEST 1: 原始prompt (字符串)
    print("\n" + "="*60)
    print("TEST 1: String prompt (baseline)")
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
        top_p=0.95,
        max_tokens=10000,
    )

    outputs = llm.generate([prompt_str], sampling_params)
    valid1 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid1}/{args.num_samples}")

    # TEST 2: 原始token ids (无padding)
    print("\n" + "="*60)
    print("TEST 2: Raw token IDs (no padding)")
    print("="*60)

    outputs = llm.generate([{"prompt_token_ids": raw_prompt_ids}], sampling_params)
    valid2 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid2}/{args.num_samples}")

    # TEST 3: 左padding后的token ids
    print("\n" + "="*60)
    print("TEST 3: Left-padded token IDs (like training)")
    print("="*60)

    outputs = llm.generate([{"prompt_token_ids": padded_ids.tolist()}], sampling_params)
    valid3 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid3}/{args.num_samples}")

    # TEST 4: 检查vLLM是否自动处理padding
    print("\n" + "="*60)
    print("TEST 4: Token IDs with explicit attention mask")
    print("="*60)

    # 创建attention mask，padding部分为0
    attention_mask = torch.cat([
        torch.zeros(pad_length, dtype=torch.long),
        torch.ones(len(raw_prompt_ids), dtype=torch.long)
    ])

    outputs = llm.generate([{
        "prompt_token_ids": padded_ids.tolist(),
        "attention_mask": attention_mask.tolist()
    }], sampling_params)
    valid4 = sum(1 for out in outputs[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid4}/{args.num_samples}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TEST 1 - String prompt:         {valid1}/{args.num_samples} ({100*valid1/args.num_samples:.0f}%)")
    print(f"TEST 2 - Raw token IDs:         {valid2}/{args.num_samples} ({100*valid2/args.num_samples:.0f}%)")
    print(f"TEST 3 - Left-padded token IDs: {valid3}/{args.num_samples} ({100*valid3/args.num_samples:.0f}%)")
    print(f"TEST 4 - With attention mask:   {valid4}/{args.num_samples} ({100*valid4/args.num_samples:.0f}%)")
    print("="*60)

    # 分析
    if valid2 < valid1:
        print("⚠ Token IDs (no padding) 比字符串prompt效果差")
    if valid3 < valid2:
        print("⚠ Left padding 进一步降低格式正确率")
    if valid3 < valid1:
        print("⚠ Padding是问题的根源！")

if __name__ == "__main__":
    import torch
    main()
