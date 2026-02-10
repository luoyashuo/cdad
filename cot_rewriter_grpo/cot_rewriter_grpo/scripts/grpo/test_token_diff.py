#!/usr/bin/env python3
"""
检查字符串prompt和token ids在vLLM中的处理差异
"""

import argparse
import json
from transformers import AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B")
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

    # Decode back to check consistency
    decoded_back = tokenizer.decode(raw_prompt_ids, skip_special_tokens=False)

    print("="*60)
    print("PROMPT ANALYSIS")
    print("="*60)
    print(f"Original prompt length: {len(prompt_str)} chars")
    print(f"Token IDs length: {len(raw_prompt_ids)} tokens")
    print(f"Decoded back length: {len(decoded_back)} chars")
    print()

    # Check if decoded matches original
    if prompt_str == decoded_back:
        print("✓ Decoded text matches original")
    else:
        print("✗ Decoded text DIFFERS from original!")
        print()
        # Find first difference
        for i, (c1, c2) in enumerate(zip(prompt_str, decoded_back)):
            if c1 != c2:
                print(f"First difference at position {i}:")
                print(f"  Original: '{prompt_str[max(0,i-20):i+50]}'")
                print(f"  Decoded:  '{decoded_back[max(0,i-20):i+50]}'")
                break

    print()
    print("="*60)
    print("TOKEN ANALYSIS")
    print("="*60)

    # Check first and last tokens
    print(f"First 10 token IDs: {raw_prompt_ids[:10]}")
    print(f"Last 10 token IDs: {raw_prompt_ids[-10:]}")
    print()

    # Decode first and last few tokens
    first_tokens = tokenizer.decode(raw_prompt_ids[:20], skip_special_tokens=False)
    last_tokens = tokenizer.decode(raw_prompt_ids[-20:], skip_special_tokens=False)

    print(f"First 20 tokens decoded: {first_tokens[:200]}...")
    print()
    print(f"Last 20 tokens decoded: ...{last_tokens[-200:]}")
    print()

    # Check for special tokens
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print()

    bos_in_prompt = tokenizer.bos_token_id in raw_prompt_ids
    eos_in_prompt = tokenizer.eos_token_id in raw_prompt_ids
    print(f"BOS in prompt tokens: {bos_in_prompt}")
    print(f"EOS in prompt tokens: {eos_in_prompt}")

    # Check chat template
    print()
    print("="*60)
    print("CHAT TEMPLATE")
    print("="*60)
    print(tokenizer.chat_template[:500] if tokenizer.chat_template else "None")
    print()

    # Check how vLLM might tokenize differently
    print("="*60)
    print("VLLM TOKENIZATION HYPOTHESIS")
    print("="*60)
    print("When using string prompt, vLLM will:")
    print("1. Apply its own tokenizer")
    print("2. Add special tokens if configured")
    print()
    print("When using token_ids, vLLM will:")
    print("1. Use the tokens directly")
    print("2. NOT add special tokens")
    print()
    print("Potential issues:")
    print("- If vLLM's tokenizer has different special token handling")
    print("- If vLLM expects BOS/EOS at specific positions")
    print("- If position_ids are misaligned")

if __name__ == "__main__":
    main()
