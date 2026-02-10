"""
Compare different prompt methods for format accuracy.
"""

import argparse
import os
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

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
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load training data sample
    with open('/data/luoys/data/cdad/cot_rewrite_max_fsf_v3/train.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if "Task B-2.2" in data.get("question", ""):
                prompt_content = data["problem"]
                print(f"Found sample with: {data['question'][:60]}...")
                break
        else:
            data = json.loads(lines[0])
            prompt_content = data["problem"]

    # Apply chat template
    messages = [{"role": "user", "content": prompt_content}]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get raw prompt token ids
    raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    print(f"\nPrompt length: {len(raw_prompt_ids)} tokens")

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=0.4,
        disable_log_stats=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=10000,
    )

    print(f"\n{'='*80}")
    print(f"METHOD 1: Using string prompt")
    print(f"{'='*80}")
    outputs_str = llm.generate([prompt_str], sampling_params)
    valid_str = sum(1 for out in outputs_str[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid_str}/{args.num_samples}")

    print(f"\n{'='*80}")
    print(f"METHOD 2: Using prompt_token_ids")
    print(f"{'='*80}")
    outputs_ids = llm.generate([{"prompt_token_ids": raw_prompt_ids}], sampling_params)
    valid_ids = sum(1 for out in outputs_ids[0].outputs if check_format(out.text)[0])
    print(f"Valid format: {valid_ids}/{args.num_samples}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"String prompt:     {valid_str}/{args.num_samples} ({100*valid_str/args.num_samples:.0f}%)")
    print(f"Prompt token IDs:  {valid_ids}/{args.num_samples} ({100*valid_ids/args.num_samples:.0f}%)")

    # Check if methods produce same results
    if valid_str != valid_ids:
        print(f"\n⚠ WARNING: Different format accuracy between methods!")


if __name__ == "__main__":
    main()
