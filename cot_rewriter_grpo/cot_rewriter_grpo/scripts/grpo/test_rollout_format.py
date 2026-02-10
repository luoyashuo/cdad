"""
Test script to verify if the rollout produces correct format.
Simulates the training rollout process more closely.
"""

import argparse
import os
import sys
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add verl to path
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo/verl')

# Use same data as training
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
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load training data sample
    with open('/data/luoys/data/cdad/cot_rewrite_max_fsf_v3/train.jsonl') as f:
        lines = f.readlines()
        # Use the same sample as in training log (Task B-2.2)
        # Find the sample with "Task B-2.2"
        for line in lines:
            data = json.loads(line)
            if "Task B-2.2" in data.get("question", ""):
                prompt_content = data["problem"]
                print(f"Found sample with: {data['question'][:60]}...")
                break
        else:
            # Use first sample if not found
            data = json.loads(lines[0])
            prompt_content = data["problem"]

    # Apply chat template
    messages = [{"role": "user", "content": prompt_content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get raw prompt token ids (like training does)
    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    print(f"\nPrompt length: {len(raw_prompt_ids)} tokens")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}...")
    print(f"\nPrompt preview (last 500 chars):\n{prompt[-500:]}...")

    # Initialize vLLM with tokenized input (like training does)
    # Use prompt_token_ids instead of string prompt
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Use same engine config as training
        disable_log_stats=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nGenerating {args.num_samples} samples using prompt_token_ids...")
    # Use prompt_token_ids like training does
    outputs = llm.generate([{"prompt_token_ids": raw_prompt_ids}], sampling_params)

    valid_count = 0
    for i, output in enumerate(outputs[0].outputs):
        response = output.text
        is_valid, content = check_format(response)

        print(f"\n{'='*80}")
        print(f"Sample {i+1}: {'✓ VALID FORMAT' if is_valid else '✗ INVALID FORMAT'}")
        print(f"Finish reason: {output.finish_reason}")
        print(f"{'='*80}")

        if is_valid:
            valid_count += 1
            content_preview = content[:500]
            if len(content) > 500:
                content_preview += "..."
            print(f"\nContent preview:\n{content_preview}")
        else:
            response_preview = response[:800]
            if len(response) > 800:
                response_preview += "..."
            print(f"\nFull response (preview):\n{response_preview}")

            if "<rewrite_reasoning>" not in response:
                print("\n⚠ Missing <rewrite_reasoning> tag")
            if "</rewrite_reasoning>" not in response:
                print("\n⚠ Missing </rewrite_reasoning> closing tag")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {valid_count}/{args.num_samples} samples have valid format")
    print(f"{'='*80}")

    if valid_count == 0:
        print("\n⚠ WARNING: No valid format detected!")
        print("This suggests the rollout process has issues.")


if __name__ == "__main__":
    main()
