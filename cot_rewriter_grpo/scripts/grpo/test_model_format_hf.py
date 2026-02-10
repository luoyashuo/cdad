"""
Simple test script using HuggingFace transformers.
Run this when training is stopped.
"""

import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Sample prompt from the training data
SAMPLE_PROMPT = """Task: Your task is to rewrite and improve the reasoning process for a given question.

<question>
Evaluate the line integral \\( \\int_C \\sin(x)\\sin(y)dx - \\cos(x)\\cos(y)dy \\), where \\( C \\) is the line segment from \\((0,-\\pi)\\) to \\(\\left(\\frac{3\\pi}{2},\\frac{\\pi}{2}\\right)\\). The line segment is parameterized by \\( x = \\frac{3\\pi}{2}t \\) and \\( y = \\frac{3\\pi}{2}t - \\pi \\).
</question>

<reference_answer>
To evaluate the line integral \\( \\int_C \\sin(x)\\sin(y)dx - \\cos(x)\\cos(y)dy \\), where \\( C \\) is the line segment from \\((0,-\\pi)\\) to \\(\\left(\\frac{3\\pi}{2},\\frac{\\pi}{2}\\right)\\), we first check if the vector field is conservative. The vector field \\( \\mathbf{F} = (P, Q) \\) with \\( P = \\sin(x)\\sin(y) \\) and \\( Q = -\\cos(x)\\cos(y) \\) is conservative. Thus, the potential function is \\( f(x, y) = -\\cos(x)\\sin(y) \\). The line integral equals 0.
</reference_answer>

<original_reasoning>
I need to evaluate this line integral. The vector field F = (P, Q) where P = sin(x)sin(y) and Q = -cos(x)cos(y). To check if it's conservative, I verify if ∂P/∂y = ∂Q/∂x. ∂P/∂y = sin(x)cos(y) and ∂Q/∂x = sin(x)cos(y). They are equal, so the field is conservative. The potential function f(x, y) = -cos(x)sin(y). Evaluating at endpoints: f(3π/2, π/2) = 0 and f(0, -π) = 0. The integral is 0 - 0 = 0.
</original_reasoning>

### Output Format
Put your rewritten reasoning process inside <rewrite_reasoning> tags. Do not output the final answer again outside the tags, just the reasoning.

<rewrite_reasoning>
[Your rewritten reasoning process here]
</rewrite_reasoning>
"""


def check_format(response: str) -> tuple[bool, str]:
    """Check if response has valid <rewrite_reasoning> tags."""
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
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Apply chat template
    messages = [{"role": "user", "content": SAMPLE_PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")
    print(f"Prompt preview (first 300 chars):\n{prompt[:300]}...\n")

    print(f"Generating {args.num_samples} samples...\n")

    valid_count = 0
    for i in range(args.num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        is_valid, content = check_format(response)

        print(f"\n{'='*80}")
        print(f"Sample {i+1}: {'✓ VALID FORMAT' if is_valid else '✗ INVALID FORMAT'}")
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
            print(f"\nResponse preview:\n{response_preview}")

            if "<rewrite_reasoning>" not in response:
                print("\n⚠ Missing <rewrite_reasoning> tag")
            if "</rewrite_reasoning>" not in response:
                print("\n⚠ Missing </rewrite_reasoning> closing tag")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {valid_count}/{args.num_samples} samples have valid format")
    print(f"{'='*80}")

    if valid_count == 0:
        print("\n⚠ WARNING: No valid format detected!")
        print("\nThis suggests the model may not be able to generate the required format.")
        print("\nRecommendations:")
        print("1. Add few-shot examples with correct format")
        print("2. Add more explicit format instructions")
        print("3. Consider fine-tuning on format examples first")


if __name__ == "__main__":
    main()
