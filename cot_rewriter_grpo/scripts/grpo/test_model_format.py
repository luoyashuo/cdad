"""
Test script to verify if the model can output the correct format.
Uses vLLM for inference.
"""

import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Sample prompt from the training data
SAMPLE_PROMPT = """Task: Your task is to rewrite and improve the reasoning process for a given question.

<question>
Evaluate the line integral \\( \\int_C \\sin(x)\\sin(y)dx - \\cos(x)\\cos(y)dy \\), where \\( C \\) is the line segment from \\((0,-\\pi)\\) to \\(\\left(\\frac{3\\pi}{2},\\frac{\\pi}{2}\\right)\\). The line segment is parameterized by \\( x = \\frac{3\\pi}{2}t \\) and \\( y = \\frac{3\\pi}{2}t - \\pi \\).
</question>

<reference_answer>
To evaluate the line integral \\( \\int_C \\sin(x)\\sin(y)dx - \\cos(x)\\cos(y)dy \\), where \\( C \\) is the line segment from \\((0,-\\pi)\\) to \\(\\left(\\frac{3\\pi}{2},\\frac{\\pi}{2}\\right)\\), we first check if the vector field is conservative.

The vector field \\( \\mathbf{F} = (P, Q) \\) with \\( P = \\sin(x)\\sin(y) \\) and \\( Q = -\\cos(x)\\cos(y) \\) is conservative because the partial derivatives satisfy:
\\[
\\frac{\\partial P}{\\partial y} = \\sin(x)\\cos(y) = \\frac{\\partial Q}{\\partial x}
\\]

Thus, there exists a potential function \\( f \\) such that \\( \\nabla f = \\mathbf{F} \\). We find \\( f \\) by integrating \\( P \\) with respect to \\( x \\):
\\[
f(x, y) = \\int \\sin(x)\\sin(y) \\, dx = -\\cos(x)\\sin(y) + C(y)
\\]

Next, we differentiate \\( f \\) with respect to \\( y \\) and compare with \\( Q \\):
\\[
\\frac{\\partial f}{\\partial y} = -\\cos(x)\\cos(y) + C'(y) = -\\cos(x)\\cos(y) \\implies C'(y) = 0 \\implies C(y) \\text{ is a constant}
\\]

Thus, the potential function is \\( f(x, y) = -\\cos(x)\\sin(y) \\). Evaluating \\( f \\) at the endpoints \\((0, -\\pi)\\) and \\(\\left(\\frac{3\\pi}{2}, \\frac{\\pi}{2}\\right)\\):
\\[
f\\left(\\frac{3\\pi}{2},\\frac{\\pi}{2}\\right) = -\\cos\\left(\\frac{3\\pi}{2}\\right)\\sin\\left(\\frac{\\pi}{2}\\right) = -0 \\cdot 1 = 0
\\]
\\[
f(0, -\\pi) = -\\cos(0)\\sin(-\\pi) = -1 \\cdot 0 = 0
\\]

The line integral is the difference of the potential function at the endpoints:
\\[
f\\left(\\frac{3\\pi}{2}, \\frac{\\pi}{2}\\right) - f(0, -\\pi) = 0 - 0 = 0
\\]

To confirm, we parameterize the curve \\( C \\) as \\( x = \\frac{3\\pi}{2}t \\) and \\( y = \\frac{3\\pi}{2}t - \\pi \\) for \\( t \\in [0, 1] \\). Substituting into the integral and simplifying using trigonometric identities also results in 0. Both methods confirm the integral is zero.

Thus, the final answer is:
\\[
\\boxed{0}
\\]
</reference_answer>

<original_reasoning>
Alright, so I need to evaluate this line integral: \\( \\int_C \\sin(x)\\sin(y)dx - \\cos(x)\\cos(y)dy \\), where C is the line segment from (0, -π) to (3π/2, π/2). The parameterization is given as x = (3π/2)t and y = (3π/2)t - π. Hmm, okay. Let me try to work through this step by step.

First, I remember that for line integrals, if the vector field is conservative, the integral can be path-independent and potentially easier to compute by finding a potential function. But if it's not conservative, then I need to parameterize the curve and compute the integral directly. Let me check if the vector field here is conservative.

The vector field given is F = (P, Q) where P = sin(x)sin(y) and Q = -cos(x)cos(y). To check if it's conservative, I need to verify if the partial derivative of P with respect to y is equal to the partial derivative of Q with respect to x.

Compute ∂P/∂y: derivative of sin(x)sin(y) with respect to y is sin(x)cos(y).

Compute ∂Q/∂x: derivative of -cos(x)cos(y) with respect to x is sin(x)cos(y).

Wait, so ∂P/∂y = sin(x)cos(y) and ∂Q/∂x = sin(x)cos(y). So they are equal. Therefore, the vector field is conservative! That means there exists a potential function f such that ∇f = F. Therefore, the line integral is path-independent and can be evaluated as f(endpoint) - f(startpoint).

So instead of parameterizing and integrating, maybe I can find the potential function f and subtract its values at the endpoints. Let me try that.

To find f, we need to integrate P with respect to x and Q with respect to y, ensuring consistency.

First, integrate P with respect to x:

f(x, y) = ∫ sin(x)sin(y) dx + C(y) = -cos(x)sin(y) + C(y).

Now, take the partial derivative of this f with respect to y and set it equal to Q:

∂f/∂y = -cos(x)cos(y) + C'(y) = Q = -cos(x)cos(y).

Therefore, comparing both sides, C'(y) = 0, so C(y) is a constant. Therefore, the potential function is f(x, y) = -cos(x)sin(y) + C. Since constants don't affect the difference, we can take C = 0 for simplicity.

Therefore, f(x, y) = -cos(x)sin(y). Let me check if that works.

Compute ∇f = (sin(x)sin(y), -cos(x)cos(y)), which matches the given F. Perfect.

Therefore, the line integral is f(3π/2, π/2) - f(0, -π).

Compute f(3π/2, π/2):

First, cos(3π/2) is 0, since cos(3π/2) is the same as cos(270 degrees) which is 0. Then sin(π/2) is 1. So f(3π/2, π/2) = -0 * 1 = 0.

Then f(0, -π):

cos(0) = 1, sin(-π) = 0. So f(0, -π) = -1 * 0 = 0.

Wait, so the integral is 0 - 0 = 0? That seems... surprising. Let me double-check.

Wait, cos(3π/2) is indeed 0. sin(π/2) is 1. So 0*1=0. For the other point, cos(0)=1, sin(-π)=0. So 1*0=0. So yes, 0 - 0 = 0.

But maybe I made a mistake in the potential function. Let's check again.

If f(x, y) = -cos(x)sin(y), then ∇f = (sin(x)sin(y), -cos(x)cos(y)) which is exactly the vector field. So that's correct.

Alternatively, maybe I should compute the integral directly using the parameterization just to verify.

Given the parameterization:

x(t) = (3π/2)t

y(t) = (3π/2)t - π

And t goes from 0 to 1, since when t=0, x=0, y=-π; and when t=1, x=3π/2, y=3π/2 - π = π/2.

So let's compute dx/dt = 3π/2

dy/dt = 3π/2

Then, substitute x and y in terms of t into the integral:

Integral from t=0 to t=1 of [sin(x)sin(y) dx/dt - cos(x)cos(y) dy/dt] dt

Which is:

Integral from 0 to 1 [ sin((3π/2)t) sin((3π/2)t - π) * (3π/2) - cos((3π/2)t) cos((3π/2)t - π) * (3π/2) ] dt

Factor out (3π/2):

(3π/2) Integral from 0 to 1 [ sin((3π/2)t) sin((3π/2)t - π) - cos((3π/2)t) cos((3π/2)t - π) ] dt

Hmm. Let's simplify the integrand. Notice that sin(A)sin(B) - cos(A)cos(B) is equal to -cos(A + B). Because cos(A + B) = cosA cosB - sinA sinB, so -cos(A + B) = sinA sinB - cosA cosB. So the integrand becomes -cos(A + B), where A = (3π/2)t and B = (3π/2)t - π.

Therefore, A + B = (3π/2)t + (3π/2)t - π = 3π t - π.

Therefore, the integrand is -cos(3π t - π). So the integral becomes:

(3π/2) Integral from 0 to 1 [ -cos(3π t - π) ] dt

= -(3π/2) Integral from 0 to 1 cos(3π t - π) dt

Let me compute this integral. Let u = 3π t - π, so du/dt = 3π, so dt = du/(3π). When t=0, u= -π; when t=1, u=3π - π=2π.

Therefore, Integral becomes:

-(3π/2) * [1/(3π)] Integral from u=-π to u=2π cos(u) du

Simplify coefficients: -(3π/2)*(1/(3π)) = -1/2

So we have -1/2 [ sin(u) ] from -π to 2π

Compute sin(2π) - sin(-π) = 0 - 0 = 0

Therefore, the integral is -1/2 * 0 = 0

So both methods give 0. So that confirms it. So the integral is indeed 0. Therefore, the answer is 0. Hmm, that's interesting. So even though the path is a bit complicated, the integral ends up being zero because the vector field is conservative and the potential function has the same value at both endpoints. Wait, but f(3π/2, π/2) and f(0, -π) both evaluated to zero. So that's why the difference is zero.

Alternatively, maybe there's symmetry here. Since the integral is over a closed loop? Wait, no, the path is from (0, -π) to (3π/2, π/2), not a closed loop. But even then, the potential function gives the same value at both points, so the integral is zero. That seems correct.

So I think the answer is 0. Both methods confirm it. Therefore, the line integral equals zero.

**Final Answer**
\\boxed{0}
</original_reasoning>

### Output Format
Put your rewritten reasoning process inside <rewrite_reasoning> tags. Do not output the final answer again outside the tags, just the reasoning.

<rewrite_reasoning>
[Your rewritten reasoning process here]
</rewrite_reasoning>
"""


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
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Apply chat template
    messages = [{"role": "user", "content": SAMPLE_PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nPrompt length: {len(tokenizer.encode(prompt))} tokens")
    print(f"Prompt preview (first 500 chars):\n{prompt[:500]}...")

    # Initialize vLLM
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=15500,
        gpu_memory_utilization=0.3,
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nGenerating {args.num_samples} samples...")
    outputs = llm.generate([prompt], sampling_params)

    valid_count = 0
    for i, output in enumerate(outputs[0].outputs):
        response = output.text
        is_valid, content = check_format(response)

        print(f"\n{'='*80}")
        print(f"Sample {i+1}: {'✓ VALID FORMAT' if is_valid else '✗ INVALID FORMAT'}")
        print(f"{'='*80}")

        if is_valid:
            valid_count += 1
            # Show preview of valid content
            content_preview = content[:500]
            if len(content) > 500:
                content_preview += "..."
            print(f"\nContent preview:\n{content_preview}")
        else:
            # Show full response for invalid cases to debug
            response_preview = response[:800]
            if len(response) > 800:
                response_preview += "..."
            print(f"\nFull response (preview):\n{response_preview}")

            # Check for common issues
            if "<rewrite_reasoning>" not in response:
                print("\n⚠ Missing <rewrite_reasoning> tag")
            if "</rewrite_reasoning>" not in response:
                print("\n⚠ Missing </rewrite_reasoning> closing tag")
            if response.strip().startswith("<rewrite_reasoning>\n"):
                print("\n⚠ Tag might be malformed")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {valid_count}/{args.num_samples} samples have valid format")
    print(f"{'='*80}")

    if valid_count == 0:
        print("\n⚠ WARNING: No valid format detected!")
        print("This suggests the model may not be able to generate the required format.")
        print("Consider:")
        print("1. Adding format examples to the prompt")
        print("2. Using few-shot prompting")
        print("3. Fine-tuning the model on format examples")


if __name__ == "__main__":
    main()
