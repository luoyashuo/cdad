"""
验证 _build_prompt 方法中 answer_start_idx 的计算是否正确

修复后的方法：用两次 apply_chat_template，一次用 dummy_answer，一次用 answer
answer_start_idx = len(tokenize(question + cot + dummy_answer)) - 1
"""

import sys
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo/verl')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    '/data/luoys/modelscope_cache/models/Qwen/Qwen3-1.7B',
    trust_remote_code=True
)


def compute_answer_start_idx_fixed(question: str, cot: str, answer: str):
    """
    修复后的方法：用两次 apply_chat_template

    原理：
    1. 第一次：用 dummy_answer ("X") tokenize，得到位置
    2. 第二次：用真实 answer tokenize
    3. answer_start_idx = len(tokens_with_dummy) - 1
    """
    # 完整 prompt（用于最终使用）
    messages_full = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer, 'reasoning_content': cot}
    ]
    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )

    # 用 dummy_answer 计算 answer_start_idx
    dummy_answer = "X"
    messages_with_dummy = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': dummy_answer, 'reasoning_content': cot}
    ]
    text_with_dummy = tokenizer.apply_chat_template(
        messages_with_dummy, tokenize=False, add_generation_prompt=False
    )
    tokens_with_dummy = tokenizer.encode(text_with_dummy, add_special_tokens=False)

    # 找到 dummy_answer ("X") 在 tokens 中的位置
    # 这个位置就是 answer 开始的地方
    dummy_token_id = tokenizer.encode(dummy_answer, add_special_tokens=False)[0]
    # 从后向前找 dummy_token_id（因为可能有重复）
    for i in range(len(tokens_with_dummy) - 1, -1, -1):
        if tokens_with_dummy[i] == dummy_token_id:
            answer_start_idx = i
            break
    else:
        # 如果找不到，使用长度作为备选
        answer_start_idx = len(tokens_with_dummy) - 1

    # 安全检查
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    if answer_start_idx > len(full_tokens):
        answer_start_idx = max(0, len(full_tokens) - 10)

    return full_text, answer_start_idx, full_tokens


def verify(question: str, cot: str, answer: str, test_name: str):
    """验证单个测试用例"""
    print("=" * 100)
    print(f"{test_name}")
    print("=" * 100)

    print(f"Question: {question[:80]}{'...' if len(question) > 80 else ''}")
    print(f"CoT: {cot[:80]}{'...' if len(cot) > 80 else ''}")
    print(f"Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
    print()

    full_text, answer_start_idx, full_tokens = compute_answer_start_idx_fixed(question, cot, answer)

    print(f"Total tokens: {len(full_tokens)}")
    print(f"Answer start index (logprob_start_len): {answer_start_idx}")
    print()

    # 解码验证
    from_answer_tokens = full_tokens[answer_start_idx:]
    from_answer_text = tokenizer.decode(from_answer_tokens, skip_special_tokens=False)
    print(f"Tokens {answer_start_idx} ~ end decoded:")
    print(f"  {repr(from_answer_text)}")
    print()

    # 检查
    starts_with = from_answer_text.lstrip().startswith(answer)
    contains = answer in from_answer_text
    valid = answer_start_idx <= len(full_tokens)

    print(f"Original answer: {repr(answer)}")
    print(f"Decoded text starts with answer: {starts_with}")
    print(f"Answer in decoded text: {contains}")
    print(f"Validation: {answer_start_idx} <= {len(full_tokens)} ? {valid}")

    return {
        'answer_start_idx': answer_start_idx,
        'total_tokens': len(full_tokens),
        'starts_with': starts_with,
        'valid': valid
    }


if __name__ == "__main__":
    print("\n")
    print("*" * 100)
    print("验证 answer_start_idx 计算方法")
    print("方法: answer_start_idx = len(tokenize(question + cot + dummy_answer)) - 1")
    print("*" * 100)

    result1 = verify(
        question="What is 1+1?",
        cot="Let me think... 1+1=2",
        answer="The answer is 2.",
        test_name="测试 1: 简单数学问题"
    )

    print("\n")

    result2 = verify(
        question="Find the smallest natural number that gives remainder 2.",
        cot="I need to find LCM of 3,4,6,8 and add 2. LCM is 24, so answer is 26.",
        answer="The smallest natural number is 26.",
        test_name="测试 2: 长文本"
    )

    print("\n")

    result3 = verify(
        question="Find $(\\frac{1+i}{\\sqrt{2}})^{47}$.",
        cot="First, convert to polar form. Modulus is 1, argument is $\\pi/4$.",
        answer="$e^{i \\cdot 3\\pi/4} = -\\frac{1}{\\sqrt{2}} + \\frac{i}{\\sqrt{2}}$",
        test_name="测试 3: 数学公式"
    )

    print("\n")
    print("=" * 100)
    print("总结")
    print("=" * 100)
    print(f"测试 1: {'PASS ✓' if result1['valid'] and result1['starts_with'] else 'FAIL ✗'}")
    print(f"  - answer_start_idx: {result1['answer_start_idx']}, total_tokens: {result1['total_tokens']}")
    print()
    print(f"测试 2: {'PASS ✓' if result2['valid'] and result2['starts_with'] else 'FAIL ✗'}")
    print(f"  - answer_start_idx: {result2['answer_start_idx']}, total_tokens: {result2['total_tokens']}")
    print()
    print(f"测试 3: {'PASS ✓' if result3['valid'] and result3['starts_with'] else 'FAIL ✗'}")
    print(f"  - answer_start_idx: {result3['answer_start_idx']}, total_tokens: {result3['total_tokens']}")
    print()
    print("计算方法: answer_start_idx = len(tokenize(question+cot+dummy_answer)) - 1")
