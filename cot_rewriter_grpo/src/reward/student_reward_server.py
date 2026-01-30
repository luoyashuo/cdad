"""
Student Reward Server

A standalone HTTP server that runs on GPU and computes log probabilities.
This server can be accessed by RewardLoopWorker (which runs on CPU) via HTTP.

Uses Qwen3 chat template with reasoning_content field for proper formatting.

Usage:
    # Start the server on a specific GPU
    CUDA_VISIBLE_DEVICES=5 python -m src.reward.student_reward_server --port 8765
"""

import argparse
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Student Reward Server")

# Global model and tokenizer
model = None
tokenizer = None
device = None


class LogProbRequest(BaseModel):
    """Request for computing log probabilities."""
    questions: List[str]
    rewrite_cots: List[str]
    answers: List[str]
    original_cots: Optional[List[str]] = None  # 用于比较


class LogProbResponse(BaseModel):
    """Response containing computed log probabilities."""
    rewards: List[float]  # 0 或 1 的 reward (比较模式) 或 mean log prob (非比较模式)
    rewrite_mean_log_probs: List[float]  # rewrite_cot 条件下的 mean log prob
    original_mean_log_probs: List[float]  # original_cot 条件下的 mean log prob
    rewrite_token_log_probs: List[List[float]]  # rewrite 的 token-level logprob
    rewrite_token_ids: List[List[int]]  # rewrite 的 answer token IDs
    # 向后兼容
    log_probs: List[float]
    token_log_probs: List[List[float]]
    token_ids: List[List[int]]


def load_model(model_path: str, device_id: int = 0, dtype: str = "bfloat16"):
    """Load the student model onto GPU."""
    global model, tokenizer, device

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading Student Model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded successfully on {device}")


@torch.no_grad()
def compute_single_log_prob(question: str, cot: str, answer: str) -> dict:
    """
    Compute log P(Answer | Question, CoT) for a single sample.

    Uses Qwen3 chat template with reasoning_content field:
    - User message: question
    - Assistant message: answer with reasoning_content=cot

    Format:
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        <think>
        {cot}
        </think>

        {answer}<|im_end|>

    Returns:
        dict with:
            - mean_log_prob: float, answer 部分 token logprob 的平均值
            - sum_log_prob: float, answer 部分 token logprob 的总和
            - token_log_probs: List[float], 每个 answer token 的 logprob
            - token_ids: List[int], answer 部分的 token IDs
    """
    # 使用 Qwen3 chat template 格式
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer, 'reasoning_content': cot}
    ]

    # 应用 chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 构建 prefix（不包含 answer）用于定位 answer 开始位置
    prefix_messages = [
        {'role': 'user', 'content': question},
    ]
    prefix_base = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 手动添加 <think> 部分
    prefix_with_think = prefix_base + f"<think>\n{cot}\n</think>\n\n"

    # Tokenize
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    all_token_ids = inputs["input_ids"][:, 1:]

    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, all_token_ids.unsqueeze(2)).squeeze(2)

    # 计算 prefix 的 token 长度
    prefix_token_ids = tokenizer(
        prefix_with_think,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )["input_ids"][0]
    prefix_len = len(prefix_token_ids) - 1

    # Extract answer token logprobs and IDs
    answer_log_probs = token_log_probs[0, prefix_len:].tolist()
    answer_token_ids = all_token_ids[0, prefix_len:].tolist()

    # 移除末尾的特殊 token
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = tokenizer.eos_token_id
    special_ids = {im_end_token_id, eos_token_id}
    if im_end_token_id is None:
        special_ids = {eos_token_id}

    while answer_token_ids and answer_token_ids[-1] in special_ids:
        answer_token_ids.pop()
        answer_log_probs.pop()

    if answer_log_probs:
        return {
            "mean_log_prob": sum(answer_log_probs) / len(answer_log_probs),
            "sum_log_prob": sum(answer_log_probs),
            "token_log_probs": answer_log_probs,
            "token_ids": answer_token_ids,
        }

    return {
        "mean_log_prob": -100.0,
        "sum_log_prob": -100.0,
        "token_log_probs": [],
        "token_ids": [],
    }


@app.post("/compute_log_probs", response_model=LogProbResponse)
async def compute_log_probs(request: LogProbRequest):
    """
    Compute log probabilities for a batch of samples.

    如果提供了 original_cots，会比较 rewrite 和 original 的 mean log prob：
    - 如果 rewrite_mean > original_mean → reward = 1
    - 否则 → reward = 0

    如果没有提供 original_cots，直接返回 rewrite 的 mean log prob 作为 reward。
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rewards = []
    rewrite_mean_log_probs = []
    original_mean_log_probs = []
    rewrite_token_log_probs = []
    rewrite_token_ids = []

    original_cots = request.original_cots or []

    for i, (q, rewrite_cot, ans) in enumerate(
        zip(request.questions, request.rewrite_cots, request.answers)
    ):
        try:
            # 计算 rewrite_cot 条件下的 log prob
            rewrite_result = compute_single_log_prob(q, rewrite_cot, ans)
            rewrite_mean = rewrite_result["mean_log_prob"]
            rewrite_mean_log_probs.append(rewrite_mean)
            rewrite_token_log_probs.append(rewrite_result["token_log_probs"])
            rewrite_token_ids.append(rewrite_result["token_ids"])

            # 如果提供了 original_cot，计算比较
            if i < len(original_cots) and original_cots[i]:
                original_result = compute_single_log_prob(q, original_cots[i], ans)
                original_mean = original_result["mean_log_prob"]
                original_mean_log_probs.append(original_mean)

                # Reward: 如果 rewrite 的 log prob 更高（更好），给 reward = 1
                reward = 1.0 if rewrite_mean > original_mean else 0.0
                rewards.append(reward)
            else:
                original_mean_log_probs.append(-100.0)
                rewards.append(rewrite_mean)  # 直接用 mean log prob 作为 reward

        except Exception as e:
            print(f"[StudentRewardServer] Error for sample {i}: {e}")
            rewards.append(0.0)
            rewrite_mean_log_probs.append(-100.0)
            original_mean_log_probs.append(-100.0)
            rewrite_token_log_probs.append([])
            rewrite_token_ids.append([])

    return LogProbResponse(
        rewards=rewards,
        rewrite_mean_log_probs=rewrite_mean_log_probs,
        original_mean_log_probs=original_mean_log_probs,
        rewrite_token_log_probs=rewrite_token_log_probs,
        rewrite_token_ids=rewrite_token_ids,
        # 向后兼容
        log_probs=rewrite_mean_log_probs,
        token_log_probs=rewrite_token_log_probs,
        token_ids=rewrite_token_ids,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(device) if device else "not initialized",
        "model_loaded": model is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Student Reward Server")
    parser.add_argument("--model_path", type=str, default="/data-share/LLM/Qwen3-1.7B",
                        help="Path to the student model")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    # Load model
    load_model(args.model_path, args.device_id, args.dtype)

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
