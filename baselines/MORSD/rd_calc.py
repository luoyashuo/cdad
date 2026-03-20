#!/usr/bin/env python3
"""
MORSD: Rationale Difficulty (RD) 计算脚本

RD(r, q) = PPL(answer | rationale, question) / PPL(answer | question)

特性:
- 多卡 DP: 每张 GPU 运行独立 vllm 实例处理数据分片，而非 TP 浪费通信
- 断点续传: worker 定期写 checkpoint，中断后自动跳过已完成项
- 增量计算: 多次用不同模型运行，结果追加到 rds 列表
- 拉满利用率: 一次性将分片内全部 prompt 交给 vllm 自行调度

输入/输出格式:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "solution": "...",
            "rds": [
                {
                    "model_name": "Llama-3.2-3B-Instruct",
                    "model_path": "/path/to/model",
                    "ppl_with_rationale": xxx,
                    "ppl_without_rationale": xxx,
                    "rd": xxx
                }
            ]
        }, ...
    ]
}

用法:
  # 8 卡 DP，自动检测 GPU 数量
  python rd_calc.py -i input.jsonl -o output.jsonl -m /path/to/model

  # 指定 4 卡
  python rd_calc.py -i input.jsonl -o output.jsonl -m /path/to/model --num-gpus 4

  # 中断后重新运行，自动续传
  python rd_calc.py -i input.jsonl -o output.jsonl -m /path/to/model
"""

import json
import argparse
import math
import os
import sys
import subprocess
import signal
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

import numpy as np


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="计算 MORSD Rationale Difficulty (RD)")
    parser.add_argument("--input", "-i", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model", "-m", required=True, help="学生模型路径")
    parser.add_argument("--model-name", default=None,
                        help="模型名称（默认取路径最后一级）")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="DP GPU 数量（默认自动检测可用 GPU 数）")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92,
                        help="GPU 显存利用率（默认 0.92，尽量拉满）")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--checkpoint-steps", type=int, default=500,
                        help="每处理 N 条 reasoning 写一次 checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="强制重算（即使该模型已有 RD 记录）")

    # Worker 内部参数（对用户隐藏）
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_assign-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=str, help=argparse.SUPPRESS)

    return parser.parse_args()


# ============================================================================
# 数据工具
# ============================================================================

def load_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def model_already_computed(reasoning: Dict[str, Any], model_name: str) -> bool:
    for entry in reasoning.get("rds", []):
        if entry.get("model_name") == model_name:
            return True
    return False


def get_available_gpus() -> List[int]:
    """获取可用 GPU 列表（尊重 CUDA_VISIBLE_DEVICES）"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]


# ============================================================================
# PPL / RD 计算核心
# ============================================================================

def get_answer_start_token_idx(tokenizer, full_text: str, solution: str) -> int:
    """
    找到 solution 在 full_text 中对应的起始 token index。
    rfind 定位字符串位置，tokenize 前缀得到 token 数。
    """
    pos = full_text.rfind(solution)
    if pos == -1:
        pos = full_text.rfind(solution.strip())
        if pos == -1:
            return -1
    prefix_text = full_text[:pos]
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    return len(prefix_tokens)


def build_prompt_with_rationale(tokenizer, question: str, thought: str, solution: str) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n\n{solution}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_prompt_without_rationale(tokenizer, question: str, solution: str) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def compute_ppl_from_logprobs(logprobs: list, answer_start_idx: int) -> float:
    answer_logprobs = []
    for idx in range(answer_start_idx, len(logprobs)):
        lp = logprobs[idx]
        if lp is None:
            continue
        for token_id, logprob_obj in lp.items():
            answer_logprobs.append(logprob_obj.logprob)
            break
    if not answer_logprobs:
        return float('inf')
    avg_nll = -sum(answer_logprobs) / len(answer_logprobs)
    return math.exp(min(avg_nll, 100))


# ============================================================================
# Worker: 单 GPU 上运行 vllm 处理分配到的数据
# ============================================================================

def worker_main(args):
    """Worker 进程入口：加载模型，处理分配到的 (d_idx, r_idx) 列表"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from tqdm import tqdm

    gpu_id = args._gpu_id
    assign_file = args._assign_file
    result_file = args._result_file
    model_name = args.model_name or Path(args.model).name
    model_path = str(Path(args.model).resolve())

    # 读取分配的任务
    with open(assign_file, 'r') as f:
        assignments = json.load(f)  # [(d_idx, r_idx), ...]

    if not assignments:
        print(f"[GPU {gpu_id}] 无任务，退出")
        return

    # 读取已完成的 checkpoint
    done_set: Set[Tuple[int, int]] = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done_set.add((entry["d_idx"], entry["r_idx"]))

    remaining = [(d, r) for d, r in assignments if (d, r) not in done_set]
    print(f"[GPU {gpu_id}] 分配 {len(assignments)} 条，已完成 {len(done_set)} 条，剩余 {len(remaining)} 条")

    if not remaining:
        print(f"[GPU {gpu_id}] 全部已完成，跳过")
        return

    # 加载原始数据（只读，用于构建 prompt）
    data = load_data(args.input)

    # 加载模型
    print(f"[GPU {gpu_id}] 加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    # 构建全部 prompt，一次性交给 vllm 调度
    print(f"[GPU {gpu_id}] 构建 {len(remaining)} × 2 = {len(remaining)*2} 个 prompts...")
    all_prompts = []
    all_prefix_lens = []
    prompt_map = []  # (idx_in_remaining, type)

    for rem_idx, (d_idx, r_idx) in enumerate(tqdm(remaining, desc=f"[GPU {gpu_id}] 构建 prompts")):
        item = data[d_idx]
        question = item["question"]
        reasoning = item["reasoning"][r_idx]
        thought = reasoning["thought"]
        solution = reasoning["solution"]

        # with rationale
        full_wr = build_prompt_with_rationale(tokenizer, question, thought, solution)
        ans_start = get_answer_start_token_idx(tokenizer, full_wr, solution)
        if ans_start == -1:
            full_tokens = tokenizer.encode(full_wr, add_special_tokens=False)
            sol_tokens = tokenizer.encode(solution, add_special_tokens=False)
            ans_start = len(full_tokens) - len(sol_tokens)
        all_prompts.append(full_wr)
        all_prefix_lens.append(ans_start)
        prompt_map.append((rem_idx, "with_r"))

        # without rationale
        full_wor = build_prompt_without_rationale(tokenizer, question, solution)
        ans_start = get_answer_start_token_idx(tokenizer, full_wor, solution)
        if ans_start == -1:
            full_tokens = tokenizer.encode(full_wor, add_special_tokens=False)
            sol_tokens = tokenizer.encode(solution, add_special_tokens=False)
            ans_start = len(full_tokens) - len(sol_tokens)
        all_prompts.append(full_wor)
        all_prefix_lens.append(ans_start)
        prompt_map.append((rem_idx, "without_r"))

    # 一次性推理，让 vllm 自行调度 batch（拉满利用率）
    print(f"[GPU {gpu_id}] 开始推理 {len(all_prompts)} 个 prompts...")
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    # 收集 PPL 结果
    ppl_map: Dict[Tuple[int, str], float] = {}  # (rem_idx, type) -> ppl
    for i, output in enumerate(outputs):
        rem_idx, ptype = prompt_map[i]
        prefix_len = all_prefix_lens[i]
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            ppl = float('inf')
        else:
            ppl = compute_ppl_from_logprobs(prompt_logprobs, prefix_len)
        ppl_map[(rem_idx, ptype)] = ppl

    # 计算 RD 并写入结果文件（追加模式）
    print(f"[GPU {gpu_id}] 写入结果...")
    with open(result_file, 'a', encoding='utf-8') as f:
        for rem_idx, (d_idx, r_idx) in enumerate(remaining):
            ppl_wr = ppl_map.get((rem_idx, "with_r"), float('inf'))
            ppl_wor = ppl_map.get((rem_idx, "without_r"), float('inf'))

            if ppl_wor > 0 and ppl_wor != float('inf'):
                rd = ppl_wr / ppl_wor
            else:
                rd = float('inf')

            result = {
                "d_idx": d_idx,
                "r_idx": r_idx,
                "model_name": model_name,
                "model_path": model_path,
                "ppl_with_rationale": ppl_wr,
                "ppl_without_rationale": ppl_wor,
                "rd": rd,
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"[GPU {gpu_id}] 完成! 处理 {len(remaining)} 条 reasoning")


# ============================================================================
# Launcher: 数据分片 → 启动多 Worker → 合并结果
# ============================================================================

def launcher_main(args):
    model_name = args.model_name or Path(args.model).name
    model_path = str(Path(args.model).resolve())

    print(f"{'='*60}")
    print(f"  MORSD RD 计算 (DP 模式)")
    print(f"{'='*60}")
    print(f"模型名称: {model_name}")
    print(f"模型路径: {model_path}")

    # 确定 GPU
    all_gpus = get_available_gpus()
    num_gpus = args.num_gpus or len(all_gpus)
    gpu_ids = all_gpus[:num_gpus]
    print(f"使用 GPU: {gpu_ids} ({num_gpus} 张)")

    # 加载数据
    print(f"加载数据: {args.input}")
    data = load_data(args.input)
    print(f"共 {len(data)} 条数据, {sum(len(item['reasoning']) for item in data)} 条 reasoning")

    # 确定需要计算的 (d_idx, r_idx)
    need_compute = []
    already_done = 0
    for d_idx, item in enumerate(data):
        for r_idx, reasoning in enumerate(item["reasoning"]):
            if not args.force and model_already_computed(reasoning, model_name):
                already_done += 1
            else:
                need_compute.append((d_idx, r_idx))

    if already_done > 0:
        print(f"输入文件中已有 {already_done} 条含模型 '{model_name}' 的 RD，跳过")

    if not need_compute:
        print("所有 reasoning 均已计算，无需重复计算。使用 --force 强制重算。")
        return

    print(f"待计算: {len(need_compute)} 条 reasoning")

    # 准备工作目录
    output_path = Path(args.output)
    work_dir = output_path.parent / f".rd_calc_{output_path.stem}_{model_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 均匀分片
    shards = [[] for _ in range(num_gpus)]
    for i, item in enumerate(need_compute):
        shards[i % num_gpus].append(item)

    for i, shard in enumerate(shards):
        print(f"  GPU {gpu_ids[i]}: {len(shard)} 条 reasoning")

    # 写分配文件
    assign_files = []
    result_files = []
    for i in range(num_gpus):
        af = work_dir / f"shard_{i}.assign.json"
        rf = work_dir / f"shard_{i}.results.jsonl"
        with open(af, 'w') as f:
            json.dump(shards[i], f)
        assign_files.append(str(af))
        result_files.append(str(rf))

    # 启动 worker 子进程
    print(f"\n启动 {num_gpus} 个 worker 进程...")
    procs = []
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[i])

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--input", args.input,
            "--output", args.output,
            "--model", args.model,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--_worker",
            "--_gpu-id", str(gpu_ids[i]),
            "--_assign-file", assign_files[i],
            "--_result-file", result_files[i],
        ]
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])

        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        print(f"  Worker {i} (GPU {gpu_ids[i]}): PID {p.pid}")

    # 等待所有 worker 完成
    print(f"\n等待 {num_gpus} 个 worker 完成...")
    failed = []
    for i, p in enumerate(procs):
        ret = p.wait()
        if ret != 0:
            failed.append((i, gpu_ids[i], ret))
            print(f"  Worker {i} (GPU {gpu_ids[i]}): 退出码 {ret}")
        else:
            print(f"  Worker {i} (GPU {gpu_ids[i]}): 完成")

    if failed:
        print(f"\n警告: {len(failed)} 个 worker 失败，可重新运行自动续传")

    # 合并结果
    print("\n合并结果...")
    result_count = 0
    for rf in result_files:
        if not os.path.exists(rf):
            continue
        with open(rf, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                d_idx = entry["d_idx"]
                r_idx = entry["r_idx"]
                reasoning = data[d_idx]["reasoning"][r_idx]

                rd_entry = {
                    "model_name": entry["model_name"],
                    "model_path": entry["model_path"],
                    "ppl_with_rationale": entry["ppl_with_rationale"],
                    "ppl_without_rationale": entry["ppl_without_rationale"],
                    "rd": entry["rd"],
                }

                # 如果 force 模式，先移除旧记录
                if args.force and "rds" in reasoning:
                    reasoning["rds"] = [
                        e for e in reasoning["rds"] if e.get("model_name") != model_name
                    ]

                if "rds" not in reasoning:
                    reasoning["rds"] = []
                reasoning["rds"].append(rd_entry)
                result_count += 1

    print(f"合并 {result_count} 条 RD 结果")

    # 保存最终输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # RD 统计
    all_rds = []
    for item in data:
        for r in item["reasoning"]:
            for e in r.get("rds", []):
                if e["model_name"] == model_name and e["rd"] != float('inf'):
                    all_rds.append(e["rd"])

    if all_rds:
        rds = np.array(all_rds)
        print(f"\n模型 '{model_name}' RD 统计:")
        print(f"  数量: {len(rds)}")
        print(f"  均值: {rds.mean():.4f}")
        print(f"  中位数: {np.median(rds):.4f}")
        print(f"  最小值: {rds.min():.4f}")
        print(f"  最大值: {rds.max():.4f}")
        print(f"  标准差: {rds.std():.4f}")
        print(f"  RD < 1.0: {(rds < 1.0).sum()} ({(rds < 1.0).mean()*100:.1f}%)")

    # 清理工作目录
    if not failed:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
        print("已清理临时文件")
    else:
        print(f"保留临时文件（有 worker 失败）: {work_dir}")

    print(f"\n结果保存至: {args.output}")


# ============================================================================
# 入口
# ============================================================================

def main():
    args = parse_args()
    if args._worker:
        worker_main(args)
    else:
        launcher_main(args)


if __name__ == "__main__":
    main()
