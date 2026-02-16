#!/usr/bin/env python3
"""
Build COT rewrite training dataset from rv_optimal data by mapping to original sources.
Multi-process version for better performance on large datasets.

This script:
1. Reads rv_optimal data containing question and optimized reasoning
2. Maps each question back to its original source (math/code)
3. Uses the original reasoning structure to build COT rewrite dataset

Output format:
    {
        "question": "...",
        "original_thought": "...",  # from original source (matched with rv_optimal think)
        "solution": "...",          # from original source
        "system_prompt": "...",     # system instruction
        "user_prompt": "..."        # user input with problem and original reasoning
    }
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import itertools


def extract_think_from_rv(content: str) -> str:
    return content.split("<think>")[1].split("</think>")[0].strip()


def get_fsf(reasoning_item: Dict[str, Any]) -> float:
    """从 reasoning_item 中提取 FSF 值"""
    reasoning_graph_list = reasoning_item.get("reasoning_graph", [])
    if not reasoning_graph_list or not isinstance(reasoning_graph_list, list):
        return 0.0

    first_graph = reasoning_graph_list[0]
    if not isinstance(first_graph, dict):
        return 0.0

    return first_graph.get("fsf", 0.0)


def build_prompt(question: str, thought: str) -> Tuple[str, str]:
    """
    构建 system_prompt 和 user_prompt

    Args:
        question: 问题文本
        thought: 推理过程

    Returns:
        (system_prompt, user_prompt) 元组
    """
    system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' provides a raw derivation that leads to a correct solution. Your task is to extract the correct path and present it clearly.

Follow these guidelines:
1. **Refine and Bridge**: Retain the core logic, including essential self-reflections and verification steps. Supplement logical leaps and clarify complex steps to ensure a smooth, coherent flow.
2. **Filter Invalid Paths**: Remove derivation branches within the 'Original Reasoning' that do not lead to the final answer (e.g., failed attempts, incorrect directions, or dead ends). Keep only the effective reasoning path.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

**Output Rules:**
- Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags.
- Do NOT mention that you are rewriting another reasoning process."""

    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}"""

    return system_prompt, user_prompt


# =============================================================================
# 单文件处理函数（用于多进程）
# =============================================================================

def _load_single_rv_file(args: Tuple[str, int]) -> List[Dict[str, Any]]:
    """
    加载单个 rv_optimal 文件（用于多进程）

    Args:
        args: (file_path, worker_id) 元组

    Returns:
        包含 question 和 think 的数据列表
    """
    file_path, worker_id = args
    entries = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                conversations = data.get("conversations", [])
                if len(conversations) < 2:
                    continue

                # question 来自 conversation[0]
                question = conversations[0].get("content", "")

                # think 来自 conversation[1] 的 content
                assistant_content = conversations[1].get("content", "")
                try:
                    think = extract_think_from_rv(assistant_content)
                except IndexError:
                    continue

                entries.append({
                    "question": question,
                    "think": think,
                    "full_content": assistant_content  # 保留完整内容用于调试
                })
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading {file_path}: {e}")

    return entries


def _load_single_original_file(args: Tuple[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载单个原始数据文件（用于多进程）
    返回该文件中的 question -> list of reasoning_items 映射
    保留所有 reasoning 项，后续根据 think 内容匹配

    Args:
        args: (file_path, worker_id) 元组

    Returns:
        question -> list of reasoning_item 的字典
    """
    file_path, worker_id = args
    mapping = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = data.get("question", "")
                if not question:
                    continue

                reasoning_list = data.get("reasoning", [])
                if not reasoning_list or not isinstance(reasoning_list, list):
                    continue

                # 保留所有有效的 reasoning 项
                if question not in mapping:
                    mapping[question] = []
                
                for reasoning_item in reasoning_list:
                    thought = reasoning_item.get("thought", "")
                    if thought:
                        mapping[question].append(reasoning_item)
    except Exception as e:
        print(f"[Worker {worker_id}] Error loading {file_path}: {e}")

    return mapping


def find_matching_think(think: str, reasoning_list: List[Dict[str, Any]], question: str = "") -> Dict[str, Any]:
    """
    在 reasoning_list 中找到与 think 内容完全匹配的 reasoning_item
    
    Args:
        think: rv_optimal 中的 think 内容
        reasoning_list: 该 question 对应的所有 reasoning 项
        question: 问题文本（用于异常信息）
        
    Returns:
        匹配的 reasoning_item
        
    Raises:
        ValueError: 如果没有找到完全匹配的 reasoning_item
    """
    if not reasoning_list:
        raise ValueError(f"No reasoning items found for question: {question[:100]}...")
    
    think_stripped = think.strip()
    
    # 完全匹配
    for item in reasoning_list:
        thought = item.get("thought", "").strip()
        if thought == think_stripped:
            return item
    
    # 未找到匹配，抛出异常
    available_thoughts = [item.get("thought", "")[:200] + "..." for item in reasoning_list[:3]]
    raise ValueError(
        f"No exact match found for think content.\n"
        f"Question: {question[:100]}...\n"
        f"Think (first 200 chars): {think_stripped[:200]}...\n"
        f"Available thoughts (first 3): {available_thoughts}"
    )


def _process_single_chunk(args: Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict, int]) -> Dict[str, Any]:
    """
    处理单个数据块（用于多进程）

    Args:
        args: (rv_chunk, original_mapping, tokenizer_config, chunk_id) 元组

    Returns:
        处理结果字典
    """
    rv_chunk, original_mapping, tokenizer_config, chunk_id = args

    result_entries = []
    matched_count = 0
    no_match_count = 0
    skipped_count = 0

    # Lazy load tokenizer if needed
    tokenizer = None
    if tokenizer_config.get("tokenizer_path"):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_config["tokenizer_path"],
                trust_remote_code=True
            )
        except Exception:
            pass  # 忽略 tokenizer 加载失败

    max_problem_tokens = tokenizer_config.get("max_problem_tokens")

    for rv_entry in rv_chunk:
        question = rv_entry["question"]
        think = rv_entry["think"]

        # 在原始数据中查找该 question 的所有 reasoning 项
        reasoning_list = original_mapping.get(question)

        if reasoning_list is None:
            no_match_count += 1
            continue
        
        # 找到与 think 内容匹配的 reasoning_item（完全匹配，否则抛出异常）
        try:
            reasoning_item = find_matching_think(think, reasoning_list, question)
        except ValueError as e:
            raise ValueError(f"Failed to match think for entry at chunk {chunk_id}: {e}")
        
        matched_count += 1

        # 从 reasoning_item 中获取 original_thought 和 solution
        original_thought = reasoning_item.get("thought", "")
        solution = reasoning_item.get("solution", "")
        fsf = get_fsf(reasoning_item)

        # 构建 system_prompt 和 user_prompt（使用原始数据的 thought）
        system_prompt, user_prompt = build_prompt(question, original_thought)

        # 检查 user_prompt token 长度
        if tokenizer is not None and max_problem_tokens is not None:
            try:
                prompt_token_count = len(tokenizer.encode(user_prompt))
                if prompt_token_count > max_problem_tokens:
                    skipped_count += 1
                    continue
            except Exception:
                pass  # tokenization 失败时跳过长度检查

        entry = {
            "question": question,
            "original_thought": original_thought,  # 原始数据的完整 thought（与 rv_optimal 的 think 匹配）
            "solution": solution,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "fsf": fsf
        }
        result_entries.append(entry)

    return {
        "entries": result_entries,
        "matched": matched_count,
        "no_match": no_match_count,
        "skipped": skipped_count
    }


# =============================================================================
# 多进程数据加载函数
# =============================================================================

def load_rv_optimal_data_mp(
    file_paths: List[str],
    num_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    多进程加载 rv_optimal 数据

    Args:
        file_paths: rv_optimal JSONL 文件路径列表
        num_workers: 进程数，None 表示使用 CPU 核心数

    Returns:
        包含 question 和 think 的数据列表
    """
    if num_workers is None:
        num_workers = min(cpu_count(), len(file_paths))

    print(f"  Loading {len(file_paths)} file(s) with {num_workers} workers...")

    # 准备参数
    args_list = [(fp, i) for i, fp in enumerate(file_paths)]

    # 使用进程池并行加载
    with Pool(processes=num_workers) as pool:
        results = pool.map(_load_single_rv_file, args_list)

    # 合并结果
    entries = list(itertools.chain.from_iterable(results))
    print(f"  Loaded {len(entries)} entries from {len(file_paths)} file(s)")

    return entries


def load_original_data_mp(
    file_paths: List[str],
    num_workers: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    多进程加载原始数据，建立 question -> list of reasoning_items 的映射
    保留所有 reasoning 项，后续根据 think 内容匹配

    Args:
        file_paths: 原始 JSONL 文件路径列表
        num_workers: 进程数，None 表示使用 CPU 核心数

    Returns:
        question -> list of reasoning 数据项的字典
    """
    if num_workers is None:
        num_workers = min(cpu_count(), len(file_paths))

    print(f"  Loading {len(file_paths)} file(s) with {num_workers} workers...")

    # 准备参数
    args_list = [(fp, i) for i, fp in enumerate(file_paths)]

    # 使用进程池并行加载
    with Pool(processes=num_workers) as pool:
        results = pool.map(_load_single_original_file, args_list)

    # 合并结果，合并同一 question 的 reasoning 列表
    merged_mapping = {}
    duplicate_count = 0

    for file_mapping in results:
        for question, reasoning_list in file_mapping.items():
            if question in merged_mapping:
                duplicate_count += 1
                merged_mapping[question].extend(reasoning_list)
            else:
                merged_mapping[question] = reasoning_list

    print(f"  Loaded {len(merged_mapping)} unique questions from {len(file_paths)} file(s)")
    print(f"  Questions with multiple reasonings: {duplicate_count}")

    return merged_mapping


def process_rv_optimal_with_original_mp(
    rv_entries: List[Dict[str, Any]],
    original_mapping: Dict[str, List[Dict[str, Any]]],
    tokenizer_path: Optional[str] = None,
    max_problem_tokens: Optional[int] = None,
    num_workers: Optional[int] = None,
    chunk_size: int = 1000,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    多进程处理 rv_optimal 数据，映射到原始数据并构建 cot rewrite 条目

    Args:
        rv_entries: rv_optimal 数据列表
        original_mapping: question -> reasoning数据的映射
        tokenizer_path: 用于计算 token 长度的 tokenizer 路径
        max_problem_tokens: problem token 上界
        num_workers: 进程数，None 表示使用 CPU 核心数
        chunk_size: 每个 chunk 的数据条目数
        debug: 是否打印调试信息

    Returns:
        cot rewrite 数据条目列表
    """
    if num_workers is None:
        num_workers = cpu_count()

    total_entries = len(rv_entries)
    print(f"  Processing {total_entries} entries with {num_workers} workers...")

    # 将数据分块
    chunks = []
    for i in range(0, total_entries, chunk_size):
        chunk = rv_entries[i:i + chunk_size]
        chunks.append((chunk, original_mapping, {
            "tokenizer_path": tokenizer_path,
            "max_problem_tokens": max_problem_tokens
        }, i // chunk_size))

    print(f"  Split into {len(chunks)} chunks (~{chunk_size} entries each)")

    # 使用进程池并行处理
    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_single_chunk, chunks)

    # 合并结果
    final_entries = []
    total_matched = 0
    total_no_match = 0
    total_skipped = 0

    for result in results:
        final_entries.extend(result["entries"])
        total_matched += result["matched"]
        total_no_match += result["no_match"]
        total_skipped += result["skipped"]

    print(f"  Matched: {total_matched}, No match: {total_no_match}, Skipped (too long): {total_skipped}")
    print(f"  Output: {len(final_entries)} entries")

    return final_entries


# =============================================================================
# 辅助函数
# =============================================================================

def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """保存数据到 JSONL 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_statistics(entries: List[Dict[str, Any]], name: str = "Dataset"):
    """打印数据统计信息"""
    total = len(entries)
    if total == 0:
        print(f"\n{name}: No entries")
        return

    # 计算平均 original_thought 和 solution 长度
    avg_thought_len = sum(len(e.get("original_thought", "")) for e in entries) / total
    avg_solution_len = sum(len(e.get("solution", "")) for e in entries) / total

    print(f"\n{name} Statistics:")
    print(f"  Total entries: {total}")
    print(f"  Avg original_thought length: {avg_thought_len:.0f} chars")
    print(f"  Avg solution length: {avg_solution_len:.0f} chars")

    # 如果包含 FSF 信息，打印 FSF 统计
    if "fsf" in entries[0]:
        fsf_values = [e.get("fsf", 0.0) for e in entries]
        avg_fsf = sum(fsf_values) / len(fsf_values)
        min_fsf = min(fsf_values)
        max_fsf = max(fsf_values)
        print(f"  FSF - Avg: {avg_fsf:.4f}, Min: {min_fsf:.4f}, Max: {max_fsf:.4f}")


def split_train_val(
    entries: List[Dict[str, Any]], train_ratio: float, seed: int
):
    """
    随机划分训练集和验证集

    Args:
        entries: 所有数据条目
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_entries, val_entries) 元组
    """
    import random
    random.seed(seed)

    # 复制列表并打乱
    shuffled = entries.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_entries = shuffled[:split_idx]
    val_entries = shuffled[split_idx:]

    return train_entries, val_entries


def select_top_fsf_entries(
    entries: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """
    筛选出 FSF 值最高的 top_k 条数据

    Args:
        entries: 所有数据条目
        top_k: 筛选的数量

    Returns:
        FSF 最高的 top_k 条数据
    """
    if len(entries) <= top_k:
        print(f"  Entries count ({len(entries)}) <= top_k ({top_k}), using all entries")
        return entries

    # 按 FSF 降序排序
    sorted_entries = sorted(entries, key=lambda x: x.get("fsf", 0.0), reverse=True)
    top_entries = sorted_entries[:top_k]

    return top_entries


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "rv_optimal_paths": [
            "/data/luoys/data/cdad/omnithought_rv_optimal/train.jsonl",
            "/data/luoys/data/cdad/omnithought_rv_optimal/val.jsonl",
        ],
        "original_paths": [
            "/data/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl",
            "/data/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl",
        ],
        "output_dir": "/data/luoys/data/cdad/cot_rewrite_rv_optimal",
        "tokenizer_path": "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B",
        "max_problem_tokens": 4500,
        "train_ratio": 0.999,
        "random_seed": 42,
        "num_workers": None,  # None 表示使用所有 CPU 核心
        "chunk_size": 1000,
        "top_fsf_k": 2000,  # 筛选 FSF 最高的 top_k 条数据（RL 不需要太多数据）
    }


def main(config: Optional[Dict[str, Any]] = None):
    """
    主函数

    Args:
        config: 配置字典，如果为 None 则使用默认配置
    """
    if config is None:
        config = get_default_config()

    # 验证配置
    required_keys = ["rv_optimal_paths", "original_paths", "output_dir"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    rv_optimal_paths = config["rv_optimal_paths"]
    original_paths = config["original_paths"]
    output_dir = config["output_dir"]
    tokenizer_path = config.get("tokenizer_path")
    max_problem_tokens = config.get("max_problem_tokens")
    train_ratio = config.get("train_ratio", 0.999)
    random_seed = config.get("random_seed", 42)
    num_workers = config.get("num_workers")
    chunk_size = config.get("chunk_size", 1000)
    top_fsf_k = config.get("top_fsf_k", 2000)

    print("=" * 60)
    print("COT Rewrite Dataset Builder (Multi-Process)")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  RV_OPTIMAL_PATHS: {rv_optimal_paths}")
    print(f"  ORIGINAL_PATHS:   {original_paths}")
    print(f"  OUTPUT_DIR:       {output_dir}")
    print(f"  TRAIN_RATIO:      {train_ratio}")
    print(f"  RANDOM_SEED:      {random_seed}")
    print(f"  NUM_WORKERS:      {num_workers or 'auto'}")
    print(f"  CHUNK_SIZE:       {chunk_size}")
    print(f"  TOP_FSF_K:        {top_fsf_k}")

    # Step 1: 加载原始数据，建立映射（多进程）
    print("\n" + "-" * 60)
    print("Step 1: Loading original data (multi-process)...")
    print("-" * 60)

    original_mapping = load_original_data_mp(original_paths, num_workers=num_workers)
    print(f"Total unique questions in original data: {len(original_mapping)}")

    # Step 2: 加载 rv_optimal 数据（多进程）
    print("\n" + "-" * 60)
    print("Step 2: Loading rv_optimal data (multi-process)...")
    print("-" * 60)

    rv_entries = load_rv_optimal_data_mp(rv_optimal_paths, num_workers=num_workers)

    # Step 3: 处理数据（多进程）
    print("\n" + "-" * 60)
    print("Step 3: Processing rv_optimal data with original mapping (multi-process)...")
    print("-" * 60)

    output_entries = process_rv_optimal_with_original_mp(
        rv_entries, original_mapping,
        tokenizer_path, max_problem_tokens,
        num_workers=num_workers,
        chunk_size=chunk_size,
        debug=True  # 启用调试模式
    )

    # Step 4: 筛选 FSF 最高的 top_k 条数据
    print("\n" + "-" * 60)
    print("Step 4: Selecting top FSF entries...")
    print("-" * 60)

    print(f"Total entries before filtering: {len(output_entries)}")
    filtered_entries = select_top_fsf_entries(output_entries, top_fsf_k)
    print(f"Selected top {len(filtered_entries)} entries by FSF")
    print_statistics(filtered_entries, "Filtered Dataset (Top FSF)")

    # Step 5: 划分 train/val
    print("\n" + "-" * 60)
    print("Step 5: Splitting train/val...")
    print("-" * 60)

    final_train, final_val = split_train_val(
        filtered_entries, train_ratio, random_seed
    )

    print_statistics(final_train, "Train Set")
    print_statistics(final_val, "Validation Set")

    # Step 6: 保存数据
    print("\n" + "-" * 60)
    print("Step 6: Saving data...")
    print("-" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    save_jsonl(final_train, str(train_path))
    save_jsonl(final_val, str(val_path))

    print(f"\nOutput saved to: {output_path.absolute()}")
    print(f"  Train: {train_path.name} ({len(final_train)} entries)")
    print(f"  Val:   {val_path.name} ({len(final_val)} entries)")

    # Step 7: 保存配置和统计信息
    print("\n" + "-" * 60)
    print("Step 7: Saving config and logs...")
    print("-" * 60)

    # 保存配置
    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to: {config_path.name}")

    # 保存日志
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "counts": {
            "total_rv_optimal": len(rv_entries),
            "total_original": len(original_mapping),
            "matched": len(output_entries),
            "filtered": len(filtered_entries),
            "top_fsf_k": top_fsf_k,
            "train_count": len(final_train),
            "val_count": len(final_val),
        },
    }

    log_path = output_path / "build_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Log saved to: {log_path.name}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build COT rewrite dataset from rv_optimal (multi-process)")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--rv-optimal", type=str, nargs="*", help="RV optimal data paths")
    parser.add_argument("--original", type=str, nargs="*", help="Original data paths")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--num-workers", type=int, help="Number of worker processes (default: auto)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for processing (default: 1000)")
    parser.add_argument("--top-fsf-k", type=int, default=2000, help="Select top K entries by FSF (default: 2000)")
    parser.add_argument("--train-ratio", type=float, default=0.999, help="Train/val split ratio (default: 0.999)")
    args = parser.parse_args()

    config = None

    # 如果提供了 config 文件，从中加载配置
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    # 否则使用命令行参数或默认配置
    else:
        config = get_default_config()
        if args.rv_optimal:
            config["rv_optimal_paths"] = args.rv_optimal
        if args.original:
            config["original_paths"] = args.original
        if args.output_dir:
            config["output_dir"] = args.output_dir
        if args.num_workers:
            config["num_workers"] = args.num_workers
        config["chunk_size"] = args.chunk_size
        config["top_fsf_k"] = args.top_fsf_k
        config["train_ratio"] = args.train_ratio

    main(config)

"""
# Example usage:

# 使用默认配置
python3 build_cot_rewrite_from_sharegpt.py

# 使用自定义路径
python3 build_cot_rewrite_from_sharegpt.py \
    --rv-optimal /data/luoys/data/cdad/omnithought_rv_optimal/train.jsonl \
    --original /data/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
        /data/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    --output-dir /data/luoys/data/cdad/omnithought_rv_optimal_cot_rewrite_mp

# 指定 worker 数量
python3 build_cot_rewrite_from_sharegpt.py \
    --rv-optimal /data/luoys/data/cdad/omnithought_rv_optimal_top2000/train.jsonl \
    --original /data/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
        /data/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    --output-dir /data/luoys/data/cdad/omnithought_rv_optimal_cot_rewrite_top2000_v2 \
    --num-workers 8 \
    --top-fsf-k 2000 \
    --train-ratio 0.99 \
    --chunk-size 2000

"""
