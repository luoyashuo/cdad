#!/usr/bin/env python3
"""
Build COT rewrite training dataset from omnithought code and math data.

This script:
1. Reads JSONL files containing reasoning traces with 'thought' and 'solution'
2. Extracts question, thought, and solution from each reasoning path
3. Supports multiple sampling strategies (first, all, max_fsf, min_fsf)
4. Shuffles and splits data into train/val sets

Input format:
    {
        "question": "...",
        "reasoning": [
            {
                "thought": "...",
                "solution": "...",
                "reasoning_graph": [{"fsf": 0.1, ...}],
                ...
            },
            ...
        ]
    }

Output format:
    {
        "question": "...",
        "thought": "...",
        "solution": "...",
        "problem": "..."
    }

Usage:
    # Default: sample 1 thought per question (first one)
    python build_cot_rewrite_dataset.py \
        -i /path/to/train_merge_level1.jsonl \
        -o /path/to/output_dir

    # Sample all thoughts per question
    python build_cot_rewrite_dataset.py \
        -i /path/to/train_merge_level1.jsonl \
        -o /path/to/output_dir \
        --sampling-strategy all

    # Sample thought with highest FSF per question
    python build_cot_rewrite_dataset.py \
        -i /path/to/train_merge_level1.jsonl \
        -o /path/to/output_dir \
        --sampling-strategy max_fsf

    # Sample thought with lowest FSF per question
    python build_cot_rewrite_dataset.py \
        -i /path/to/train_merge_level1.jsonl \
        -o /path/to/output_dir \
        --sampling-strategy min_fsf
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build COT rewrite training dataset with shuffle and split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: sample 1 thought per question (first one)
  python build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
    -o ./cot_rewrite_output

  # Sample all thoughts per question
  python build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
    -o ./cot_rewrite_output \
    --sampling-strategy all

  # Sample thought with highest FSF per question
  python build_cot_rewrite_dataset.py \
    -i math.jsonl code.jsonl \
    -o ./cot_rewrite_output \
    --sampling-strategy max_fsf

  # Custom train/val split with specific seed
  python build_cot_rewrite_dataset.py \
    -i input.jsonl \
    -o output_dir \
    --sampling-strategy min_fsf \
    --train-ratio 0.95 \
    --seed 42
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        help="输入的 JSONL 文件路径，可指定多个文件",
    )
    parser.add_argument("--output", "-o", required=True, help="输出目录路径")
    parser.add_argument(
        "--train-ratio", "-t", type=float, default=0.98, help="训练集比例（默认: 0.98）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    parser.add_argument(
        "--sampling-strategy",
        "-s",
        type=str,
        default="first",
        choices=["first", "all", "max_fsf", "min_fsf"],
        help="""
采样策略（默认: first）：
  first   - 每个 question 只采样第一个 thought
  all     - 采样所有 thoughts
  max_fsf - 采样 FSF 最高的 thought
  min_fsf - 采样 FSF 最低的 thought
        """,
    )

    return parser.parse_args()


def get_fsf(reasoning_item: Dict[str, Any]) -> float:
    """
    从 reasoning_item 中提取 FSF 值
    
    Args:
        reasoning_item: 包含 reasoning_graph 的 reasoning 项
        
    Returns:
        FSF 值，如果没有则返回 0.0
    """
    reasoning_graph_list = reasoning_item.get("reasoning_graph", [])
    if not reasoning_graph_list or not isinstance(reasoning_graph_list, list):
        return 0.0
    
    first_graph = reasoning_graph_list[0]
    if not isinstance(first_graph, dict):
        return 0.0
    
    return first_graph.get("fsf", 0.0)


def sample_thoughts(
    reasoning_list: List[Dict[str, Any]], 
    strategy: str
) -> List[Dict[str, Any]]:
    """
    根据采样策略从 reasoning_list 中选择 thought
    
    Args:
        reasoning_list: 推理路径列表
        strategy: 采样策略 (first, all, max_fsf, min_fsf)
        
    Returns:
        选中的 reasoning 项列表
    """
    if not reasoning_list:
        return []
    
    # 过滤掉没有 thought 或 solution 的项
    valid_items = [
        item for item in reasoning_list 
        if isinstance(item, dict) 
        and item.get("thought", "") 
        and item.get("solution", "")
    ]
    
    if not valid_items:
        return []
    
    if strategy == "all":
        return valid_items
    
    elif strategy == "first":
        return [valid_items[0]]
    
    elif strategy == "max_fsf":
        # 选择 FSF 最高的
        return [max(valid_items, key=get_fsf)]
    
    elif strategy == "min_fsf":
        # 选择 FSF 最低的
        return [min(valid_items, key=get_fsf)]
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def process_single_file(input_path: str, strategy: str) -> List[Dict[str, Any]]:
    """
    处理单个输入文件，生成 COT rewrite 训练数据

    Args:
        input_path: 输入 JSONL 文件路径
        strategy: 采样策略

    Returns:
        数据条目列表，每个条目包含 question, thought, solution
    """
    entries = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {Path(input_path).name}"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = data.get("question", "")
            reasoning_list = data.get("reasoning", [])

            if not question or not reasoning_list:
                continue

            # 根据策略采样 thoughts
            selected_items = sample_thoughts(reasoning_list, strategy)
            
            for item in selected_items:
                entry = {
                    "question": question,
                    "thought": item.get("thought", ""),
                    "solution": item.get("solution", ""),
                }
                problem = f"""\
Please rewrite the following reasoning process based on the given question.

Question: {entry["question"]}

Reasoning Process: {entry["thought"]}
"""
                entry["problem"] = problem
                # 如果使用了 FSF 相关策略，添加 fsf 字段用于参考
                if strategy in ["max_fsf", "min_fsf"]:
                    entry["fsf"] = get_fsf(item)
                entries.append(entry)

    return entries


def split_train_val(
    entries: List[Dict[str, Any]], train_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    随机划分训练集和验证集

    Args:
        entries: 所有数据条目
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_entries, val_entries) 元组
    """
    random.seed(seed)

    # 复制列表并打乱
    shuffled = entries.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_entries = shuffled[:split_idx]
    val_entries = shuffled[split_idx:]

    return train_entries, val_entries


def print_statistics(entries: List[Dict[str, Any]], name: str = "Dataset", strategy: str = "first"):
    """打印数据统计信息"""
    total = len(entries)
    # 计算平均 thought 和 solution 长度
    avg_thought_len = sum(len(e.get("thought", "")) for e in entries) / total if total > 0 else 0
    avg_solution_len = sum(len(e.get("solution", "")) for e in entries) / total if total > 0 else 0

    print(f"\n{name} Statistics:")
    print(f"  Total entries: {total}")
    if total > 0:
        print(f"  Avg thought length: {avg_thought_len:.0f} chars")
        print(f"  Avg solution length: {avg_solution_len:.0f} chars")
        
        # 如果包含 FSF 信息，打印 FSF 统计
        if strategy in ["max_fsf", "min_fsf"] and "fsf" in entries[0]:
            fsf_values = [e.get("fsf", 0.0) for e in entries]
            avg_fsf = sum(fsf_values) / len(fsf_values)
            min_fsf = min(fsf_values)
            max_fsf = max(fsf_values)
            print(f"  FSF - Avg: {avg_fsf:.4f}, Min: {min_fsf:.4f}, Max: {max_fsf:.4f}")


def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """保存数据到 JSONL 文件"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    args = parse_arguments()

    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    # 处理所有输入文件
    all_entries = []
    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"Warning: Input file {input_path} does not exist, skipping...")
            continue

        entries = process_single_file(input_path, args.sampling_strategy)
        all_entries.extend(entries)
        print(f"Processed {input_path}: {len(entries)} entries")

    if not all_entries:
        print("No valid entries found!")
        return

    print_statistics(all_entries, "Total Data", args.sampling_strategy)

    # 划分训练集和验证集
    train_entries, val_entries = split_train_val(
        all_entries, args.train_ratio, args.seed
    )

    # 打印划分后的统计
    print_statistics(train_entries, "Train Set", args.sampling_strategy)
    print_statistics(val_entries, "Validation Set", args.sampling_strategy)

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    # 保存数据
    save_jsonl(train_entries, str(train_path))
    save_jsonl(val_entries, str(val_path))

    print(f"\n{'=' * 50}")
    print(f"Output saved to: {output_path.absolute()}")
    print(f"  Train: {train_path.name} ({len(train_entries)} entries)")
    print(f"  Val:   {val_path.name} ({len(val_entries)} entries)")

    # 保存配置信息
    config = {
        "input_files": args.input,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "sampling_strategy": args.sampling_strategy,
        "total_count": len(all_entries),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
    }

    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config: {config_path.name}")


if __name__ == "__main__":
    main()
"""
# 默认：每个 question 只采样第一个 thought
python3 build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite \
    --train-ratio 0.98 \
    --seed 42

# 采样所有 thoughts
python3 build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite_all \
    --sampling-strategy all \
    --train-ratio 0.98 \
    --seed 42

# 采样 FSF 最高的 thought
python3 build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite_max_fsf \
    --sampling-strategy max_fsf \
    --train-ratio 0.98 \
    --seed 42

# 采样 FSF 最低的 thought
python3 build_cot_rewrite_dataset.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/cot_rewrite_min_fsf \
    --sampling-strategy min_fsf \
    --train-ratio 0.98 \
    --seed 42
"""
