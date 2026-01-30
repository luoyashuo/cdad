#!/usr/bin/env python3
"""
Build training dataset for reasoning graph generation task.

This script:
1. Reads JSONL files containing reasoning traces with reasoning_graph
2. For each thought, extracts the reasoning trace and corresponding graph
3. Generates training data in sharegpt format for graph generation
4. Shuffles and splits data into train/val sets
5. Registers the dataset to LlamaFactory's dataset_info.json

Input format:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "reasoning_graph": [
                {
                    "reasoning_graph": "...",
                    "fsf": 0.2,
                    ...
                }
            ],
            ...
        },
        ...
    ]
}

Output format (sharegpt style):
{
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."},
        {"role": "user", "content": "<GRAPH_GENERATION_PROMPT>\n\nReasoning trace:\n<thought>"},
        {"role": "assistant", "content": "<reasoning_graph>"}
    ],
    "question": "...",
    "label": "Y" or "N"
}

Usage:
    # Basic usage with single file
    python build_fsf_judge_graph.py \
        -i /path/to/train_merge_level1.jsonl \
        -o /path/to/output_dir \
        -r LlamaFactory/data/dataset_info.json

    # Multiple input files
    python build_fsf_judge_graph.py \
        -i math.jsonl code.jsonl \
        -o /path/to/output_dir \
        -r LlamaFactory/data/dataset_info.json

    # Custom split ratio and seed
    python build_fsf_judge_graph.py \
        -i input.jsonl \
        -o output_dir \
        --train-ratio 0.95 \
        --seed 123 \
        -r LlamaFactory/data/dataset_info.json
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from loguru import logger


GRAPH_GENERATION_PROMPT = """
Parse the reasoning trace into a Graphviz diagram. Focus on these essentials:

Node Rules:
- One node per distinct reasoning step
- 'fillcolor=lightblue': Successful reasoning steps
- 'fillcolor=lightpink': Failed attempts

Edge Rules:
- Connect node A → node B if the information or insight from A is actually used to construct the reasoning in B; branch new attempts from their starting ancestor, not from dead ends.

Requirements:
- Use 'rankdir=TB'
- Include ALL attempts (including failures), do not miss any steps in the reasoning.
- ALWAYS start with a "problem statement" node
- ALWAYS end with a "final answer" node
- Do NOT reorder or reorganize the reasoning flow

Generate complete Graphviz DOT code in dot blocks.
""".strip()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build reasoning graph generation training dataset with shuffle and split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
    -o ./graph_generator_output \
    -r LlamaFactory/data/dataset_info.json

  # Multiple input files
  python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o ./graph_generator_output \
    -r LlamaFactory/data/dataset_info.json

  # Custom train/val split
  python build_fsf_judge_graph.py \
    -i input.jsonl \
    -o output_dir \
    --train-ratio 0.95 \
    --seed 42 \
    -r LlamaFactory/data/dataset_info.json
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
        "--dataset-info",
        "-r",
        default="LlamaFactory/data/dataset_info.json",
        help="dataset_info.json 的路径（默认: LlamaFactory/data/dataset_info.json）",
    )
    parser.add_argument(
        "--train-ratio", "-t", type=float, default=0.98, help="训练集比例（默认: 0.98）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")

    return parser.parse_args()


def get_fsf_label(thought: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    根据 thought 的 reasoning_graph 判断 FSF 标签

    Args:
        thought: 包含 reasoning_graph 的 thought 字典

    Returns:
        (label, reasoning_graph) 元组，其中 label 是 "Y" 或 "N"，
        reasoning_graph 是对应的图形数据，None if cannot determine
    """
    reasoning_graph_list = thought.get("reasoning_graph", [])
    if not reasoning_graph_list or not isinstance(reasoning_graph_list, list):
        return None

    # 获取第一个 reasoning_graph 元素的 fsf 值
    first_graph = reasoning_graph_list[0]
    if not isinstance(first_graph, dict):
        return None

    fsf = first_graph.get("fsf", 0.0)
    reasoning_graph = first_graph["reasoning_graph"]

    # 如果 fsf > 0，表示有失败步骤，输出 "Y"
    # 否则输出 "N"
    label = "Y" if fsf > 0 else "N"
    return (label, reasoning_graph)


def build_training_entry(
    question: str, thought: str, label: str, reasoning_graph: str
) -> Dict[str, Any]:
    """
    构建训练数据条目（sharegpt 格式）

    Args:
        question: 问题文本
        thought: 思考过程文本
        label: 标签 ("Y" 或 "N")
        reasoning_graph: reasoning_graph 图形数据

    Returns:
        sharegpt 格式的对话条目
    """
    conversations = [
        {
            "role": "system",
            "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams.",
        },
        {
            "role": "user",
            "content": f"{GRAPH_GENERATION_PROMPT}\n\nReasoning trace:\n{thought}",
        },
        {"role": "assistant", "content": f"{reasoning_graph}"},
    ]

    return {"conversations": conversations, "question": question, "label": label}


def process_single_file(input_path: str) -> List[Dict[str, Any]]:
    """
    处理单个输入文件，生成训练数据

    Args:
        input_path: 输入 JSONL 文件路径

    Returns:
        训练数据条目列表
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

            # 处理每个 thought
            for thought_data in reasoning_list:
                if not isinstance(thought_data, dict):
                    continue

                thought_text = thought_data.get("thought", "")
                if not thought_text:
                    continue

                # 获取 FSF 标签和 reasoning_graph
                result = get_fsf_label(thought_data)
                if result is None:
                    continue

                label, reasoning_graph = result

                # 构建训练条目
                entry = build_training_entry(
                    question, thought_text, label, reasoning_graph
                )
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


def print_statistics(entries: List[Dict[str, Any]], name: str = "Dataset"):
    """打印数据统计信息"""
    y_count = sum(1 for e in entries if e["label"] == "Y")
    n_count = sum(1 for e in entries if e["label"] == "N")
    total = len(entries)

    print(f"\n{name} Statistics:")
    print(f"  Total: {total}")
    if total > 0:
        print(f"  Y (has failure): {y_count} ({100 * y_count / total:.2f}%)")
        print(f"  N (no failure):  {n_count} ({100 * n_count / total:.2f}%)")


def register_to_dataset_info(
    dataset_info_path: str, dataset_name: str, file_path: str
) -> bool:
    """
    将数据集注册到 dataset_info.json

    Args:
        dataset_info_path: dataset_info.json 的路径
        dataset_name: 数据集名称
        file_path: 数据集文件路径

    Returns:
        是否成功注册
    """
    try:
        # 读取 dataset_info.json
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        # 计算相对路径（相对于 LlamaFactory/data 目录）
        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            # 如果文件不在 LlamaFactory/data 目录下，使用绝对路径
            relative_path = str(Path(file_path).resolve())
            logger.warning(
                f"文件不在 LlamaFactory/data 目录下，使用绝对路径: {relative_path}"
            )

        # 添加到 dataset_info
        dataset_info[dataset_name] = {
            "file_name": relative_path,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }

        # 写回 dataset_info.json
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 数据集 '{dataset_name}' 已注册到 {dataset_info_path}")
        return True

    except Exception as e:
        logger.error(f"注册数据集失败: {e}")
        return False


def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """保存数据到 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    args = parse_arguments()

    # 设置随机种子
    random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("  推理图生成数据集构建工具")
    logger.info("=" * 70)
    if len(args.input) == 1:
        logger.info(f"📁 输入文件: {args.input[0]}")
    else:
        logger.info(f"📁 输入文件 ({len(args.input)} 个):")
        for idx, input_file in enumerate(args.input, 1):
            logger.info(f"   {idx}. {input_file}")
    logger.info(f"📁 输出目录: {args.output}")
    logger.info(f"📝 dataset_info: {args.dataset_info}")
    logger.info(f"📊 训练集比例: {args.train_ratio}")
    logger.info(f"🎲 随机种子: {args.seed}")
    logger.info("=" * 70)

    # 检查输入文件
    for input_file in args.input:
        if not os.path.exists(input_file):
            logger.error(f"输入文件 '{input_file}' 不存在！")
            return 1

    # 处理所有输入文件
    all_entries = []
    for input_path in args.input:
        entries = process_single_file(input_path)
        all_entries.extend(entries)
        logger.info(f"✅ 处理完成 {input_path}: {len(entries)} 条数据")

    if not all_entries:
        logger.error("没有找到有效的数据条目！")
        return 1

    print_statistics(all_entries, "Total Data")

    # 划分训练集和验证集（按原始比例随机划分）
    train_entries, val_entries = split_train_val(
        all_entries, args.train_ratio, args.seed
    )

    # 打印划分后的统计
    print_statistics(train_entries, "Train Set")
    print_statistics(val_entries, "Validation Set")

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    # 保存数据
    save_jsonl(train_entries, str(train_path))
    save_jsonl(val_entries, str(val_path))

    logger.info(f"\n{'=' * 50}")
    logger.info(f"输出保存到: {output_path.absolute()}")
    logger.info(f"  训练集: {train_path.name} ({len(train_entries)} 条数据)")
    logger.info(f"  验证集: {val_path.name} ({len(val_entries)} 条数据)")

    # 保存配置信息
    config = {
        "input_files": args.input,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_count": len(all_entries),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "train_y_count": sum(1 for e in train_entries if e["label"] == "Y"),
        "train_n_count": sum(1 for e in train_entries if e["label"] == "N"),
        "val_y_count": sum(1 for e in val_entries if e["label"] == "Y"),
        "val_n_count": sum(1 for e in val_entries if e["label"] == "N"),
    }

    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"  配置文件: {config_path.name}")

    # 注册到 dataset_info.json
    logger.info("📝 注册数据集到 dataset_info.json...")

    # 生成数据集名称（基于输出目录名）
    dataset_base_name = output_path.name
    train_dataset_name = f"{dataset_base_name}_train"
    val_dataset_name = f"{dataset_base_name}_val"

    # 注册训练集
    train_registered = register_to_dataset_info(
        args.dataset_info, train_dataset_name, str(train_path)
    )

    # 注册验证集
    val_registered = register_to_dataset_info(
        args.dataset_info, val_dataset_name, str(val_path)
    )

    if train_registered and val_registered:
        logger.info("🎉 数据集注册成功！")
        logger.info(f"   训练集名称: {train_dataset_name}")
        logger.info(f"   验证集名称: {val_dataset_name}")
        return 0
    else:
        logger.error("数据集注册失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# ============================================================================
# 使用示例
# ============================================================================

# 1. 基本使用（单个文件）
"""
python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/graph_generator \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 2. 多个输入文件
"""
python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/graph_generator \
    --train-ratio 0.98 \
    --seed 42 \
    -r ../LlamaFactory/data/dataset_info.json
"""
python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/graph_generator \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 2. 多个输入文件
"""
python build_fsf_judge_graph.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/graph_generator \
    --train-ratio 0.98 \
    --seed 42 \
    -r ../LlamaFactory/data/dataset_info.json
"""
