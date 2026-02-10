#!/usr/bin/env python3
"""
Build DPO training dataset based on FSF (Failed-Step Fraction) values.

This script:
1. Reads JSONL files containing reasoning traces with reasoning_graph
2. For each question, finds the thought with highest FSF (rejected) and lowest FSF (chosen)
3. Generates DPO training data in LlamaFactory format

Input format:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "full_response": "...",
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

Output format (LlamaFactory DPO):
{
    "conversations": [
        {"role": "user", "content": "<question>"}
    ],
    "chosen": {"role": "assistant", "content": "<thought>\n\n<solution>"},
    "rejected": {"role": "assistant", "content": "<thought>\n\n<solution>"}
}

Usage:
    # Basic usage with single file
    python build_fsf_reward.py \\
        -i /path/to/train_merge_level1.jsonl \\
        -o /path/to/output_dir

    # Multiple input files
    python build_fsf_reward.py \\
        -i math.jsonl code.jsonl \\
        -o /path/to/output_dir

    # Custom split ratio and seed
    python build_fsf_reward.py \\
        -i input.jsonl \\
        -o output_dir \\
        --train-ratio 0.95 \\
        --seed 123
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from loguru import logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build DPO dataset based on FSF values (reject=high FSF, chosen=low FSF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_fsf_reward.py \\
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \\
    -o ./fsf_reward_output

  # Multiple input files
  python build_fsf_reward.py \\
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \\
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \\
    -o ./fsf_reward_output

  # Custom train/val split
  python build_fsf_reward.py \\
    -i input.jsonl \\
    -o output_dir \\
    --train-ratio 0.98 \\
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
        "--dataset-info",
        "-r",
        default="LlamaFactory/data/dataset_info.json",
        help="dataset_info.json 的路径（默认: LlamaFactory/data/dataset_info.json）",
    )
    parser.add_argument(
        "--min-fsf-gap",
        "-g",
        type=float,
        default=0.0,
        help="FSF 差值的最小值（默认: 0.0）。只有当 rejected 和 chosen 的 FSF 差值大于等于此值时才会保留该条目",
    )

    return parser.parse_args()


def get_fsf_from_thought(thought: Dict[str, Any]) -> Optional[float]:
    """
    从 thought 中提取 FSF 值

    Args:
        thought: 包含 reasoning_graph 的 thought 字典

    Returns:
        FSF 值，如果无法获取则返回 None
    """
    reasoning_graph_list = thought.get("reasoning_graph", [])
    if not reasoning_graph_list or not isinstance(reasoning_graph_list, list):
        return None

    first_graph = reasoning_graph_list[0]
    if not isinstance(first_graph, dict):
        return None

    return first_graph.get("fsf", 0.0)


def build_dpo_entry(
    question: str, chosen_thought: Dict[str, Any], rejected_thought: Dict[str, Any]
) -> Dict[str, Any]:
    """
    构建 DPO 训练数据条目

    Args:
        question: 问题文本
        chosen_thought: 被选中的 thought（FSF 最低）
        rejected_thought: 被拒绝的 thought（FSF 最高）

    Returns:
        DPO 格式的数据条目
    """
    # 构建 chosen 内容：thought
    chosen_content = f"{chosen_thought.get('thought', '')}".strip()

    # 构建 rejected 内容：thought
    rejected_content = f"{rejected_thought.get('thought', '')}".strip()

    return {
        "instruction": question.strip(),
        "chosen": chosen_content.strip(),
        "rejected": rejected_content.strip(),
        "question": question,
        "chosen_fsf": get_fsf_from_thought(chosen_thought),
        "rejected_fsf": get_fsf_from_thought(rejected_thought),
    }


def process_single_file(
    input_path: str, min_fsf_gap: float = 0.0
) -> List[Dict[str, Any]]:
    """
    处理单个输入文件，生成 DPO 训练数据

    Args:
        input_path: 输入 JSONL 文件路径
        min_fsf_gap: FSF 差值的最小值。只有当 rejected 和 chosen 的 FSF 差值大于等于此值时才会保留该条目

    Returns:
        DPO 训练数据条目列表
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

            if not question or not reasoning_list or len(reasoning_list) < 2:
                continue

            # 收集所有带有 FSF 的 thought
            thoughts_with_fsf = []
            for thought_data in reasoning_list:
                if not isinstance(thought_data, dict):
                    continue

                thought_text = thought_data.get("thought", "")
                if not thought_text:
                    continue

                fsf = get_fsf_from_thought(thought_data)
                if fsf is None:
                    continue

                thoughts_with_fsf.append((thought_data, fsf))

            if len(thoughts_with_fsf) < 2:
                continue

            # 按 FSF 排序，找到最低和最高的
            thoughts_with_fsf.sort(key=lambda x: x[1])

            chosen_thought = thoughts_with_fsf[0][0]  # FSF 最低的
            rejected_thought = thoughts_with_fsf[-1][0]  # FSF 最高的

            # 检查 FSF 差值是否满足最小要求
            chosen_fsf = get_fsf_from_thought(chosen_thought)
            rejected_fsf = get_fsf_from_thought(rejected_thought)

            if (
                chosen_fsf is None
                or rejected_fsf is None
                or (rejected_fsf - chosen_fsf) < min_fsf_gap
            ):
                continue

            # 构建 DPO 条目
            entry = build_dpo_entry(question, chosen_thought, rejected_thought)
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
    total = len(entries)

    if total == 0:
        print(f"\n{name} Statistics: 0 entries")
        return

    avg_chosen_fsf = sum(e["chosen_fsf"] for e in entries) / total
    avg_rejected_fsf = sum(e["rejected_fsf"] for e in entries) / total
    avg_fsf_gap = sum(e["rejected_fsf"] - e["chosen_fsf"] for e in entries) / total

    print(f"\n{name} Statistics:")
    print(f"  Total: {total}")
    print(f"  Average chosen FSF (lowest): {avg_chosen_fsf:.4f}")
    print(f"  Average rejected FSF (highest): {avg_rejected_fsf:.4f}")
    print(f"  Average FSF gap: {avg_fsf_gap:.4f}")


def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """保存数据到 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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

        # 使用绝对路径
        absolute_path = str(Path(file_path).resolve())

        # 添加到 dataset_info - 使用标准 ranking 格式
        dataset_info[dataset_name] = {
            "file_name": absolute_path,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "chosen": "chosen",
                "rejected": "rejected",
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


def main():
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("  构建 FSF-based DPO 数据集")
    logger.info("=" * 70)
    logger.info(f"📁 输入文件: {args.input}")
    logger.info(f"📁 输出目录: {args.output}")
    logger.info(f"📊 训练集比例: {args.train_ratio}")
    logger.info(f"🎲 随机种子: {args.seed}")
    logger.info(f"📏 最小 FSF 差值: {args.min_fsf_gap}")
    logger.info("=" * 70)

    # 处理所有输入文件
    all_entries = []
    for input_path in args.input:
        if not os.path.exists(input_path):
            logger.warning(f"输入文件 {input_path} 不存在，跳过...")
            continue

        entries = process_single_file(input_path, args.min_fsf_gap)
        all_entries.extend(entries)
        logger.info(f"处理 {input_path}: {len(entries)} 条数据")

    if not all_entries:
        logger.error("没有有效的数据条目！")
        return 1

    print_statistics(all_entries, "Total Data")

    # 划分训练集和验证集
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
    logger.info(f"✅ 输出已保存到: {output_path.absolute()}")
    logger.info(f"  Train: {train_path.name} ({len(train_entries)} entries)")
    logger.info(f"  Val:   {val_path.name} ({len(val_entries)} entries)")

    # 保存配置信息
    config = {
        "input_files": args.input,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "min_fsf_gap": args.min_fsf_gap,
        "total_count": len(all_entries),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
    }

    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"  Config: {config_path.name}")

    # 注册到 dataset_info.json
    logger.info("📝 注册数据集到 dataset_info.json...")

    dataset_base_name = output_path.name
    train_dataset_name = f"{dataset_base_name}_train"
    val_dataset_name = f"{dataset_base_name}_val"

    train_registered = register_to_dataset_info(
        args.dataset_info, train_dataset_name, str(train_path)
    )

    val_registered = register_to_dataset_info(
        args.dataset_info, val_dataset_name, str(val_path)
    )

    if train_registered and val_registered:
        logger.info("🎉 数据集注册成功！")
        logger.info(f"   训练集名称: {train_dataset_name}")
        logger.info(f"   验证集名称: {val_dataset_name}")
    else:
        logger.error("数据集注册失败！")

    return 0


if __name__ == "__main__":
    exit(main())

"""
python3 build_fsf_reward.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
        /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_reward \
    --train-ratio 0.98 \
    --seed 42 \
    --min-fsf-gap 0.2 \
    -r ../LlamaFactory/data/dataset_info.json
"""
