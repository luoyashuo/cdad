#!/usr/bin/env python3
"""
Build training dataset for FSF (Failed-Step Fraction) judgment task.

This script:
1. Reads JSONL files containing reasoning traces with reasoning_graph
2. For each thought, checks the FSF value from reasoning_graph
3. Generates training data with labels:
   - "Y" if fsf > 0 (contains failed attempts)
   - "N" if fsf == 0 (no failed attempts)
4. Shuffles and splits data into train/val sets

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
        {"role": "system", "content": "You are a helpful assistant that judges reasoning traces if they contain any fail attempt."},
        {"role": "user", "content": "<JUDGE_PROMPT>\n\nReasoning trace:\n<thought>"},
        {"role": "assistant", "content": "Y" or "N"}
    ],
    "question": "...",
    "label": "Y" or "N"
}

Usage:
    # Basic usage with single file
    python build_fsf_judge.py \\
        -i /path/to/train_merge_level1.jsonl \\
        -o /path/to/output_dir

    # Multiple input files
    python build_fsf_judge.py \\
        -i math.jsonl code.jsonl \\
        -o /path/to/output_dir

    # Custom split ratio and seed
    python build_fsf_judge.py \\
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


JUDGE_PROMPT_TEMPLATE = """MANDATORY OUTPUT FORMAT:
The very first character of your response must be "Y" or "N", followed immediately by a newline ("\n"), and then the Graphviz DOT code. No other text is allowed before or after.

Judgment Logic:
- Output "Y" if the graph contains ANY nodes with 'fillcolor=lightpink' (failed attempts).
- Output "N" if ALL nodes in the graph are 'fillcolor=lightblue' (successful steps).

Visualization Rules:
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

Reasoning trace:
{thought}"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build FSF judgment training dataset with shuffle and split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_fsf_judge.py \\
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \\
    -o ./fsf_judge_output

  # Multiple input files
  python build_fsf_judge.py \\
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \\
       /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \\
    -o ./fsf_judge_output

  # Custom train/val split
  python build_fsf_judge.py \\
    -i input.jsonl \\
    -o output_dir \\
    --train-ratio 0.95 \\
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
            "content": "You are a helpful assistant that judges reasoning traces if they contain any fail attempt.",
        },
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(thought=thought)},
        {"role": "assistant", "content": f"{label}\n{reasoning_graph}"},
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


def save_jsonl(entries: List[Dict[str, Any]], output_path: str):
    """保存数据到 JSONL 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    args = parse_arguments()

    # 处理所有输入文件
    all_entries = []
    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"Warning: Input file {input_path} does not exist, skipping...")
            continue

        entries = process_single_file(input_path)
        all_entries.extend(entries)
        print(f"Processed {input_path}: {len(entries)} entries")

    if not all_entries:
        print("No valid entries found!")
        return

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

    print(f"\n{'=' * 50}")
    print(f"Output saved to: {output_path.absolute()}")
    print(f"  Train: {train_path.name} ({len(train_entries)} entries)")
    print(f"  Val:   {val_path.name} ({len(val_entries)} entries)")

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
    print(f"  Config: {config_path.name}")


if __name__ == "__main__":
    main()
"""
python3 build_fsf_judge.py \
    -i /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_math_all/train_merge_level1.jsonl \
        /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/omnithought_code_all/train_merge_level1.jsonl \
    -o /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge \
    --train-ratio 0.98 \
    --seed 42
"""
