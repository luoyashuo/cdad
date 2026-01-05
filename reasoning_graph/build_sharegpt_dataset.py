#!/usr/bin/env python3
"""
将 select_diff_data.py 筛选后的数据转换为 LLamaFactory 支持的 sharegpt 格式

输入格式 (select_diff_data.py 输出):
{
  "question": "...",
  "reasoning": [...],
  "selected_by_cognitive_difficulty": {"thought": "...", "full_response": "...", ...},
  "selected_by_reasoning_verbosity": {"thought": "...", "full_response": "...", ...},
  "selected_by_lowest_fsf": {"thought": "...", "full_response": "...", ...},
  ...
}

输出格式 (sharegpt):
{
  "conversations": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<thought>\\n\\n<solution>"}
  ]
}

使用方式:
  python build_sharegpt_dataset.py \\
    -i /path/to/input.jsonl \\
    -s selected_by_cognitive_difficulty \\
    -o /path/to/output.jsonl \\
    -r LlamaFactory/data/dataset_info.json
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any, List
import random
from loguru import logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="将筛选后的数据转换为 sharegpt 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用认知难度筛选数据，输出到指定目录
  python build_sharegpt_dataset.py \\
    -i /path/to/train_merge_level1.jsonl \\
    -s selected_by_cognitive_difficulty \\
    -o /path/to/output_dir

  # 使用最低 FSF 筛选数据
  python build_sharegpt_dataset.py \\
    -i /path/to/input.jsonl \\
    -s selected_by_lowest_fsf \\
    -o /path/to/output_dir

  # 指定 dataset_info.json 路径
  python build_sharegpt_dataset.py \\
    -i input.jsonl \\
    -s selected_by_reasoning_verbosity \\
    -o /path/to/output_dir \\
    -r /custom/path/dataset_info.json

  # 合并多个输入文件（所有文件会被合并后一起处理）
  python build_sharegpt_dataset.py \\
    -i /path/to/data1.jsonl /path/to/data2.jsonl /path/to/data3.jsonl \\
    -s selected_by_cognitive_difficulty \\
    -o /path/to/merged_output_dir
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        nargs='+',
        help="输入的 JSONL 文件路径（select_diff_data.py 的输出），可指定多个文件"
    )
    parser.add_argument(
        "--selector", "-s",
        required=True,
        choices=["selected_by_cognitive_difficulty", "selected_by_lowest_fsf", "selected_by_reasoning_verbosity"],
        help="选择使用哪个筛选字段"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录路径（将生成 train.jsonl 和 val.jsonl）"
    )
    parser.add_argument(
        "--dataset-info", "-r",
        default="LlamaFactory/data/dataset_info.json",
        help="dataset_info.json 的路径（默认: LlamaFactory/data/dataset_info.json）"
    )
    parser.add_argument(
        "--train-ratio", "-t",
        type=float,
        default=0.98,
        help="训练集比例（默认: 0.98）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    return parser.parse_args()


def convert_to_sharegpt(item: Dict[str, Any], selector: str) -> Dict[str, Any]:
    """
    将单个数据项转换为 sharegpt 格式

    Args:
        item: 输入数据项
        selector: 选择器字段名

    Returns:
        sharegpt 格式的数据项
    """
    question = item.get("question", "")
    selected_data = item.get(selector, {})

    thought = selected_data.get("thought", "")
    solution = selected_data.get("full_response", "")

    # 构建 assistant 的内容：thought + solution
    assistant_content = f"{thought}\n\n{solution}".strip()

    # 构建 sharegpt 格式
    sharegpt_item = {
        "conversations": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }

    return sharegpt_item


def register_to_dataset_info(dataset_info_path: str, dataset_name: str, file_path: str) -> bool:
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
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        # 计算相对路径（相对于 LlamaFactory/data 目录）
        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            # 如果文件不在 LlamaFactory/data 目录下，使用绝对路径
            relative_path = str(Path(file_path).resolve())
            logger.warning(f"文件不在 LlamaFactory/data 目录下，使用绝对路径: {relative_path}")

        # 添加到 dataset_info
        dataset_info[dataset_name] = {
            "file_name": relative_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }

        # 写回 dataset_info.json
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 数据集 '{dataset_name}' 已注册到 {dataset_info_path}")
        return True

    except Exception as e:
        logger.error(f"注册数据集失败: {e}")
        return False


def main():
    args = parse_arguments()
    
    # Handle case where input might be a single string with spaces (from shell script)
    if len(args.input) == 1 and ' ' in args.input[0]:
        # Split the string by spaces to get individual file paths
        args.input = args.input[0].split()

    logger.info("="*70)
    logger.info("  数据格式转换工具")
    logger.info("="*70)
    if len(args.input) == 1:
        logger.info(f"📁 输入文件: {args.input[0]}")
    else:
        logger.info(f"📁 输入文件 ({len(args.input)} 个):")
        for idx, input_file in enumerate(args.input, 1):
            logger.info(f"   {idx}. {input_file}")
    logger.info(f"🎯 选择器: {args.selector}")
    logger.info(f"📁 输出文件: {args.output}")
    logger.info(f"📝 dataset_info: {args.dataset_info}")
    logger.info(f"📊 训练集比例: {args.train_ratio}")
    logger.info(f"🎲 随机种子: {args.seed}")
    logger.info("="*70)

    # 检查输入文件
    for input_file in args.input:
        if not os.path.exists(input_file):
            logger.error(f"输入文件 '{input_file}' 不存在！")
            return 1

    # 读取输入数据
    logger.info("📖 读取输入数据...")
    data = []
    total_files = len(args.input)
    
    for file_idx, input_file in enumerate(args.input, 1):
        if total_files > 1:
            logger.info(f"  读取文件 {file_idx}/{total_files}: {input_file}")
        
        file_data_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                    file_data_count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"文件 {input_file} 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
        
        if total_files > 1:
            logger.info(f"    ✅ 读取 {file_data_count} 条数据")

    total_items = len(data)
    logger.info(f"✅ 读取完成，共 {total_items} 条数据")

    if total_items == 0:
        logger.error("没有可处理的数据！")
        return 1

    # 转换为 sharegpt 格式
    logger.info("🔄 转换为 sharegpt 格式...")
    sharegpt_data = []
    for item in data:
        try:
            sharegpt_item = convert_to_sharegpt(item, args.selector)
            sharegpt_data.append(sharegpt_item)
        except Exception as e:
            logger.warning(f"转换数据失败: {e}")
            continue

    logger.info(f"✅ 转换完成，共 {len(sharegpt_data)} 条数据")

    # 随机打乱并拆分训练集/验证集
    logger.info("🔀 拆分训练集和验证集...")
    random.seed(args.seed)
    random.shuffle(sharegpt_data)

    train_size = int(len(sharegpt_data) * args.train_ratio)
    train_data = sharegpt_data[:train_size]
    val_data = sharegpt_data[train_size:]

    logger.info(f"📊 训练集: {len(train_data)} 条")
    logger.info(f"📊 验证集: {len(val_data)} 条")

    # 处理输出路径（目录或文件）
    output_path = Path(args.output)
    
    # 统一逻辑：在输出目录下生成 train.jsonl 和 val.jsonl
    if output_path.is_dir() or (not output_path.suffix and not output_path.name.endswith('.jsonl')):
        # 如果指定的是目录
        output_dir = output_path
    else:
        # 如果指定的是文件，则使用其所在目录
        output_dir = output_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    # 保存训练集
    logger.info(f"💾 保存训练集: {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存验证集
    logger.info(f"💾 保存验证集: {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info("✅ 数据保存完成")

    # 注册到 dataset_info.json
    logger.info("📝 注册数据集到 dataset_info.json...")

    # 生成数据集名称（基于输出目录名）
    dataset_base_name = output_dir.name
    train_dataset_name = f"{dataset_base_name}_train"
    val_dataset_name = f"{dataset_base_name}_val"

    # 注册训练集
    train_registered = register_to_dataset_info(
        args.dataset_info,
        train_dataset_name,
        str(train_path)
    )

    # 注册验证集
    val_registered = register_to_dataset_info(
        args.dataset_info,
        val_dataset_name,
        str(val_path)
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
    exit(main())

"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl  \
    -s selected_by_cognitive_difficulty \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_cg \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl  \
    -s selected_by_reasoning_verbosity \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_rv \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl  \
    -s selected_by_lowest_fsf \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_fsf \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""