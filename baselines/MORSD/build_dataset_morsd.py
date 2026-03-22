#!/usr/bin/env python3
"""
MORSD: 基于 RD 的 Difficulty Selection 数据集构建脚本

根据论文，Difficulty Selection 选择每个 question 下 RD 最小的 k 个 reasoning。

输入格式（rd_calc.py 的输出）:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "solution": "...",
            "rds": [
                {
                    "model_name": "xxx",
                    "model_path": "...",
                    "ppl_with_rationale": xxx,
                    "ppl_without_rationale": xxx,
                    "rd": xxx
                }
            ]
        },
        ...
    ]
}

输出格式（LlamaFactory sharegpt 格式）:
{
    "conversations": [
        {"role": "user", "content": "<question>"},
        {"role": "assistant", "content": "<think>\\n{thought}\\n</think>\\n\\n{solution}"}
    ]
}
"""

import json
import argparse
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="MORSD Difficulty Selection 数据集构建")
    parser.add_argument("--input", "-i", required=True, help="输入 JSONL 文件路径（含 rds 字段）")
    parser.add_argument("--output", "-o", required=True, help="输出目录路径")
    parser.add_argument(
        "--model-name", "-m", required=True,
        help="使用哪个模型的 RD 进行筛选（对应 rds[].model_name）"
    )
    parser.add_argument(
        "--topk", "-k", type=int, default=1,
        help="每个 question 选择 RD 最小的 k 个 reasoning（默认: 1）"
    )
    parser.add_argument("--train-ratio", "-t", type=float, default=0.99, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--dataset-info", "-r", default=None,
        help="dataset_info.json 路径（可选，注册到 LlamaFactory）"
    )
    parser.add_argument(
        "--rd-threshold", type=float, default=None,
        help="RD 阈值，仅选择 RD < threshold 的 reasoning（可选）"
    )
    return parser.parse_args()


def load_data(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_rd_for_model(reasoning: Dict[str, Any], model_name: str) -> Optional[float]:
    """从 rds 列表中取指定模型的 rd 值"""
    for entry in reasoning.get("rds", []):
        if entry.get("model_name") == model_name:
            rd = entry.get("rd")
            if rd is not None and rd != float('inf'):
                return rd
    return None


def select_by_difficulty(
    reasoning_list: List[Dict[str, Any]],
    model_name: str,
    topk: int = 1,
    rd_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Difficulty Selection: 选择指定模型 RD 最小的 topk 个 reasoning
    """
    candidates = []
    for r in reasoning_list:
        rd = get_rd_for_model(r, model_name)
        if rd is not None:
            if rd_threshold is None or rd < rd_threshold:
                candidates.append((rd, r))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])
    return [r for _, r in candidates[:topk]]


def build_sharegpt_item(question: str, reasoning: Dict[str, Any]) -> Dict[str, Any]:
    """将一条 reasoning 转换为 sharegpt 格式"""
    thought = reasoning.get("thought", "").strip()
    solution = reasoning.get("solution", "").strip()
    assistant_content = f"<think>\n{thought}\n</think>\n\n{solution}"

    return {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def register_to_dataset_info(dataset_info_path: str, dataset_name: str, file_path: str) -> bool:
    """将数据集注册到 dataset_info.json"""
    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            relative_path = str(Path(file_path).resolve())

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

        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"数据集 '{dataset_name}' 已注册到 {dataset_info_path}")
        return True
    except Exception as e:
        logger.error(f"注册数据集失败: {e}")
        return False


def main():
    args = parse_args()
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  MORSD Difficulty Selection 数据集构建")
    logger.info("=" * 60)
    logger.info(f"输入: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"Top-K: {args.topk}")
    if args.rd_threshold:
        logger.info(f"RD 阈值: {args.rd_threshold}")

    data = load_data(args.input)
    logger.info(f"共 {len(data)} 条数据")

    # Difficulty Selection
    sharegpt_data = []
    skipped = 0

    for item in data:
        question = item["question"]
        reasoning_list = item.get("reasoning", [])

        selected = select_by_difficulty(reasoning_list, args.model_name, args.topk, args.rd_threshold)

        if not selected:
            skipped += 1
            continue

        for r in selected:
            sharegpt_data.append(build_sharegpt_item(question, r))

    logger.info(f"Difficulty Selection 完成: {len(sharegpt_data)} 条样本")
    if skipped > 0:
        logger.info(f"跳过 {skipped} 条数据（无模型 '{args.model_name}' 的有效 RD）")

    # 打乱并拆分
    random.shuffle(sharegpt_data)
    train_size = int(len(sharegpt_data) * args.train_ratio)
    train_data = sharegpt_data[:train_size]
    val_data = sharegpt_data[train_size:]

    logger.info(f"训练集: {len(train_data)} 条")
    logger.info(f"验证集: {len(val_data)} 条")

    # 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"训练集保存至: {train_path}")
    logger.info(f"验证集保存至: {val_path}")

    # 注册到 dataset_info.json
    if args.dataset_info:
        dataset_base_name = output_dir.name
        register_to_dataset_info(args.dataset_info, f"{dataset_base_name}_train", str(train_path))
        register_to_dataset_info(args.dataset_info, f"{dataset_base_name}_val", str(val_path))

    logger.info("完成!")


if __name__ == "__main__":
    main()


"""
使用示例:

# Step 1: 用不同学生模型计算 RD（可多次运行，增量追加）
python rd_calc.py \
    -i /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl \
    -o /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -m /data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct

python rd_calc.py \
    -i /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -o /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -m /path/to/another_model

# Step 2: 用指定模型的 RD 构建数据集
python build_dataset_morsd.py \
    -i /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -o /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_morsd_llama3_3b_30k \
    -m Llama-3.2-3B-Instruct \
    -k 1 \
    --train-ratio 0.99 \
    -r ../../LlamaFactory/data/dataset_info.json
    
python build_dataset_morsd.py \
    -i /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -o /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_morsd_llama3_1b_30k \
    -m Llama-3.2-1B-Instruct \
    -k 1 \
    --train-ratio 0.99 \
    -r ../../LlamaFactory/data/dataset_info.json
    
python build_dataset_morsd.py \
    -i /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl \
    -o /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_morsd_qwen2_5_7b_30k \
    -m Qwen2.5-7B-Instruct \
    -k 1 \
    --train-ratio 0.99 \
    -r ../../LlamaFactory/data/dataset_info.json
"""
