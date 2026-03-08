#!/usr/bin/env python3
"""
从 Bespoke-Stratos-17k 数据集中提取代码类问题并转换为 sharegpt 格式。

过滤逻辑:
  - 仅保留以指定代码提示前缀开头的问题
  - 替换 think/solution 标签为标准格式

输入格式 (Bespoke-Stratos-17k parquet):
{
  "conversations": [
    {"from": "human", "value": "<code_question_prefix>\n<question>"},
    {"from": "gpt",   "value": "<|begin_of_thought|>...<|end_of_thought|>...<|begin_of_solution|>...<|end_of_solution|>"}
  ]
}

输出格式 (sharegpt):
{
  "conversations": [
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<think>...</think>\n\n<solution>"}
  ]
}

使用方式:
  python build_bespoke_code.py
  python build_bespoke_code.py \
    --input  /fs/fast/u20240075/hf_home/datasets/bespokelabs/Bespoke-Stratos-17k/data/train-00000-of-00001.parquet \
    --output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/bespoke_code \
    --dataset-info ../LlamaFactory/data/dataset_info.json
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


CODE_QUESTION_PREFIX = (
    "Generate an executable Python function generated from the given prompt. "
    "The function should take stdin as input and print the output. "
    "Simply call the function after the definition."
)

DEFAULT_INPUT = (
    "/fs/fast/u20240075/hf_home/datasets/bespokelabs/"
    "Bespoke-Stratos-17k/data/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT    = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/bespoke_code"
DEFAULT_TOKENIZER = "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_MAX_TOKENS = 12000


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 Bespoke-Stratos-17k 提取代码问题并转换为 sharegpt 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"输入 parquet 文件路径（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"输出目录路径（默认: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.98,
        help="训练集比例（默认: 0.98）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    parser.add_argument(
        "--dataset-info",
        type=str,
        default="LlamaFactory/data/dataset_info.json",
        help="dataset_info.json 路径（默认: LlamaFactory/data/dataset_info.json）",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"tokenizer 路径，用于统计 token 长度（默认: {DEFAULT_TOKENIZER}）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"assistant response 最大 token 数，超过则过滤（默认: {DEFAULT_MAX_TOKENS}）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("需要 transformers: pip install transformers")

    logger.info(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    logger.info(f"  词表大小: {tokenizer.vocab_size}")
    return tokenizer


def get_response_token_lengths(
    data: List[Dict[str, Any]], tokenizer
) -> List[int]:
    """批量计算每条数据 assistant response 的 token 数。"""
    responses = []
    for item in data:
        for conv in item.get("conversations", []):
            if conv.get("role") == "assistant":
                responses.append(conv.get("content", ""))
                break
        else:
            responses.append("")

    # batch encode，不截断
    encodings = tokenizer(
        responses,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in encodings["input_ids"]]


def filter_by_token_length(
    data: List[Dict[str, Any]], tokenizer, max_tokens: int
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    过滤 response token 超过 max_tokens 的样本。

    Returns:
        (filtered_data, token_lengths_of_kept, n_filtered)
        token_lengths_of_kept: 保留样本的 token 长度列表（顺序与 filtered_data 对应）
    """
    logger.info(f"  计算 {len(data)} 条 response 的 token 长度（批量编码）...")
    token_lengths = get_response_token_lengths(data, tokenizer)

    filtered, kept_lens = [], []
    n_filtered = 0
    for item, tlen in zip(data, token_lengths):
        if tlen <= max_tokens:
            filtered.append(item)
            kept_lens.append(tlen)
        else:
            n_filtered += 1

    return filtered, kept_lens, n_filtered


def print_token_distribution(token_lengths: List[int], max_tokens: int):
    """打印 token 长度分布直方图（分桶统计）。"""
    if not token_lengths:
        return

    buckets = [512, 1024, 2048, 4096, 6144, max_tokens]
    labels  = [
        f"≤512", "513-1k", "1k-2k", "2k-4k", "4k-6k",
        f"6k-{max_tokens//1024}k", f">{max_tokens//1024}k（已过滤）"
    ]
    counts = [0] * (len(buckets) + 1)

    for tlen in token_lengths:
        placed = False
        for i, b in enumerate(buckets):
            if tlen <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    total = len(token_lengths)
    logger.info("  token 长度分布（assistant response）:")
    logger.info(f"    {'区间':<20} {'数量':>6}  {'占比':>6}  {'累计':>6}")
    logger.info(f"    {'-'*44}")
    cumsum = 0
    for label, cnt in zip(labels, counts):
        cumsum += cnt
        pct     = cnt    / total * 100
        cum_pct = cumsum / total * 100
        logger.info(f"    {label:<20} {cnt:>6,}  {pct:>5.1f}%  {cum_pct:>5.1f}%")

    sorted_lens = sorted(token_lengths)
    n = len(sorted_lens)
    percentiles = {50: sorted_lens[n//2], 75: sorted_lens[int(n*0.75)],
                   90: sorted_lens[int(n*0.90)], 95: sorted_lens[int(n*0.95)],
                   99: sorted_lens[int(n*0.99)]}
    logger.info("")
    logger.info("  token 长度百分位数:")
    for p, v in percentiles.items():
        logger.info(f"    p{p:<3}: {v:>7,} tokens")
    logger.info(f"    min : {min(token_lengths):>7,} tokens")
    logger.info(f"    max : {max(token_lengths):>7,} tokens")
    logger.info(f"    avg : {round(sum(token_lengths)/total):>7,} tokens")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_parquet(file_path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas: pip install pandas pyarrow")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    logger.info(f"加载 parquet 文件: {file_path}")
    df = pd.read_parquet(file_path)
    data = df.to_dict("records")
    logger.info(f"  共加载 {len(data)} 条记录")
    return data


# ---------------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------------

def process_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    处理单条记录，返回 sharegpt 格式，若不是代码问题则返回 None。
    """
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    question_raw = conversations[0].get("value", "")
    full_response = conversations[1].get("value", "")

    if not question_raw.startswith(CODE_QUESTION_PREFIX):
        return None

    # 去除前缀，保留纯题目文本
    question = question_raw[len(CODE_QUESTION_PREFIX):].strip()

    # 替换 Bespoke 专用标签为标准 <think> 标签
    full_response = (
        full_response
        .replace("<|begin_of_thought|>", "<think>")
        .replace("<|end_of_thought|>", "</think>")
        .replace("<|begin_of_solution|>\n\n", "")
        .replace("<|begin_of_solution|>", "")
        .replace("<|end_of_solution|>", "")
        .strip()
    )

    return {
        "conversations": [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": full_response},
        ]
    }


def process_all(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    处理所有记录，返回 (处理后数据, 统计信息)。
    """
    processed = []
    stats = {
        "total":       len(data),
        "code":        0,
        "non_code":    0,
        "invalid":     0,
    }

    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            stats["invalid"] += 1
            continue

        result = process_item(item)
        if result is not None:
            processed.append(result)
            stats["code"] += 1
        else:
            stats["non_code"] += 1

    return processed, stats


# ---------------------------------------------------------------------------
# 统计信息
# ---------------------------------------------------------------------------

def compute_content_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 question / response / think / solution 的长度统计。"""
    q_lens, r_lens, think_lens, sol_lens = [], [], [], []

    for item in data:
        convs = item.get("conversations", [])
        for conv in convs:
            role    = conv.get("role", "")
            content = conv.get("content", "")
            if role == "user":
                q_lens.append(len(content))
            elif role == "assistant":
                r_lens.append(len(content))
                # 拆分 <think> 和 solution 部分
                if "<think>" in content and "</think>" in content:
                    think_end = content.index("</think>") + len("</think>")
                    think_part = content[:think_end]
                    sol_part   = content[think_end:].strip()
                else:
                    think_part = ""
                    sol_part   = content
                think_lens.append(len(think_part))
                sol_lens.append(len(sol_part))

    def _stats(lst):
        if not lst:
            return {"min": 0, "max": 0, "avg": 0}
        return {
            "min": min(lst),
            "max": max(lst),
            "avg": round(sum(lst) / len(lst)),
        }

    return {
        "question":  _stats(q_lens),
        "response":  _stats(r_lens),
        "think":     _stats(think_lens),
        "solution":  _stats(sol_lens),
    }


def print_statistics(
    raw_stats:    Dict[str, int],
    train_data:   List[Dict[str, Any]],
    val_data:     List[Dict[str, Any]],
    token_lengths: List[int],
    n_tok_filtered: int,
    max_tokens:   int,
):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  统计信息")
    logger.info("=" * 60)

    # 原始数据过滤统计
    total    = raw_stats["total"]
    code     = raw_stats["code"]
    non_code = raw_stats["non_code"]
    invalid  = raw_stats["invalid"]
    logger.info(f"  原始记录总数      : {total:,}")
    logger.info(f"  代码类问题        : {code:,}  ({code/total*100:.1f}%)")
    logger.info(f"  非代码类问题      : {non_code:,}  ({non_code/total*100:.1f}%)")
    logger.info(f"  格式异常跳过      : {invalid:,}")
    logger.info(f"  超长过滤(>{max_tokens} tok): {n_tok_filtered:,}")
    logger.info("")

    # 训练/验证集划分
    kept = len(train_data) + len(val_data)
    logger.info(f"  最终保留          : {kept:,}")
    logger.info(f"  训练集            : {len(train_data):,}")
    logger.info(f"  验证集            : {len(val_data):,}")
    logger.info("")

    # 字符长度统计
    all_data = train_data + val_data
    content_stats = compute_content_stats(all_data)
    logger.info("  内容长度统计（字符数）:")
    for field, st in content_stats.items():
        logger.info(
            f"    {field:<10}: min={st['min']:>7,}  max={st['max']:>7,}  avg={st['avg']:>7,}"
        )
    logger.info("")

    # Token 长度分布
    print_token_distribution(token_lengths, max_tokens)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 数据集注册
# ---------------------------------------------------------------------------

def register_to_dataset_info(
    dataset_info_path: str, dataset_name: str, file_path: str
) -> bool:
    try:
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            relative_path = str(Path(file_path).resolve())
            logger.warning(f"使用绝对路径注册: {relative_path}")

        dataset_info[dataset_name] = {
            "file_name":  relative_path,
            "formatting": "sharegpt",
            "columns":    {"messages": "conversations"},
            "tags": {
                "role_tag":      "role",
                "content_tag":   "content",
                "user_tag":      "user",
                "assistant_tag": "assistant",
            },
        }

        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"  数据集 '{dataset_name}' 已注册到 {dataset_info_path}")
        return True
    except Exception as e:
        logger.error(f"注册失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Bespoke-Stratos 代码数据集构建工具")
    logger.info("=" * 60)
    logger.info(f"  输入文件    : {args.input}")
    logger.info(f"  输出目录    : {args.output}")
    logger.info(f"  tokenizer   : {args.tokenizer}")
    logger.info(f"  最大 token  : {args.max_tokens}")
    logger.info(f"  训练集比例  : {args.train_ratio}")
    logger.info(f"  随机种子    : {args.seed}")
    logger.info(f"  dataset_info: {args.dataset_info}")
    logger.info("=" * 60)

    # Step 1: 加载 tokenizer
    logger.info("\n[Step 1] 加载 tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 2: 加载数据
    logger.info("\n[Step 2] 加载 parquet 数据...")
    raw_data = load_parquet(args.input)

    # Step 3: 过滤代码类问题 & 转换格式
    logger.info("\n[Step 3] 过滤代码类问题并转换格式...")
    processed, raw_stats = process_all(raw_data)
    logger.info(
        f"  过滤完成: {raw_stats['code']:,} 条代码问题 / {raw_stats['total']:,} 条总记录"
        f"（跳过非代码 {raw_stats['non_code']:,} 条，格式异常 {raw_stats['invalid']:,} 条）"
    )

    if not processed:
        logger.error("没有可处理的代码类数据，退出。")
        return 1

    # Step 4: 按 token 长度过滤（> max_tokens 的样本）
    logger.info(f"\n[Step 4] 按 token 长度过滤（response > {args.max_tokens} tokens 的样本）...")
    processed, token_lengths, n_tok_filtered = filter_by_token_length(
        processed, tokenizer, args.max_tokens
    )
    logger.info(
        f"  过滤完成: 保留 {len(processed):,} 条，"
        f"过滤超长 {n_tok_filtered:,} 条"
    )

    if not processed:
        logger.error("过滤后没有剩余数据，退出。")
        return 1

    # Step 5: 打乱 & 划分训练/验证集
    logger.info("\n[Step 5] 打乱并划分训练集/验证集...")
    # 同步打乱 processed 和 token_lengths
    combined = list(zip(processed, token_lengths))
    random.shuffle(combined)
    processed, token_lengths = zip(*combined)
    processed     = list(processed)
    token_lengths = list(token_lengths)

    train_size = int(len(processed) * args.train_ratio)
    train_data = processed[:train_size]
    val_data   = processed[train_size:]
    logger.info(f"  训练集: {len(train_data):,} 条  |  验证集: {len(val_data):,} 条")

    # Step 6: 输出统计信息
    logger.info("\n[Step 6] 统计信息汇总...")
    print_statistics(raw_stats, train_data, val_data, token_lengths, n_tok_filtered, args.max_tokens)

    # Step 7: 保存
    logger.info("\n[Step 7] 保存输出文件...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"  已保存 {len(data):,} 条 -> {path}")

    # Step 8: 注册到 dataset_info.json
    logger.info("\n[Step 8] 注册数据集...")
    dataset_base = output_dir.name
    register_to_dataset_info(args.dataset_info, f"{dataset_base}_train", str(train_path))
    register_to_dataset_info(args.dataset_info, f"{dataset_base}_val",   str(val_path))

    logger.info("\n  全部完成。")
    return 0


if __name__ == "__main__":
    exit(main())
