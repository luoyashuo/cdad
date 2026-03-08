#!/usr/bin/env python3
"""
从 KodCode-V1-SFT-R1 数据集构建代码训练数据并转换为 sharegpt 格式。

输入格式 (KodCode parquet):
{
  "conversations": [
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<think>...</think>\n\n<solution>"}
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
  python build_kodcode.py
  python build_kodcode.py \
    --input  /fs/fast/u20240075/hf_home/datasets/KodCode/KodCode-V1-SFT-R1/data \
    --output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/kodcode_10k \
    --dataset-info ../LlamaFactory/data/dataset_info.json
"""

import json
import argparse
import os
import glob
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


DEFAULT_INPUT     = "/fs/fast/u20240075/hf_home/datasets/KodCode/KodCode-V1-SFT-R1/data"
DEFAULT_OUTPUT    = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/kodcode"
DEFAULT_TOKENIZER = "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_MAX_TOKENS  = 8192
DEFAULT_MAX_SAMPLES = 10000
TOKENIZE_BATCH_SIZE = 2000  # 分批 tokenization 的批大小，用于早停


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 KodCode-V1-SFT-R1 构建代码数据集并转换为 sharegpt 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"输入数据目录（会 glob train*.parquet，默认: {DEFAULT_INPUT}）",
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"最大采样数量上限，达到后停止处理（默认: {DEFAULT_MAX_SAMPLES}）",
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
    """对每条数据的完整 conversation 应用 chat template 后计算 token 数。"""
    from tqdm import tqdm
    lengths = []
    for item in tqdm(data, desc="tokenizing", unit="sample"):
        messages = item.get("conversations", [])
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            lengths.append(len(ids))
        except Exception:
            lengths.append(0)
    return lengths


def filter_by_token_length_with_limit(
    data: List[Dict[str, Any]],
    tokenizer,
    max_tokens: int,
    max_samples: int,
    batch_size: int = TOKENIZE_BATCH_SIZE,
) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    分批计算 token 长度，过滤超长样本，达到 max_samples 后提前停止。

    Returns:
        (filtered_data, token_lengths_of_kept, n_filtered)
    """
    result_data: List[Dict[str, Any]] = []
    result_lens: List[int] = []
    n_filtered = 0

    logger.info(
        f"  分批 tokenize（batch={batch_size}），"
        f"收集满 {max_samples:,} 条有效样本后停止..."
    )

    for start in range(0, len(data), batch_size):
        batch = data[start: start + batch_size]
        batch_lens = get_response_token_lengths(batch, tokenizer)

        for item, tlen in zip(batch, batch_lens):
            if tlen <= max_tokens:
                result_data.append(item)
                result_lens.append(tlen)
                if len(result_data) >= max_samples:
                    # 统计本 batch 剩余的超长条数（近似，不继续 tokenize）
                    logger.info(
                        f"  已收集 {len(result_data):,} 条有效样本，提前停止。"
                    )
                    return result_data, result_lens, n_filtered
            else:
                n_filtered += 1

        logger.info(
            f"  进度: 已处理 {min(start + batch_size, len(data)):,} / {len(data):,} 条，"
            f"当前有效 {len(result_data):,} 条"
        )

    return result_data, result_lens, n_filtered


def print_token_distribution(token_lengths: List[int], max_tokens: int):
    """打印 token 长度分布直方图（分桶统计）。"""
    if not token_lengths:
        return

    buckets = [512, 1024, 2048, 4096, 6144, max_tokens]
    labels  = [
        "≤512", "513-1k", "1k-2k", "2k-4k", "4k-6k",
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

def load_parquet_files(input_dir: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas: pip install pandas pyarrow")

    pattern = os.path.join(input_dir, "train*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")

    logger.info(f"发现 {len(files)} 个 parquet 文件:")
    all_data = []
    for f in files:
        logger.info(f"  加载: {f}")
        df = pd.read_parquet(f)
        records = df.to_dict("records")
        logger.info(f"    {len(records):,} 条记录")
        all_data.extend(records)

    logger.info(f"  共加载 {len(all_data):,} 条记录")
    return all_data


# ---------------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------------

def process_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    处理单条记录，转换为 sharegpt 格式；格式异常则返回 None。
    """
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        return None

    question      = conversations[0].get("value", "").strip()
    full_response = conversations[1].get("value", "").strip()

    if not question or not full_response:
        return None

    return {
        "conversations": [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": full_response},
        ]
    }


def process_all(
    data: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    将所有原始记录转换为 sharegpt 格式，返回 (转换后数据, 统计信息)。
    """
    processed = []
    stats = {
        "total":   len(data),
        "valid":   0,
        "invalid": 0,
    }

    for item in data:
        result = process_item(item)
        if result is not None:
            processed.append(result)
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    return processed, stats


# ---------------------------------------------------------------------------
# 统计信息
# ---------------------------------------------------------------------------

def compute_content_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 question / response / think / solution 的字符长度统计。"""
    q_lens, r_lens, think_lens, sol_lens = [], [], [], []

    for item in data:
        for conv in item.get("conversations", []):
            role    = conv.get("role", "")
            content = conv.get("content", "")
            if role == "user":
                q_lens.append(len(content))
            elif role == "assistant":
                r_lens.append(len(content))
                if "<think>" in content and "</think>" in content:
                    think_end  = content.index("</think>") + len("</think>")
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
    raw_stats:      Dict[str, int],
    train_data:     List[Dict[str, Any]],
    val_data:       List[Dict[str, Any]],
    token_lengths:  List[int],
    n_tok_filtered: int,
    max_tokens:     int,
    max_samples:    int,
):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  统计信息")
    logger.info("=" * 60)

    total   = raw_stats["total"]
    valid   = raw_stats["valid"]
    invalid = raw_stats["invalid"]
    logger.info(f"  原始记录总数      : {total:,}")
    logger.info(f"  格式有效          : {valid:,}  ({valid/total*100:.1f}%)")
    logger.info(f"  格式异常跳过      : {invalid:,}")
    logger.info(f"  超长过滤(>{max_tokens} tok): {n_tok_filtered:,}")
    logger.info(f"  采样上限          : {max_samples:,}")
    logger.info("")

    kept = len(train_data) + len(val_data)
    logger.info(f"  最终保留          : {kept:,}")
    logger.info(f"  训练集            : {len(train_data):,}")
    logger.info(f"  验证集            : {len(val_data):,}")
    logger.info("")

    all_data = train_data + val_data
    content_stats = compute_content_stats(all_data)
    logger.info("  内容长度统计（字符数）:")
    for field, st in content_stats.items():
        logger.info(
            f"    {field:<10}: min={st['min']:>7,}  max={st['max']:>7,}  avg={st['avg']:>7,}"
        )
    logger.info("")

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
    logger.info("  KodCode 代码数据集构建工具")
    logger.info("=" * 60)
    logger.info(f"  输入目录    : {args.input}")
    logger.info(f"  输出目录    : {args.output}")
    logger.info(f"  tokenizer   : {args.tokenizer}")
    logger.info(f"  最大 token  : {args.max_tokens}")
    logger.info(f"  采样上限    : {args.max_samples}")
    logger.info(f"  训练集比例  : {args.train_ratio}")
    logger.info(f"  随机种子    : {args.seed}")
    logger.info(f"  dataset_info: {args.dataset_info}")
    logger.info("=" * 60)

    # Step 1: 加载 tokenizer
    logger.info("\n[Step 1] 加载 tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 2: 加载数据
    logger.info("\n[Step 2] 加载 parquet 数据...")
    raw_data = load_parquet_files(args.input)

    # Step 3: 转换格式
    logger.info("\n[Step 3] 转换为 sharegpt 格式...")
    processed, raw_stats = process_all(raw_data)
    logger.info(
        f"  转换完成: {raw_stats['valid']:,} 条有效 / {raw_stats['total']:,} 条总记录"
        f"（格式异常 {raw_stats['invalid']:,} 条）"
    )

    if not processed:
        logger.error("没有可处理的数据，退出。")
        return 1

    # Step 4: 打乱（先打乱再做 token 过滤，确保采样随机性）
    logger.info("\n[Step 4] 随机打乱数据...")
    random.shuffle(processed)
    logger.info(f"  打乱完成，共 {len(processed):,} 条")

    # Step 5: 分批按 token 长度过滤，达到采样上限后提前停止
    logger.info(
        f"\n[Step 5] 按 token 长度过滤（response > {args.max_tokens} tokens）"
        f"，采样上限 {args.max_samples:,} 条..."
    )
    processed, token_lengths, n_tok_filtered = filter_by_token_length_with_limit(
        processed, tokenizer, args.max_tokens, args.max_samples
    )
    logger.info(
        f"  过滤完成: 保留 {len(processed):,} 条，"
        f"过滤超长 {n_tok_filtered:,} 条"
    )

    if not processed:
        logger.error("过滤后没有剩余数据，退出。")
        return 1

    # Step 6: 划分训练/验证集
    logger.info("\n[Step 6] 划分训练集/验证集...")
    train_size = int(len(processed) * args.train_ratio)
    train_data = processed[:train_size]
    val_data   = processed[train_size:]
    logger.info(f"  训练集: {len(train_data):,} 条  |  验证集: {len(val_data):,} 条")

    # Step 7: 输出统计信息
    logger.info("\n[Step 7] 统计信息汇总...")
    print_statistics(
        raw_stats, train_data, val_data,
        token_lengths, n_tok_filtered,
        args.max_tokens, args.max_samples,
    )

    # Step 8: 保存
    logger.info("\n[Step 8] 保存输出文件...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"  已保存 {len(data):,} 条 -> {path}")

    # Step 9: 注册到 dataset_info.json
    logger.info("\n[Step 9] 注册数据集...")
    dataset_base = output_dir.name
    register_to_dataset_info(args.dataset_info, f"{dataset_base}_train", str(train_path))
    register_to_dataset_info(args.dataset_info, f"{dataset_base}_val",   str(val_path))

    logger.info("\n  全部完成。")
    return 0


if __name__ == "__main__":
    exit(main())
