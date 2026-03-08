#!/usr/bin/env python3
"""
从 OmniThought parquet 数据集中抽取 SFT 和 RL 数据集（支持 RV/CD 范围筛选 + token 长度过滤）

功能：
1. 按文件批量读取 parquet，攒够所需数据即提前停止
2. 按 token 长度过滤（thought < max_thought_tokens，question < max_question_tokens）
3. 根据 RV/CD 范围筛选 thought（只保留至少有 min_thoughts 个 thought 符合要求的数据）
4. 按 math/code 类型分类（solution 含 \\boxed 为 math）
5. 分别随机抽取 SFT 和 RL 数据，保证数据隔离

输入文件：
  - /fs/fast/u20240075/hf_home/datasets/alibaba-pai/OmniThought/data/*.parquet

输出文件：
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_60k/train.jsonl
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_20k/train.jsonl

输出格式与 select_sft_data_and_rl_data.py 保持一致：
  {"question": "...", "reasoning": [...]}
"""

import json
import argparse
import os
import glob
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from datetime import datetime
from tqdm import tqdm


# 默认配置
DEFAULT_INPUT = "/fs/fast/u20240075/hf_home/datasets/alibaba-pai/OmniThought/data"
DEFAULT_SFT_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_60k/train.jsonl"
DEFAULT_RL_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_20k/train.jsonl"
DEFAULT_SFT_MATH = 30000
DEFAULT_SFT_CODE = 30000
DEFAULT_RL_MATH = 10000
DEFAULT_RL_CODE = 10000
DEFAULT_SEED = 42

# RV/CD 默认范围
DEFAULT_RV_MIN = 0
DEFAULT_RV_MAX = 10
DEFAULT_CD_MIN = 0
DEFAULT_CD_MAX = 10
DEFAULT_MIN_THOUGHTS = 1

# Token 长度过滤默认值
DEFAULT_TOKENIZER = "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
DEFAULT_MAX_THOUGHT_TOKENS = 4096
DEFAULT_MAX_QUESTION_TOKENS = 1024
TOKENIZE_BATCH_SIZE = 512


# ============================================================================
# 参数解析
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 OmniThought parquet 数据集抽取 SFT 和 RL 数据（支持 RV/CD 筛选 + token 过滤）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（math/code 各 30k SFT + 10k RL）
  python select_sft_data_and_rl_data_omnithought.py

  # 只生成 SFT 数据集
  python select_sft_data_and_rl_data_omnithought.py --mode sft

  # 自定义 RV/CD 范围 + token 过滤
  python select_sft_data_and_rl_data_omnithought.py \\
      --rv-min 3 --rv-max 5 --cd-min 0 --cd-max 6 \\
      --max-thought-tokens 2048 --max-question-tokens 512

  # 自定义数据量和输出路径
  python select_sft_data_and_rl_data_omnithought.py \\
      --sft-math 20000 --sft-code 20000 \\
      --rl-math 10000  --rl-code 10000 \\
      --sft-output /path/to/sft/train.jsonl \\
      --rl-output  /path/to/rl/train.jsonl
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"OmniThought parquet 文件目录（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--sft-output",
        type=str,
        default=DEFAULT_SFT_OUTPUT,
        help=f"SFT 数据集输出路径（默认: {DEFAULT_SFT_OUTPUT}）",
    )
    parser.add_argument(
        "--rl-output",
        type=str,
        default=DEFAULT_RL_OUTPUT,
        help=f"RL 数据集输出路径（默认: {DEFAULT_RL_OUTPUT}）",
    )
    parser.add_argument("--sft-math", type=int, default=DEFAULT_SFT_MATH,
                        help=f"SFT math 样本数（默认: {DEFAULT_SFT_MATH}）")
    parser.add_argument("--sft-code", type=int, default=DEFAULT_SFT_CODE,
                        help=f"SFT code 样本数（默认: {DEFAULT_SFT_CODE}）")
    parser.add_argument("--rl-math", type=int, default=DEFAULT_RL_MATH,
                        help=f"RL math 样本数（默认: {DEFAULT_RL_MATH}）")
    parser.add_argument("--rl-code", type=int, default=DEFAULT_RL_CODE,
                        help=f"RL code 样本数（默认: {DEFAULT_RL_CODE}）")
    # RV/CD 范围
    parser.add_argument("--rv-min", type=int, default=DEFAULT_RV_MIN,
                        help=f"RV 下限（默认: {DEFAULT_RV_MIN}）")
    parser.add_argument("--rv-max", type=int, default=DEFAULT_RV_MAX,
                        help=f"RV 上限（默认: {DEFAULT_RV_MAX}）")
    parser.add_argument("--cd-min", type=int, default=DEFAULT_CD_MIN,
                        help=f"CD 下限（默认: {DEFAULT_CD_MIN}）")
    parser.add_argument("--cd-max", type=int, default=DEFAULT_CD_MAX,
                        help=f"CD 上限（默认: {DEFAULT_CD_MAX}）")
    parser.add_argument("--min-thoughts", type=int, default=DEFAULT_MIN_THOUGHTS,
                        help=f"过滤后每条数据最少保留的 thought 数（默认: {DEFAULT_MIN_THOUGHTS}）")
    # Token 过滤
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help=f"tokenizer 路径（默认: {DEFAULT_TOKENIZER}）")
    parser.add_argument("--max-thought-tokens", type=int, default=DEFAULT_MAX_THOUGHT_TOKENS,
                        help=f"thought 最大 token 数，超过则丢弃该 thought（默认: {DEFAULT_MAX_THOUGHT_TOKENS}）")
    parser.add_argument("--max-question-tokens", type=int, default=DEFAULT_MAX_QUESTION_TOKENS,
                        help=f"question 最大 token 数，超过则丢弃整条数据（默认: {DEFAULT_MAX_QUESTION_TOKENS}）")
    # 其他
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子（默认: {DEFAULT_SEED}）")
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "both"], default="both",
                        help="运行模式: sft=只生成SFT, rl=只生成RL, both=同时生成（默认: both）")
    return parser.parse_args()


# ============================================================================
# Tokenizer
# ============================================================================

def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("需要 transformers: pip install transformers")
    logger.info(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    logger.info(f"  词表大小: {tokenizer.vocab_size}")
    return tokenizer


def batch_tokenize(texts: List[str], tokenizer, batch_size: int = TOKENIZE_BATCH_SIZE) -> List[int]:
    """批量 tokenize，返回各文本的 token 数列表。"""
    all_lens = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
        all_lens.extend(len(ids) for ids in enc["input_ids"])
    return all_lens


# ============================================================================
# 数据处理
# ============================================================================

def get_thought_metrics(thought: Dict[str, Any]) -> Tuple[int, int]:
    """提取 thought 的 CD、RV 值。"""
    cd = thought.get("Cognitive_Difficulty", {}).get("level", 0)
    rv = thought.get("Reasoning_Verbosity", {}).get("level", 0)
    return cd, rv


def is_math_problem(item: Dict[str, Any]) -> bool:
    """通过 solution 中是否含 \\boxed 判断是否为 math 问题。"""
    for thought in item.get("reasoning", []):
        solution = thought.get("solution", "")
        if solution and "\\boxed" in solution:
            return True
    return False


def process_parquet_file(
    df,
    tokenizer,
    rv_min: int,
    rv_max: int,
    cd_min: int,
    cd_max: int,
    max_thought_tokens: int,
    max_question_tokens: int,
    min_thoughts: int,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    处理单个 parquet 文件的 DataFrame，返回 (math_items, code_items, stats)。

    过滤顺序：
      1. question token 数 >= max_question_tokens → 丢弃整条
      2. thought token 数 >= max_thought_tokens  → 丢弃该 thought
      3. thought RV/CD 不在范围               → 丢弃该 thought
      4. 剩余 thought 数 < min_thoughts         → 丢弃整条
    """
    stats = {
        "total": len(df),
        "q_filtered": 0,
        "t_total": 0,
        "t_filtered_token": 0,
        "t_filtered_rvcd": 0,
        "kept": 0,
    }

    # Step 1: 批量 tokenize 所有 question，过滤超长问题
    questions = df["question"].tolist()
    q_lens = batch_tokenize(questions, tokenizer)

    valid_indices = []
    for i, q_len in enumerate(q_lens):
        if q_len < max_question_tokens:
            valid_indices.append(i)
        else:
            stats["q_filtered"] += 1

    if not valid_indices:
        return [], [], stats

    # Step 2: 收集所有有效行的 thought 文本，批量 tokenize
    # 记录 (valid_row_idx, thought_idx) 的映射
    all_thought_texts: List[str] = []
    thought_map: List[Tuple[int, int]] = []  # (valid_row_pos, thought_idx)

    for vpos, vi in enumerate(valid_indices):
        reasoning = list(df.iloc[vi]["reasoning"])
        stats["t_total"] += len(reasoning)
        for t_idx, thought in enumerate(reasoning):
            all_thought_texts.append(thought.get("thought", ""))
            thought_map.append((vpos, t_idx))

    t_lens = batch_tokenize(all_thought_texts, tokenizer) if all_thought_texts else []

    # Step 3: 构建每个 thought 是否通过 token 过滤的 map
    # thought_pass[(vpos, t_idx)] = True/False
    thought_pass: Dict[Tuple[int, int], bool] = {}
    for (vpos, t_idx), t_len in zip(thought_map, t_lens):
        if t_len < max_thought_tokens:
            thought_pass[(vpos, t_idx)] = True
        else:
            thought_pass[(vpos, t_idx)] = False
            stats["t_filtered_token"] += 1

    # Step 4: 对每个有效行，过滤 thoughts 并分类
    math_items: List[Dict] = []
    code_items: List[Dict] = []

    for vpos, vi in enumerate(valid_indices):
        reasoning = list(df.iloc[vi]["reasoning"])
        valid_thoughts = []

        for t_idx, thought in enumerate(reasoning):
            if not thought_pass.get((vpos, t_idx), False):
                continue
            cd, rv = get_thought_metrics(thought)
            if rv_min <= rv <= rv_max and cd_min <= cd <= cd_max:
                valid_thoughts.append(thought)
            else:
                stats["t_filtered_rvcd"] += 1

        if len(valid_thoughts) < min_thoughts:
            continue

        item = {"question": df.iloc[vi]["question"], "reasoning": valid_thoughts}
        stats["kept"] += 1

        if is_math_problem(item):
            math_items.append(item)
        else:
            code_items.append(item)

    return math_items, code_items, stats


# ============================================================================
# SFT/RL 分割
# ============================================================================

def split_data_for_sft_rl(
    data: List[Dict[str, Any]],
    sft_samples: int,
    rl_samples: int,
    seed: int,
    domain_name: str,
    mode: str = "both",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """shuffle 后切片分给 SFT 和 RL，保证无重叠。"""
    shuffled = data.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    if mode == "sft":
        total_needed = sft_samples
    elif mode == "rl":
        total_needed = rl_samples
    else:
        total_needed = sft_samples + rl_samples

    if len(shuffled) < total_needed:
        logger.warning(
            f"{domain_name} 数据不足！需要 {total_needed} 条，实际只有 {len(shuffled)} 条，将使用全部数据"
        )

    if mode == "sft":
        sft_data = shuffled[:sft_samples]
        rl_data = []
    elif mode == "rl":
        sft_data = []
        rl_data = shuffled[:rl_samples]
    else:
        sft_data = shuffled[:sft_samples]
        rl_data = shuffled[sft_samples: sft_samples + rl_samples]

    logger.info(f"  {domain_name}: 池共 {len(shuffled)} 条 → SFT={len(sft_data)} 条, RL={len(rl_data)} 条")
    return sft_data, rl_data


# ============================================================================
# 验证、保存、统计
# ============================================================================

def verify_data_isolation(
    math_sft: List[Dict],
    math_rl: List[Dict],
    code_sft: List[Dict],
    code_rl: List[Dict],
) -> bool:
    """验证 SFT 和 RL 数据集之间无重叠（基于对象 id）。"""
    math_overlap = {id(x) for x in math_sft} & {id(x) for x in math_rl}
    if math_overlap:
        logger.error(f"Math SFT/RL 有重叠！重叠数: {len(math_overlap)}")
        return False
    code_overlap = {id(x) for x in code_sft} & {id(x) for x in code_rl}
    if code_overlap:
        logger.error(f"Code SFT/RL 有重叠！重叠数: {len(code_overlap)}")
        return False
    logger.info("数据隔离验证通过：SFT 和 RL 数据集之间无重叠 item")
    return True


def get_output_file_path(output_path: str) -> str:
    output_path = output_path.rstrip("/")
    if output_path.endswith(".jsonl"):
        return output_path
    return os.path.join(output_path, "train.jsonl")


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    actual_path = get_output_file_path(file_path)
    os.makedirs(os.path.dirname(actual_path), exist_ok=True)
    with open(actual_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"已保存 {len(data)} 条数据到: {actual_path}")


def print_statistics(data: List[Dict[str, Any]], name: str):
    if not data:
        logger.info(f"{name}: 空数据集")
        return
    q_lens = [len(item.get("question", "")) for item in data]
    thought_counts = [len(item.get("reasoning", [])) for item in data]
    logger.info(
        f"{name}: {len(data)} 条, 平均问题长度: {sum(q_lens)/len(q_lens):.0f} 字符, "
        f"平均 thought 数: {sum(thought_counts)/len(thought_counts):.1f}"
    )


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_arguments()

    title = {"sft": "SFT 数据集抽取工具", "rl": "RL 数据集抽取工具", "both": "SFT 和 RL 数据集抽取工具"}[args.mode]

    logger.info("=" * 70)
    logger.info(f"  {title}（OmniThought parquet 版）")
    logger.info("=" * 70)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"tokenizer: {args.tokenizer}")
    logger.info(f"token 过滤: question < {args.max_question_tokens}, thought < {args.max_thought_tokens}")
    if args.mode in ("sft", "both"):
        logger.info(f"SFT 输出: {args.sft_output}")
        logger.info(f"SFT Math: {args.sft_math} 条 | SFT Code: {args.sft_code} 条 | 合计: {args.sft_math + args.sft_code} 条")
    if args.mode in ("rl", "both"):
        logger.info(f"RL 输出: {args.rl_output}")
        logger.info(f"RL  Math: {args.rl_math} 条 | RL  Code: {args.rl_code} 条 | 合计: {args.rl_math + args.rl_code} 条")
    logger.info(f"RV 范围: [{args.rv_min}, {args.rv_max}]  CD 范围: [{args.cd_min}, {args.cd_max}]")
    logger.info(f"最小 thought 数: {args.min_thoughts}  随机种子: {args.seed}  模式: {args.mode}")
    logger.info("=" * 70)

    # 步骤 1: 加载 tokenizer
    logger.info("\n步骤 1: 加载 tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # 步骤 2: 收集 parquet 文件列表并随机打乱
    logger.info("\n步骤 2: 收集 parquet 文件...")
    parquet_files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not parquet_files:
        logger.error(f"未找到 parquet 文件: {args.input}/*.parquet")
        return 1
    logger.info(f"  共发现 {len(parquet_files)} 个 parquet 文件")

    # 用 seed 打乱文件顺序，确保随机采样
    rng = random.Random(args.seed)
    rng.shuffle(parquet_files)

    # 计算各 pool 所需数量
    if args.mode == "sft":
        math_needed = args.sft_math
        code_needed = args.sft_code
    elif args.mode == "rl":
        math_needed = args.rl_math
        code_needed = args.rl_code
    else:
        math_needed = args.sft_math + args.rl_math
        code_needed = args.sft_code + args.rl_code

    logger.info(f"  目标: math_pool >= {math_needed}, code_pool >= {code_needed}")

    # 步骤 3: 批量处理 parquet 文件，攒够数据即停止
    logger.info("\n步骤 3: 批量读取 parquet 并过滤...")
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas: pip install pandas pyarrow")

    math_pool: List[Dict] = []
    code_pool: List[Dict] = []

    total_stats = {
        "files_processed": 0,
        "total_rows": 0,
        "q_filtered": 0,
        "t_total": 0,
        "t_filtered_token": 0,
        "t_filtered_rvcd": 0,
        "kept": 0,
    }

    with tqdm(parquet_files, desc="parquet files", unit="file") as pbar:
        for fpath in pbar:
            df = pd.read_parquet(fpath)

            math_new, code_new, fstats = process_parquet_file(
                df, tokenizer,
                args.rv_min, args.rv_max, args.cd_min, args.cd_max,
                args.max_thought_tokens, args.max_question_tokens,
                args.min_thoughts,
            )

            # 只取 pool 还缺少的部分
            math_remaining = max(0, math_needed - len(math_pool))
            code_remaining = max(0, code_needed - len(code_pool))
            math_pool.extend(math_new[:math_remaining])
            code_pool.extend(code_new[:code_remaining])

            # 累计统计
            total_stats["files_processed"] += 1
            total_stats["total_rows"] += fstats["total"]
            total_stats["q_filtered"] += fstats["q_filtered"]
            total_stats["t_total"] += fstats["t_total"]
            total_stats["t_filtered_token"] += fstats["t_filtered_token"]
            total_stats["t_filtered_rvcd"] += fstats["t_filtered_rvcd"]
            total_stats["kept"] += fstats["kept"]

            pbar.set_postfix({
                "math": f"{len(math_pool)}/{math_needed}",
                "code": f"{len(code_pool)}/{code_needed}",
            })

            if len(math_pool) >= math_needed and len(code_pool) >= code_needed:
                logger.info(
                    f"\n  已收集足够数据，提前停止（处理了 {total_stats['files_processed']}/{len(parquet_files)} 个文件）"
                )
                break

    # 过滤统计
    logger.info("\n" + "-" * 60)
    logger.info("过滤统计:")
    logger.info(f"  处理文件数     : {total_stats['files_processed']} / {len(parquet_files)}")
    logger.info(f"  处理总行数     : {total_stats['total_rows']:,}")
    logger.info(f"  question 过长丢弃: {total_stats['q_filtered']:,}")
    logger.info(f"  thought 总数   : {total_stats['t_total']:,}")
    logger.info(f"  thought token过长: {total_stats['t_filtered_token']:,}")
    logger.info(f"  thought RV/CD不符: {total_stats['t_filtered_rvcd']:,}")
    logger.info(f"  最终保留行数   : {total_stats['kept']:,}")
    logger.info(f"  math pool      : {len(math_pool):,} 条（需要 {math_needed:,} 条）")
    logger.info(f"  code pool      : {len(code_pool):,} 条（需要 {code_needed:,} 条）")

    if not math_pool and not code_pool:
        logger.error("没有符合条件的数据！")
        return 1

    # 步骤 4: 分割 SFT / RL
    logger.info("\n步骤 4: 分割 SFT / RL 数据...")
    math_sft, math_rl = split_data_for_sft_rl(
        math_pool, args.sft_math, args.rl_math, args.seed, "Math", mode=args.mode
    )
    code_sft, code_rl = split_data_for_sft_rl(
        code_pool, args.sft_code, args.rl_code, args.seed + 1000, "Code", mode=args.mode
    )

    # 步骤 5: 合并并打乱
    logger.info("\n步骤 5: 合并并打乱...")
    sft_data = math_sft + code_sft
    rl_data = math_rl + code_rl
    random.seed(args.seed)
    if sft_data:
        random.shuffle(sft_data)
    if rl_data:
        random.shuffle(rl_data)
    if sft_data:
        logger.info(f"  SFT 数据集: {len(sft_data)} 条 (Math {len(math_sft)} + Code {len(code_sft)})")
    if rl_data:
        logger.info(f"  RL  数据集: {len(rl_data)} 条 (Math {len(math_rl)} + Code {len(code_rl)})")

    # 步骤 6: 验证数据隔离
    if args.mode == "both":
        logger.info("\n步骤 6: 验证数据隔离...")
        if not verify_data_isolation(math_sft, math_rl, code_sft, code_rl):
            logger.error("数据隔离验证失败！")
            return 1

    # 步骤 7: 保存
    logger.info("\n步骤 7: 保存数据...")
    if sft_data:
        save_jsonl(sft_data, args.sft_output)
    if rl_data:
        save_jsonl(rl_data, args.rl_output)

    # 步骤 8: 统计信息
    logger.info("\n步骤 8: 统计信息...")
    if sft_data:
        print_statistics(math_sft, "Math SFT")
        print_statistics(code_sft, "Code SFT")
        print_statistics(sft_data, "合并 SFT")
    if rl_data:
        print_statistics(math_rl, "Math RL")
        print_statistics(code_rl, "Code RL")
        print_statistics(rl_data, "合并 RL")

    # 保存配置日志
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "input": args.input,
            "sft_output": args.sft_output,
            "rl_output": args.rl_output,
            "sft_math": args.sft_math,
            "sft_code": args.sft_code,
            "rl_math": args.rl_math,
            "rl_code": args.rl_code,
            "rv_min": args.rv_min,
            "rv_max": args.rv_max,
            "cd_min": args.cd_min,
            "cd_max": args.cd_max,
            "min_thoughts": args.min_thoughts,
            "max_thought_tokens": args.max_thought_tokens,
            "max_question_tokens": args.max_question_tokens,
            "seed": args.seed,
            "mode": args.mode,
        },
        "filter_stats": total_stats,
        "pool_stats": {
            "math_pool": len(math_pool),
            "code_pool": len(code_pool),
            "math_needed": math_needed,
            "code_needed": code_needed,
        },
        "output_stats": {
            "math_sft": len(math_sft),
            "code_sft": len(code_sft),
            "math_rl": len(math_rl),
            "code_rl": len(code_rl),
            "sft_total": len(sft_data),
            "rl_total": len(rl_data),
        },
    }

    for out_path, has_data in [(args.sft_output, bool(sft_data)), (args.rl_output, bool(rl_data))]:
        if has_data:
            log_path = Path(get_output_file_path(out_path)).parent / "selection_log.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"日志已保存: {log_path}")

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("  处理完成！")
    logger.info("=" * 70)
    logger.info(f"筛选条件: question < {args.max_question_tokens} tokens, thought < {args.max_thought_tokens} tokens")
    logger.info(f"          RV∈[{args.rv_min},{args.rv_max}], CD∈[{args.cd_min},{args.cd_max}]")
    if sft_data:
        logger.info(f"\nSFT 数据集: {args.sft_output}")
        logger.info(f"  总计: {len(sft_data)} 条  (Math {len(math_sft)} | Code {len(code_sft)})")
    if rl_data:
        logger.info(f"\nRL  数据集: {args.rl_output}")
        logger.info(f"  总计: {len(rl_data)} 条  (Math {len(math_rl)} | Code {len(code_rl)})")
    if args.mode == "both":
        logger.info("\n注意: SFT 和 RL 数据集之间已保证数据隔离")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())


"""
# 示例命令

# 默认配置：math/code 各 30k SFT + 10k RL
python reasoning_graph/select_sft_data_and_rl_data_omnithought.py

# 自定义 RV/CD 范围
python reasoning_graph/select_sft_data_and_rl_data_omnithought.py \
    --rv-min 3 --rv-max 5 --cd-min 0 --cd-max 6 \
    --sft-math 50000 --sft-code 50000 \
    --rl-math 10000  --rl-code 10000 \
    --sft-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_100k/train.jsonl \
    --rl-output  /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_20k/train.jsonl
    
python reasoning_graph/select_sft_data_and_rl_data_omnithought.py \
    --rv-min 3 --rv-max 5 --cd-min 0 --cd-max 6 \
    --sft-math 50000 --sft-code 0 \
    --rl-math 20000  --rl-code 0 \
    --sft-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_math_50k/train.jsonl \
    --rl-output  /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_math_20k/train.jsonl
"""
