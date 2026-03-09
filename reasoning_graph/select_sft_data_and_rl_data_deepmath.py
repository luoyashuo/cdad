#!/usr/bin/env python3
"""
从 DeepMath parquet 数据集中抽取 SFT 和 RL 数据集（支持 token 长度过滤）

实际数据格式（parquet，每行字段）：
  columns: question, final_answer, difficulty, topic, r1_solution_1, r1_solution_2, r1_solution_3

每个 r1_solution_i 为完整的推理文本（多行），末尾含 \\boxed{answer}。
解析方式（参考格式提示）：
  thought = r1_solution_i          # 完整推理文本作为 thought
  answer  = final_answer           # final_answer 字段作为 solution
  full_response = f"<think>\\n{thought}\\n</think>\\n{answer}"

功能：
1. 按文件批量读取 parquet，多进程并行处理，攒够所需数据即提前停止
2. 字符数下界快速预判：字符数 < max_tokens * 2 → 直接通过，跳过 tokenize
3. 按 token 长度精确过滤（thought < max_thought_tokens，question < max_question_tokens）
4. 每条数据至少保留 min_solutions 个有效 solution（thought）
5. DeepMath 全为 math 数据，直接按 math 数量抽取
6. 分别随机抽取 SFT 和 RL 数据，保证数据隔离

输入文件：
  - /data/hf_home/datasets/zwhe99/DeepMath-103K/data/*.parquet

输出文件：
  - /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_60k/train.jsonl
  - /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_20k/train.jsonl

输出格式与 select_sft_data_and_rl_data_omnithought.py 保持一致：
  {"question": "...", "reasoning": [{"thought": "...", "solution": "..."}, ...]}
"""

import json
import argparse
import os
import glob
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp


# 默认配置
DEFAULT_INPUT = "/data/hf_home/datasets/zwhe99/DeepMath-103K/data"
DEFAULT_SFT_OUTPUT = "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_60k/train.jsonl"
DEFAULT_RL_OUTPUT = "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_20k/train.jsonl"
DEFAULT_SFT_MATH = 60000
DEFAULT_RL_MATH = 20000
DEFAULT_SEED = 42
DEFAULT_MIN_SOLUTIONS = 1

# Token 长度过滤默认值
DEFAULT_TOKENIZER = "/data/modelscope_cache/models/Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_THOUGHT_TOKENS = 4096
DEFAULT_MAX_QUESTION_TOKENS = 1024
TOKENIZE_BATCH_SIZE = 2048

# 字符数下界快速预判：chars < max_tokens * CHAR_PER_TOKEN_LOWER → 必然通过，跳过 tokenize
# 对于数学+LaTeX 文本，Qwen2.5 约 2~5 chars/token，取 2 作保守下界
CHAR_PER_TOKEN_LOWER = 2

# 默认并行 worker 数
DEFAULT_WORKERS = 4

# DeepMath 中 r1_solution 列名
SOLUTION_COLUMNS = ["r1_solution_1", "r1_solution_2", "r1_solution_3"]


# ============================================================================
# 参数解析
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 DeepMath parquet 数据集抽取 SFT 和 RL 数据（支持 token 过滤）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（60k SFT + 20k RL）
  python select_sft_data_and_rl_data_deepmath.py

  # 只生成 SFT 数据集
  python select_sft_data_and_rl_data_deepmath.py --mode sft

  # 自定义 token 过滤和并行数
  python select_sft_data_and_rl_data_deepmath.py \\
      --max-thought-tokens 2048 --max-question-tokens 512 --workers 8

  # 自定义数据量和输出路径
  python select_sft_data_and_rl_data_deepmath.py \\
      --sft-math 50000 --rl-math 10000 \\
      --max-thought-tokens 4096 --max-question-tokens 1024 \\
      --sft-output /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_50k/train.jsonl \\
      --rl-output  /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_10k/train.jsonl
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help=f"DeepMath parquet 文件目录（默认: {DEFAULT_INPUT}）",
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
    parser.add_argument("--rl-math", type=int, default=DEFAULT_RL_MATH,
                        help=f"RL math 样本数（默认: {DEFAULT_RL_MATH}）")
    parser.add_argument("--min-solutions", type=int, default=DEFAULT_MIN_SOLUTIONS,
                        help=f"每条数据过滤后最少保留的 solution 数（默认: {DEFAULT_MIN_SOLUTIONS}）")
    # Token 过滤
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help=f"tokenizer 路径（默认: {DEFAULT_TOKENIZER}）")
    parser.add_argument("--max-thought-tokens", type=int, default=DEFAULT_MAX_THOUGHT_TOKENS,
                        help=f"thought 最大 token 数，超过则丢弃该 solution（默认: {DEFAULT_MAX_THOUGHT_TOKENS}）")
    parser.add_argument("--max-question-tokens", type=int, default=DEFAULT_MAX_QUESTION_TOKENS,
                        help=f"question 最大 token 数，超过则丢弃整条数据（默认: {DEFAULT_MAX_QUESTION_TOKENS}）")
    # 其他
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"随机种子（默认: {DEFAULT_SEED}）")
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "both"], default="both",
                        help="运行模式: sft=只生成SFT, rl=只生成RL, both=同时生成（默认: both）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行处理 parquet 文件的进程数，0=禁用多进程（默认: {DEFAULT_WORKERS}）")
    return parser.parse_args()


# ============================================================================
# Tokenizer
# ============================================================================

def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("需要 transformers: pip install transformers")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    return tokenizer


def batch_tokenize_needed(texts: List[str], max_tokens: int, tokenizer) -> List[bool]:
    """
    批量判断各文本是否通过 token 数过滤（token 数 < max_tokens）。
    优化：字符数 < max_tokens * CHAR_PER_TOKEN_LOWER 的直接判定为通过，跳过 tokenize。
    """
    pass_threshold_chars = max_tokens * CHAR_PER_TOKEN_LOWER

    # 分成"直接通过"和"需要 tokenize"两组
    results = [False] * len(texts)
    tokenize_indices = []

    for i, text in enumerate(texts):
        if len(text) < pass_threshold_chars:
            results[i] = True          # 字符数极短，必然通过
        else:
            tokenize_indices.append(i) # 需要实际 tokenize

    if tokenize_indices:
        batch_texts = [texts[i] for i in tokenize_indices]
        all_lens = []
        for start in range(0, len(batch_texts), TOKENIZE_BATCH_SIZE):
            batch = batch_texts[start: start + TOKENIZE_BATCH_SIZE]
            enc = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False)
            all_lens.extend(len(ids) for ids in enc["input_ids"])
        for idx, t_len in zip(tokenize_indices, all_lens):
            results[idx] = t_len < max_tokens

    return results


# ============================================================================
# 数据处理（可被多进程调用）
# ============================================================================

def get_solution_columns(df_columns: List[str]) -> List[str]:
    """返回 DataFrame 中实际存在的 r1_solution_* 列名。"""
    return [col for col in SOLUTION_COLUMNS if col in df_columns]


def process_one_file(args_tuple) -> Tuple[List[Dict], Dict[str, int]]:
    """
    处理单个 parquet 文件，返回 (items, stats)。
    设计为可被 multiprocessing.Pool 调用（参数打包为 tuple）。

    每条输出 item 格式：
      {"question": "...", "reasoning": [{"thought": "...", "solution": "..."}, ...]}

    过滤顺序：
      1. question 字符数快速预判 + tokenize 过滤超长问题
      2. thought 字符数快速预判 + tokenize 过滤超长 solution
      3. 剩余 solution 数 < min_solutions → 丢弃整条
    """
    fpath, tokenizer_path, max_thought_tokens, max_question_tokens, min_solutions = args_tuple

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    df = pd.read_parquet(fpath)

    stats = {
        "total": len(df),
        "q_filtered": 0,
        "t_total": 0,
        "t_filtered_token": 0,
        "kept": 0,
    }

    sol_cols = get_solution_columns(list(df.columns))
    if not sol_cols:
        return [], stats

    # Step 1: 过滤超长 question
    questions = df["question"].tolist()
    q_pass = batch_tokenize_needed(questions, max_question_tokens, tokenizer)

    valid_indices = []
    for i, passed in enumerate(q_pass):
        if passed:
            valid_indices.append(i)
        else:
            stats["q_filtered"] += 1

    if not valid_indices:
        return [], stats

    # Step 2: 收集所有有效行的 thought 文本，过滤超长 thought
    all_thought_texts: List[str] = []
    thought_map: List[Tuple[int, str]] = []  # (valid_row_pos, col_name)

    for vpos, vi in enumerate(valid_indices):
        row = df.iloc[vi]
        stats["t_total"] += len(sol_cols)
        for col in sol_cols:
            val = row[col]
            all_thought_texts.append(str(val) if val is not None else "")
            thought_map.append((vpos, col))

    t_pass = batch_tokenize_needed(all_thought_texts, max_thought_tokens, tokenizer) if all_thought_texts else []

    thought_pass: Dict[Tuple[int, str], bool] = {}
    for (vpos, col), passed in zip(thought_map, t_pass):
        thought_pass[(vpos, col)] = passed
        if not passed:
            stats["t_filtered_token"] += 1

    # Step 3: 构建输出 item
    items: List[Dict] = []

    for vpos, vi in enumerate(valid_indices):
        row = df.iloc[vi]
        final_answer = str(row["final_answer"]) if row["final_answer"] is not None else ""
        valid_reasoning = []

        for col in sol_cols:
            if not thought_pass.get((vpos, col), False):
                continue
            val = row[col]
            valid_reasoning.append({
                "thought": str(val) if val is not None else "",
                "solution": final_answer,
            })

        if len(valid_reasoning) < min_solutions:
            continue

        items.append({
            "question": row["question"],
            "reasoning": valid_reasoning,
        })
        stats["kept"] += 1

    return items, stats


# ============================================================================
# SFT/RL 分割
# ============================================================================

def split_data_for_sft_rl(
    data: List[Dict[str, Any]],
    sft_samples: int,
    rl_samples: int,
    seed: int,
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
            f"Math 数据不足！需要 {total_needed} 条，实际只有 {len(shuffled)} 条，将使用全部数据"
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

    logger.info(f"  Math: 池共 {len(shuffled)} 条 → SFT={len(sft_data)} 条, RL={len(rl_data)} 条")
    return sft_data, rl_data


# ============================================================================
# 验证、保存、统计
# ============================================================================

def verify_data_isolation(sft_data: List[Dict], rl_data: List[Dict]) -> bool:
    overlap = {id(x) for x in sft_data} & {id(x) for x in rl_data}
    if overlap:
        logger.error(f"SFT/RL 有重叠！重叠数: {len(overlap)}")
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
    reasoning_counts = [len(item.get("reasoning", [])) for item in data]
    logger.info(
        f"{name}: {len(data)} 条, 平均问题长度: {sum(q_lens)/len(q_lens):.0f} 字符, "
        f"平均 solution 数: {sum(reasoning_counts)/len(reasoning_counts):.1f}"
    )


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_arguments()

    title = {"sft": "SFT 数据集抽取工具", "rl": "RL 数据集抽取工具", "both": "SFT 和 RL 数据集抽取工具"}[args.mode]

    logger.info("=" * 70)
    logger.info(f"  {title}（DeepMath parquet 版）")
    logger.info("=" * 70)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"tokenizer: {args.tokenizer}")
    logger.info(f"token 过滤: question < {args.max_question_tokens}, thought < {args.max_thought_tokens}")
    logger.info(f"字符快速预判下界: chars < max_tokens * {CHAR_PER_TOKEN_LOWER} → 直接通过")
    if args.mode in ("sft", "both"):
        logger.info(f"SFT 输出: {args.sft_output}")
        logger.info(f"SFT Math: {args.sft_math} 条")
    if args.mode in ("rl", "both"):
        logger.info(f"RL 输出: {args.rl_output}")
        logger.info(f"RL  Math: {args.rl_math} 条")
    logger.info(f"最小 solution 数: {args.min_solutions}  随机种子: {args.seed}  模式: {args.mode}  workers: {args.workers}")
    logger.info("=" * 70)

    # 步骤 1: 收集 parquet 文件列表并随机打乱
    logger.info("\n步骤 1: 收集 parquet 文件...")
    parquet_files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not parquet_files:
        logger.error(f"未找到 parquet 文件: {args.input}/*.parquet")
        return 1
    logger.info(f"  共发现 {len(parquet_files)} 个 parquet 文件")

    rng = random.Random(args.seed)
    rng.shuffle(parquet_files)

    # 计算所需数量
    if args.mode == "sft":
        math_needed = args.sft_math
    elif args.mode == "rl":
        math_needed = args.rl_math
    else:
        math_needed = args.sft_math + args.rl_math

    logger.info(f"  目标: math_pool >= {math_needed}")

    # 步骤 2: 批量处理 parquet 文件（多进程）
    logger.info(f"\n步骤 2: 批量读取 parquet 并过滤（workers={args.workers}）...")

    math_pool: List[Dict] = []
    total_stats = {
        "files_processed": 0,
        "total_rows": 0,
        "q_filtered": 0,
        "t_total": 0,
        "t_filtered_token": 0,
        "kept": 0,
    }

    task_args = [
        (fpath, args.tokenizer, args.max_thought_tokens, args.max_question_tokens, args.min_solutions)
        for fpath in parquet_files
    ]

    workers = args.workers if args.workers > 0 else 1
    use_mp = args.workers > 1

    if use_mp:
        pool = mp.Pool(processes=workers)
        result_iter = pool.imap_unordered(process_one_file, task_args)
    else:
        result_iter = map(process_one_file, task_args)

    try:
        with tqdm(total=len(parquet_files), desc="parquet files", unit="file") as pbar:
            for new_items, fstats in result_iter:
                math_remaining = max(0, math_needed - len(math_pool))
                math_pool.extend(new_items[:math_remaining])

                total_stats["files_processed"] += 1
                total_stats["total_rows"] += fstats["total"]
                total_stats["q_filtered"] += fstats["q_filtered"]
                total_stats["t_total"] += fstats["t_total"]
                total_stats["t_filtered_token"] += fstats["t_filtered_token"]
                total_stats["kept"] += fstats["kept"]

                pbar.update(1)
                pbar.set_postfix({"math": f"{len(math_pool)}/{math_needed}"})

                if len(math_pool) >= math_needed:
                    logger.info(
                        f"\n  已收集足够数据，提前停止（处理了 {total_stats['files_processed']}/{len(parquet_files)} 个文件）"
                    )
                    break
    finally:
        if use_mp:
            pool.terminate()
            pool.join()

    # 过滤统计
    logger.info("\n" + "-" * 60)
    logger.info("过滤统计:")
    logger.info(f"  处理文件数        : {total_stats['files_processed']} / {len(parquet_files)}")
    logger.info(f"  处理总行数        : {total_stats['total_rows']:,}")
    logger.info(f"  question 过长丢弃 : {total_stats['q_filtered']:,}")
    logger.info(f"  solution 总数     : {total_stats['t_total']:,}")
    logger.info(f"  solution token过长: {total_stats['t_filtered_token']:,}")
    logger.info(f"  最终保留行数      : {total_stats['kept']:,}")
    logger.info(f"  math pool         : {len(math_pool):,} 条（需要 {math_needed:,} 条）")

    if not math_pool:
        logger.error("没有符合条件的数据！")
        return 1

    # 步骤 3: 分割 SFT / RL
    logger.info("\n步骤 3: 分割 SFT / RL 数据...")
    sft_data, rl_data = split_data_for_sft_rl(
        math_pool, args.sft_math, args.rl_math, args.seed, mode=args.mode
    )

    # 步骤 4: 打乱
    logger.info("\n步骤 4: 打乱数据...")
    random.seed(args.seed)
    if sft_data:
        random.shuffle(sft_data)
        logger.info(f"  SFT 数据集: {len(sft_data)} 条")
    if rl_data:
        random.shuffle(rl_data)
        logger.info(f"  RL  数据集: {len(rl_data)} 条")

    # 步骤 5: 验证数据隔离
    if args.mode == "both":
        logger.info("\n步骤 5: 验证数据隔离...")
        if not verify_data_isolation(sft_data, rl_data):
            logger.error("数据隔离验证失败！")
            return 1

    # 步骤 6: 保存
    logger.info("\n步骤 6: 保存数据...")
    if sft_data:
        save_jsonl(sft_data, args.sft_output)
    if rl_data:
        save_jsonl(rl_data, args.rl_output)

    # 步骤 7: 统计信息
    logger.info("\n步骤 7: 统计信息...")
    if sft_data:
        print_statistics(sft_data, "SFT")
    if rl_data:
        print_statistics(rl_data, "RL")

    # 保存配置日志
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "input": args.input,
            "sft_output": args.sft_output,
            "rl_output": args.rl_output,
            "sft_math": args.sft_math,
            "rl_math": args.rl_math,
            "min_solutions": args.min_solutions,
            "max_thought_tokens": args.max_thought_tokens,
            "max_question_tokens": args.max_question_tokens,
            "seed": args.seed,
            "mode": args.mode,
            "workers": args.workers,
        },
        "filter_stats": total_stats,
        "pool_stats": {
            "math_pool": len(math_pool),
            "math_needed": math_needed,
        },
        "output_stats": {
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
    if sft_data:
        logger.info(f"\nSFT 数据集: {args.sft_output}")
        logger.info(f"  总计: {len(sft_data)} 条")
    if rl_data:
        logger.info(f"\nRL  数据集: {args.rl_output}")
        logger.info(f"  总计: {len(rl_data)} 条")
    if args.mode == "both":
        logger.info("\n注意: SFT 和 RL 数据集之间已保证数据隔离")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())


"""
# 示例命令

# 默认配置：60k SFT + 20k RL，4 个 worker
python reasoning_graph/select_sft_data_and_rl_data_deepmath.py

# 只生成 SFT 数据集
python reasoning_graph/select_sft_data_and_rl_data_deepmath.py --mode sft

# 自定义 token 过滤和并行数
python reasoning_graph/select_sft_data_and_rl_data_deepmath.py \\
    --max-thought-tokens 2048 --max-question-tokens 512 --workers 8

# 自定义数据量和输出路径
python reasoning_graph/select_sft_data_and_rl_data_deepmath.py \
    --sft-math 50000 --rl-math 10000 \
    --max-thought-tokens 4096 --max-question-tokens 1024 \
    --sft-output /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_50k/train.jsonl \
    --rl-output  /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_10k/train.jsonl
"""
