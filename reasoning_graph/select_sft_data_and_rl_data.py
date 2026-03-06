#!/usr/bin/env python3
"""
从筛选后的数据中分别抽取 SFT 和 RL 数据集（支持 RV/CD 范围筛选）

功能：
1. 根据 RV/CD 范围筛选数据（只保留至少有一个 thought 符合范围要求的数据）
2. 从符合条件的数据中分别随机抽取指定数量的 SFT 和 RL 数据
3. 合并生成 SFT 数据集（math + code）
4. 保证数据隔离，再从剩余数据中抽取 RL 数据集（math + code）

输入文件：
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_math_all/train_merge_level1.jsonl
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_code_all/train_merge_level1.jsonl

输出文件：
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_20k/train.jsonl (20k条)
  - /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_20k/train.jsonl (20k条)

数据隔离：SFT 和 RL 数据集之间没有重叠数据
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from loguru import logger
from datetime import datetime


# 默认配置
DEFAULT_MATH_INPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_math_all/train_merge.jsonl"
DEFAULT_CODE_INPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_code_all/train_merge.jsonl"
DEFAULT_SFT_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_50k/train.jsonl"
DEFAULT_RL_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_50k/train.jsonl"
DEFAULT_SFT_MATH = 25000   # SFT math 数据量
DEFAULT_SFT_CODE = 25000   # SFT code 数据量
DEFAULT_RL_MATH = 25000    # RL math 数据量
DEFAULT_RL_CODE = 25000    # RL code 数据量
DEFAULT_SEED = 42

# RV/CD 默认范围（参考 build_sharegpt_dataset.py）
DEFAULT_RV_MIN = 0
DEFAULT_RV_MAX = 10
DEFAULT_CD_MIN = 0
DEFAULT_CD_MAX = 10
DEFAULT_MIN_THOUGHTS = 1  # 过滤后最小 thought 数量，低于此值的数据会被丢弃


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 math 和 code 数据源抽取 SFT 和 RL 数据集（支持 RV/CD 范围筛选）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（math/code 各抽取25k，共50k）
  python select_sft_data_and_rl_data.py

  # 只生成 SFT 数据集
  python select_sft_data_and_rl_data.py --mode sft

  # 只生成 RL 数据集
  python select_sft_data_and_rl_data.py --mode rl

  # 自定义 SFT 和 RL 的 math/code 数据量
  python select_sft_data_and_rl_data.py \
      --sft-math 15000 --sft-code 10000 \
      --rl-math 8000  --rl-code 5000

  # 自定义 RV/CD 范围筛选
  python select_sft_data_and_rl_data.py \
      --rv-min 3 --rv-max 5 \
      --cd-min 0 --cd-max 6

  # 自定义输入输出路径
  python select_sft_data_and_rl_data.py \
      --math-input /path/to/math.jsonl \
      --code-input /path/to/code.jsonl \
      --sft-output /path/to/sft.jsonl \
      --rl-output /path/to/rl.jsonl \
      --sft-math 10000 --sft-code 10000 \
      --rl-math 10000  --rl-code 10000 \
      --seed 123
        """,
    )
    parser.add_argument(
        "--math-input",
        type=str,
        default=DEFAULT_MATH_INPUT,
        help=f"Math 数据源路径（默认: {DEFAULT_MATH_INPUT}）",
    )
    parser.add_argument(
        "--code-input",
        type=str,
        default=DEFAULT_CODE_INPUT,
        help=f"Code 数据源路径（默认: {DEFAULT_CODE_INPUT}）",
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
    parser.add_argument(
        "--sft-math",
        type=int,
        default=DEFAULT_SFT_MATH,
        help=f"SFT 数据集中 math 样本数（默认: {DEFAULT_SFT_MATH}）",
    )
    parser.add_argument(
        "--sft-code",
        type=int,
        default=DEFAULT_SFT_CODE,
        help=f"SFT 数据集中 code 样本数（默认: {DEFAULT_SFT_CODE}）",
    )
    parser.add_argument(
        "--rl-math",
        type=int,
        default=DEFAULT_RL_MATH,
        help=f"RL 数据集中 math 样本数（默认: {DEFAULT_RL_MATH}）",
    )
    parser.add_argument(
        "--rl-code",
        type=int,
        default=DEFAULT_RL_CODE,
        help=f"RL 数据集中 code 样本数（默认: {DEFAULT_RL_CODE}）",
    )
    # RV/CD 范围筛选参数
    parser.add_argument(
        "--rv-min",
        type=int,
        default=DEFAULT_RV_MIN,
        help=f"RV (Reasoning Verbosity) 范围下限（默认: {DEFAULT_RV_MIN}）",
    )
    parser.add_argument(
        "--rv-max",
        type=int,
        default=DEFAULT_RV_MAX,
        help=f"RV (Reasoning Verbosity) 范围上限（默认: {DEFAULT_RV_MAX}）",
    )
    parser.add_argument(
        "--cd-min",
        type=int,
        default=DEFAULT_CD_MIN,
        help=f"CD (Cognitive Difficulty) 范围下限（默认: {DEFAULT_CD_MIN}）",
    )
    parser.add_argument(
        "--cd-max",
        type=int,
        default=DEFAULT_CD_MAX,
        help=f"CD (Cognitive Difficulty) 范围上限（默认: {DEFAULT_CD_MAX}）",
    )
    parser.add_argument(
        "--min-thoughts",
        type=int,
        default=DEFAULT_MIN_THOUGHTS,
        help=f"过滤后每个数据保留的最小 thought 数量，低于此值的数据会被丢弃（默认: {DEFAULT_MIN_THOUGHTS}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子（默认: {DEFAULT_SEED}）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "rl", "both"],
        default="both",
        help="运行模式: sft=只生成SFT, rl=只生成RL, both=同时生成（默认: both）",
    )
    return parser.parse_args()


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件

    Args:
        file_path: JSONL 文件路径

    Returns:
        数据列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                continue
    return data


def get_output_file_path(output_path: str) -> str:
    """
    智能处理输出路径：
    - 如果是文件路径（以.jsonl结尾），直接使用
    - 如果是文件夹路径，在该目录下创建 train.jsonl
    
    Args:
        output_path: 用户传入的输出路径
        
    Returns:
        实际的文件路径
    """
    output_path = output_path.rstrip('/')
    
    # 如果以 .jsonl 结尾，认为是文件路径
    if output_path.endswith('.jsonl'):
        return output_path
    
    # 否则认为是文件夹路径，自动添加 train.jsonl
    return os.path.join(output_path, 'train.jsonl')


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    保存数据到 JSONL 文件

    Args:
        data: 数据列表
        file_path: 输出文件路径（可以是文件夹或文件路径）
    """
    # 智能处理输出路径
    actual_file_path = get_output_file_path(file_path)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(actual_file_path), exist_ok=True)

    with open(actual_file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✅ 已保存 {len(data)} 条数据到: {actual_file_path}")


def get_thought_metrics(thought: Dict[str, Any]) -> Tuple[int, int]:
    """
    从单个 thought 中提取 CD、RV 指标（参考 build_sharegpt_dataset.py）

    Returns:
        (cd, rv) 元组
    """
    cd = thought.get("Cognitive_Difficulty", {}).get("level", 0)
    rv = thought.get("Reasoning_Verbosity", {}).get("level", 0)
    return cd, rv


def has_valid_thought(item: Dict[str, Any], rv_min: int, rv_max: int, cd_min: int, cd_max: int) -> bool:
    """
    检查数据项是否至少有一个 thought 符合 RV/CD 范围要求

    Args:
        item: 数据项（包含 reasoning 字段）
        rv_min, rv_max: RV 范围
        cd_min, cd_max: CD 范围

    Returns:
        是否有至少一个 thought 符合要求
    """
    reasoning = item.get("reasoning", [])
    if not reasoning:
        return False
    
    for thought in reasoning:
        cd, rv = get_thought_metrics(thought)
        if rv_min <= rv <= rv_max and cd_min <= cd <= cd_max:
            return True
    
    return False


def filter_thoughts_in_item(
    item: Dict[str, Any],
    rv_min: int,
    rv_max: int,
    cd_min: int,
    cd_max: int,
) -> Tuple[Dict[str, Any], int, int]:
    """
    过滤单个数据项中的 thoughts，只保留符合 RV/CD 范围的

    Args:
        item: 数据项（包含 reasoning 字段）
        rv_min, rv_max: RV 范围
        cd_min, cd_max: CD 范围

    Returns:
        (过滤后的数据项, 原始thought数, 保留thought数)
    """
    item = item.copy()  # 避免修改原始数据
    reasoning = item.get("reasoning", [])
    if not reasoning:
        return item, 0, 0
    
    original_count = len(reasoning)
    filtered_reasoning = []
    
    for thought in reasoning:
        cd, rv = get_thought_metrics(thought)
        if rv_min <= rv <= rv_max and cd_min <= cd <= cd_max:
            filtered_reasoning.append(thought)
    
    item["reasoning"] = filtered_reasoning
    return item, original_count, len(filtered_reasoning)


def filter_data_by_rv_cd(
    data: List[Dict[str, Any]],
    rv_min: int,
    rv_max: int,
    cd_min: int,
    cd_max: int,
    domain_name: str,
    filter_thoughts: bool = True,
    min_thoughts: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    根据 RV/CD 范围筛选数据

    Args:
        data: 原始数据
        rv_min, rv_max: RV 范围
        cd_min, cd_max: CD 范围
        domain_name: 领域名称（用于日志）
        filter_thoughts: 是否过滤 thought 本身（True=只保留符合要求的thought，False=只检查是否有符合条件的thought）
        min_thoughts: 过滤后每个数据保留的最小 thought 数量

    Returns:
        (筛选后的数据列表, 统计信息字典)
    """
    filtered_data = []
    total_original_thoughts = 0
    total_retained_thoughts = 0
    items_with_thoughts = 0
    
    for item in data:
        if filter_thoughts:
            # 过滤 thought 本身
            filtered_item, orig_count, retained_count = filter_thoughts_in_item(
                item, rv_min, rv_max, cd_min, cd_max
            )
            total_original_thoughts += orig_count
            total_retained_thoughts += retained_count
            
            # 只保留至少有 min_thoughts 个 thought 符合条件的数据
            if retained_count >= min_thoughts:
                filtered_data.append(filtered_item)
                items_with_thoughts += 1
        else:
            # 只检查是否有符合条件的 thought，不删除 thought
            if has_valid_thought(item, rv_min, rv_max, cd_min, cd_max):
                filtered_data.append(item)
    
    # 计算统计信息
    stats = {
        "original_items": len(data),
        "filtered_items": len(filtered_data),
        "items_with_valid_thoughts": items_with_thoughts if filter_thoughts else len(filtered_data),
        "total_original_thoughts": total_original_thoughts if filter_thoughts else None,
        "total_retained_thoughts": total_retained_thoughts if filter_thoughts else None,
        "avg_thoughts_per_item_before": total_original_thoughts / len(data) if filter_thoughts and data else None,
        "avg_thoughts_per_item_after": total_retained_thoughts / len(filtered_data) if filter_thoughts and filtered_data else None,
        "thought_retention_rate": total_retained_thoughts / total_original_thoughts * 100 if filter_thoughts and total_original_thoughts > 0 else None,
    }
    
    if filter_thoughts:
        logger.info(f"  {domain_name}: 原始 {len(data)} 条数据，筛选后 {len(filtered_data)} 条数据")
        logger.info(f"    Thought统计: 原始 {total_original_thoughts} 个 → 保留 {total_retained_thoughts} 个")
        logger.info(f"    平均thought数: {stats['avg_thoughts_per_item_before']:.2f} → {stats['avg_thoughts_per_item_after']:.2f}")
        logger.info(f"    Thought保留率: {stats['thought_retention_rate']:.1f}%")
    else:
        logger.info(f"  {domain_name}: 原始 {len(data)} 条，筛选后 {len(filtered_data)} 条符合 RV∈[{rv_min},{rv_max}] CD∈[{cd_min},{cd_max}]")
    
    return filtered_data, stats


def split_data_for_sft_rl(
    data: List[Dict[str, Any]],
    sft_samples: int,
    rl_samples: int,
    seed: int,
    domain_name: str,
    mode: str = "both",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将数据 shuffle 后直接切片分给 SFT 和 RL，保证条数精确且两者无重叠。

    Args:
        data: 原始数据（已经过 RV/CD 筛选和类型分类）
        sft_samples: SFT 目标条数
        rl_samples: RL 目标条数
        seed: 随机种子
        domain_name: 类型名称（用于日志）
        mode: "sft" / "rl" / "both"

    Returns:
        (sft_data, rl_data) 元组
    """
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
        rl_data = shuffled[sft_samples:sft_samples + rl_samples]

    logger.info(f"  {domain_name}: 池共 {len(shuffled)} 条 → SFT={len(sft_data)} 条, RL={len(rl_data)} 条")

    return sft_data, rl_data


def verify_data_isolation(
    math_sft: List[Dict[str, Any]],
    math_rl: List[Dict[str, Any]],
    code_sft: List[Dict[str, Any]],
    code_rl: List[Dict[str, Any]],
) -> bool:
    """
    验证 SFT 和 RL 数据集之间没有重叠（基于 item 对象 id，shuffle+slice 保证天然无重叠）
    """
    math_sft_ids = {id(item) for item in math_sft}
    math_rl_ids = {id(item) for item in math_rl}
    code_sft_ids = {id(item) for item in code_sft}
    code_rl_ids = {id(item) for item in code_rl}

    math_overlap = math_sft_ids & math_rl_ids
    if math_overlap:
        logger.error(f"❌ Math 数据集中 SFT 和 RL 有重叠！重叠数: {len(math_overlap)}")
        return False

    code_overlap = code_sft_ids & code_rl_ids
    if code_overlap:
        logger.error(f"❌ Code 数据集中 SFT 和 RL 有重叠！重叠数: {len(code_overlap)}")
        return False

    logger.info("✅ 数据隔离验证通过：SFT 和 RL 数据集之间无重叠 item")
    return True


def print_statistics(data: List[Dict[str, Any]], name: str):
    """打印数据集统计信息"""
    if not data:
        logger.info(f"{name}: 空数据集")
        return

    # 统计问题长度
    question_lengths = [len(item.get("question", "")) for item in data]
    avg_question_len = sum(question_lengths) / len(question_lengths)

    # 统计 thought 数量（如果存在）
    thought_counts = []
    for item in data:
        reasoning = item.get("reasoning", [])
        if isinstance(reasoning, list):
            thought_counts.append(len(reasoning))

    if thought_counts:
        avg_thoughts = sum(thought_counts) / len(thought_counts)
        logger.info(
            f"{name}: {len(data)} 条, 平均问题长度: {avg_question_len:.0f}, 平均 thought 数: {avg_thoughts:.1f}"
        )
    else:
        logger.info(f"{name}: {len(data)} 条, 平均问题长度: {avg_question_len:.0f}")


def is_math_problem(item: Dict[str, Any]) -> bool:
    """
    通过检查 reasoning 数组中任意一个元素的 solution 字段是否包含 \\boxed 来判断是否为 math 问题
    """
    for reasoning_item in item.get("reasoning", []):
        if not isinstance(reasoning_item, dict):
            continue
        solution = reasoning_item.get("solution", "")
        if solution and "\\boxed" in solution:
            return True
    return False


def classify_data(
    data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将数据按 math/code 类型分类（通过 solution 中是否含 \\boxed 判断）

    Returns:
        (math_data, code_data) 元组
    """
    math_data = []
    code_data = []
    for item in data:
        if is_math_problem(item):
            math_data.append(item)
        else:
            code_data.append(item)
    return math_data, code_data


def print_source_type_stats(
    source_name: str,
    total_raw: int,
    total_filtered: int,
    math_count: int,
    code_count: int,
):
    """打印数据源中 math/code 分布统计"""
    if total_filtered > 0:
        math_pct = math_count / total_filtered * 100
        code_pct = code_count / total_filtered * 100
    else:
        math_pct = code_pct = 0.0
    logger.info(f"  [{source_name}] 原始 {total_raw} 条 → RV/CD 筛选后 {total_filtered} 条")
    logger.info(f"    - Math (含 \\boxed): {math_count} 条 ({math_pct:.1f}%)")
    logger.info(f"    - Code (不含 \\boxed): {code_count} 条 ({code_pct:.1f}%)")


def main():
    args = parse_arguments()

    # 根据模式显示标题
    if args.mode == "sft":
        title = "SFT 数据集抽取工具"
    elif args.mode == "rl":
        title = "RL 数据集抽取工具"
    else:
        title = "SFT 和 RL 数据集抽取工具"

    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)
    logger.info(f"📁 Math 输入: {args.math_input}")
    logger.info(f"📁 Code 输入: {args.code_input}")
    if args.mode in ("sft", "both"):
        logger.info(f"📁 SFT 输出: {args.sft_output}")
        logger.info(f"📊 SFT Math: {args.sft_math} 条 | SFT Code: {args.sft_code} 条 | SFT 合计: {args.sft_math + args.sft_code} 条")
    if args.mode in ("rl", "both"):
        logger.info(f"📁 RL 输出: {args.rl_output}")
        logger.info(f"📊 RL  Math: {args.rl_math} 条 | RL  Code: {args.rl_code} 条 | RL  合计: {args.rl_math + args.rl_code} 条")
    logger.info(f"📊 RV 范围: [{args.rv_min}, {args.rv_max}]")
    logger.info(f"📊 CD 范围: [{args.cd_min}, {args.cd_max}]")
    logger.info(f"📊 过滤后最小 thought 数: {args.min_thoughts}")
    logger.info(f"🎲 随机种子: {args.seed}")
    logger.info(f"🔧 运行模式: {args.mode}")
    logger.info("=" * 70)

    # 检查输入文件
    if not os.path.exists(args.math_input):
        logger.error(f"Math 输入文件不存在: {args.math_input}")
        return 1

    if not os.path.exists(args.code_input):
        logger.error(f"Code 输入文件不存在: {args.code_input}")
        return 1

    # 步骤 1: 加载数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 1: 加载数据...")
    logger.info("-" * 60)

    logger.info("加载 Math 数据...")
    math_data = load_jsonl(args.math_input)
    logger.info(f"  ✅ 加载完成: {len(math_data)} 条")

    logger.info("加载 Code 数据...")
    code_data = load_jsonl(args.code_input)
    logger.info(f"  ✅ 加载完成: {len(code_data)} 条")

    # 步骤 2: 根据 RV/CD 范围筛选数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 2: 根据 RV/CD 范围筛选数据...")
    logger.info("-" * 60)
    logger.info(f"筛选条件: RV∈[{args.rv_min},{args.rv_max}], CD∈[{args.cd_min},{args.cd_max}]")
    logger.info("（只保留至少有一个 thought 符合范围要求的数据）")

    math_source_filtered, math_stats = filter_data_by_rv_cd(
        math_data, args.rv_min, args.rv_max, args.cd_min, args.cd_max, "math_input", filter_thoughts=True, min_thoughts=args.min_thoughts
    )
    code_source_filtered, code_stats = filter_data_by_rv_cd(
        code_data, args.rv_min, args.rv_max, args.cd_min, args.cd_max, "code_input", filter_thoughts=True, min_thoughts=args.min_thoughts
    )

    if not math_source_filtered and not code_source_filtered:
        logger.error("没有符合 RV/CD 范围要求的数据！")
        return 1

    # 步骤 3: 按 math/code 类型分类，统计各数据源中的比例
    logger.info("\n" + "-" * 60)
    logger.info("步骤 3: 按 math/code 类型分类（通过 solution 中是否含 \\boxed 判断）...")
    logger.info("-" * 60)

    math_src_math, math_src_code = classify_data(math_source_filtered)
    code_src_math, code_src_code = classify_data(code_source_filtered)

    print_source_type_stats("math_input", len(math_data), len(math_source_filtered), len(math_src_math), len(math_src_code))
    print_source_type_stats("code_input", len(code_data), len(code_source_filtered), len(code_src_math), len(code_src_code))

    # 合并两个数据源的 math/code 池
    all_math = math_src_math + code_src_math
    all_code = math_src_code + code_src_code
    logger.info(f"  汇总 Math 池: {len(all_math)} 条 (math_input {len(math_src_math)} + code_input {len(code_src_math)})")
    logger.info(f"  汇总 Code 池: {len(all_code)} 条 (math_input {len(math_src_code)} + code_input {len(code_src_code)})")

    # 步骤 4: 分割数据
    logger.info("\n" + "-" * 60)
    if args.mode == "both":
        logger.info("步骤 4: 分割 SFT 和 RL 数据（保证数据隔离）...")
    elif args.mode == "sft":
        logger.info("步骤 4: 抽取 SFT 数据...")
    else:  # rl
        logger.info("步骤 4: 抽取 RL 数据...")
    logger.info("-" * 60)

    math_sft, math_rl = split_data_for_sft_rl(
        all_math, args.sft_math, args.rl_math,
        args.seed, "Math", mode=args.mode
    )

    # Code 使用不同的种子，确保与 Math 的随机性不同
    code_seed = args.seed + 1000
    code_sft, code_rl = split_data_for_sft_rl(
        all_code, args.sft_code, args.rl_code,
        code_seed, "Code", mode=args.mode
    )

    # 步骤 5: 合并数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 5: 合并数据...")
    logger.info("-" * 60)

    sft_data = math_sft + code_sft
    rl_data = math_rl + code_rl

    # 再次随机打乱合并后的数据
    random.seed(args.seed)
    if sft_data:
        random.shuffle(sft_data)
    if rl_data:
        random.shuffle(rl_data)

    if sft_data:
        logger.info(f"  SFT 数据集: {len(sft_data)} 条 (Math {len(math_sft)} + Code {len(code_sft)})")
    if rl_data:
        logger.info(f"  RL 数据集: {len(rl_data)} 条 (Math {len(math_rl)} + Code {len(code_rl)})")

    # 步骤 6: 验证数据隔离（仅在 both 模式下）
    if args.mode == "both":
        logger.info("\n" + "-" * 60)
        logger.info("步骤 6: 验证数据隔离...")
        logger.info("-" * 60)

        if not verify_data_isolation(math_sft, math_rl, code_sft, code_rl):
            logger.error("数据隔离验证失败！")
            return 1

    # 步骤 7: 保存数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 7: 保存数据...")
    logger.info("-" * 60)

    if sft_data:
        save_jsonl(sft_data, args.sft_output)
    if rl_data:
        save_jsonl(rl_data, args.rl_output)

    # 步骤 8: 打印统计信息
    logger.info("\n" + "-" * 60)
    logger.info("步骤 8: 统计信息...")
    logger.info("-" * 60)

    logger.info("原始数据统计:")
    print_statistics(math_data, "math_input 原始")
    print_statistics(code_data, "code_input 原始")

    logger.info("\n各数据源 math/code 类型分布 (RV/CD 筛选后):")
    print_source_type_stats("math_input", len(math_data), len(math_source_filtered), len(math_src_math), len(math_src_code))
    print_source_type_stats("code_input", len(code_data), len(code_source_filtered), len(code_src_math), len(code_src_code))
    logger.info(f"  汇总 Math 池: {len(all_math)} 条 | Code 池: {len(all_code)} 条")

    if sft_data:
        logger.info("\nSFT 数据集统计:")
        print_statistics(math_sft, "Math SFT")
        print_statistics(code_sft, "Code SFT")
        print_statistics(sft_data, "合并 SFT")
    if rl_data:
        logger.info("\nRL 数据集统计:")
        print_statistics(math_rl, "Math RL")
        print_statistics(code_rl, "Code RL")
        print_statistics(rl_data, "合并 RL")

    # 保存配置日志
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "math_input": args.math_input,
            "code_input": args.code_input,
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
            "seed": args.seed,
            "mode": args.mode,
        },
        "statistics": {
            "math_input_total": len(math_data),
            "code_input_total": len(code_data),
            "math_input_filtered": len(math_source_filtered),
            "code_input_filtered": len(code_source_filtered),
            "math_input_stats": math_stats,
            "code_input_stats": code_stats,
            "math_input_math_count": len(math_src_math),
            "math_input_code_count": len(math_src_code),
            "code_input_math_count": len(code_src_math),
            "code_input_code_count": len(code_src_code),
            "pool_math_total": len(all_math),
            "pool_code_total": len(all_code),
            "math_sft": len(math_sft),
            "math_rl": len(math_rl),
            "code_sft": len(code_sft),
            "code_rl": len(code_rl),
            "sft_total": len(sft_data),
            "rl_total": len(rl_data),
        },
    }

    # 保存日志到 SFT 输出目录
    if sft_data:
        sft_output_dir = Path(get_output_file_path(args.sft_output)).parent
        sft_log_path = sft_output_dir / "selection_log.json"
        with open(sft_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"\n日志已保存: {sft_log_path}")

    # 保存日志到 RL 输出目录
    if rl_data:
        rl_output_dir = Path(get_output_file_path(args.rl_output)).parent
        rl_log_path = rl_output_dir / "selection_log.json"
        with open(rl_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"日志已保存: {rl_log_path}")

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("  处理完成！")
    logger.info("=" * 70)
    logger.info(f"📊 筛选条件: RV∈[{args.rv_min},{args.rv_max}], CD∈[{args.cd_min},{args.cd_max}]")
    logger.info(f"   - math_input: {len(math_source_filtered)}/{len(math_data)} 条符合 ({len(math_source_filtered)/len(math_data)*100:.1f}%)"
                f"  [Math {len(math_src_math)} ({len(math_src_math)/len(math_source_filtered)*100:.1f}%) | Code {len(math_src_code)} ({len(math_src_code)/len(math_source_filtered)*100:.1f}%)]"
                if math_source_filtered else f"   - math_input: 0/{len(math_data)} 条符合")
    logger.info(f"   - code_input: {len(code_source_filtered)}/{len(code_data)} 条符合 ({len(code_source_filtered)/len(code_data)*100:.1f}%)"
                f"  [Math {len(code_src_math)} ({len(code_src_math)/len(code_source_filtered)*100:.1f}%) | Code {len(code_src_code)} ({len(code_src_code)/len(code_source_filtered)*100:.1f}%)]"
                if code_source_filtered else f"   - code_input: 0/{len(code_data)} 条符合")
    if sft_data:
        logger.info("")
        logger.info(f"✅ SFT 数据集: {args.sft_output}")
        logger.info(f"   - 总计: {len(sft_data)} 条")
        logger.info(f"   - Math: {len(math_sft)} 条")
        logger.info(f"   - Code: {len(code_sft)} 条")
    if sft_data and rl_data:
        logger.info("")
    if rl_data:
        logger.info(f"✅ RL 数据集: {args.rl_output}")
        logger.info(f"   - 总计: {len(rl_data)} 条")
        logger.info(f"   - Math: {len(math_rl)} 条")
        logger.info(f"   - Code: {len(code_rl)} 条")
    if args.mode == "both":
        logger.info("\n⚠️  注意: SFT 和 RL 数据集之间已保证数据隔离")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())


"""
# 自定义 RV/CD 范围（例如：RV 3-5，CD 0-6）
# 示例1：指定 math/code 数量（两个输入文件会合并分类后再抽取）
python reasoning_graph/select_sft_data_and_rl_data.py \
    --rv-min 3 --rv-max 5 \
    --cd-min 0 --cd-max 6 \
    --min-thoughts 2 \
    --sft-math 40000 --sft-code 10000 \
    --rl-math 8000  --rl-code 2000 \
    --sft-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_50k/train.jsonl \
    --rl-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_10k/train.jsonl

"""
