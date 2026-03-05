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
DEFAULT_SFT_SAMPLES = 25000  # SFT 每个领域的数据量
DEFAULT_RL_SAMPLES = 25000   # RL 每个领域的数据量
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
  # 使用默认配置（各抽取25k，共50k）
  python select_sft_data_and_rl_data.py

  # 只生成 SFT 数据集
  python select_sft_data_and_rl_data.py --mode sft

  # 只生成 RL 数据集
  python select_sft_data_and_rl_data.py --mode rl

  # 自定义 SFT 和 RL 的数据量
  python select_sft_data_and_rl_data.py \
      --sft-samples-per-domain 10000 \
      --rl-samples-per-domain 5000

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
      --sft-samples-per-domain 10000 \
      --rl-samples-per-domain 10000 \
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
        "--sft-samples-per-domain",
        type=int,
        default=DEFAULT_SFT_SAMPLES,
        help=f"每个领域抽取的 SFT 样本数（默认: {DEFAULT_SFT_SAMPLES}）",
    )
    parser.add_argument(
        "--rl-samples-per-domain",
        type=int,
        default=DEFAULT_RL_SAMPLES,
        help=f"每个领域抽取的 RL 样本数（默认: {DEFAULT_RL_SAMPLES}）",
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
    将数据分割为 SFT 和 RL 两部分，保证 question 级别的数据隔离
    （SFT 和 RL 之间不会有相同的 question）

    Args:
        data: 原始数据（已经过 RV/CD 筛选）
        sft_samples: 需要抽取的 SFT 样本数
        rl_samples: 需要抽取的 RL 样本数
        seed: 随机种子
        domain_name: 领域名称（用于日志）
        mode: 运行模式: "sft"=只取SFT, "rl"=只取RL, "both"=两者都取

    Returns:
        (sft_data, rl_data) 元组，根据 mode 可能有一个为空列表
    """
    # 按 question 分组，保留所有数据
    from collections import defaultdict
    question_groups = defaultdict(list)
    for item in data:
        question = item.get("question", "")
        question_groups[question].append(item)
    
    # 获取所有唯一的 question
    unique_questions = list(question_groups.keys())
    
    # 根据模式计算需要的 question 数量
    if mode == "sft":
        total_needed = sft_samples
    elif mode == "rl":
        total_needed = rl_samples
    else:  # both
        total_needed = sft_samples + rl_samples

    if len(unique_questions) < total_needed:
        logger.warning(
            f"{domain_name} 唯一 question 数量不足！需要 {total_needed} 个，实际只有 {len(unique_questions)} 个"
        )
        if mode == "both":
            # 按比例分配
            total = len(unique_questions)
            sft_q_size = min(sft_samples, int(total * sft_samples / total_needed))
            sft_q_size = max(0, min(sft_q_size, total))
            rl_q_size = total - sft_q_size
            sft_questions = unique_questions[:sft_q_size]
            rl_questions = unique_questions[sft_q_size:]
        elif mode == "sft":
            sft_questions = unique_questions
            rl_questions = []
        else:  # rl
            sft_questions = []
            rl_questions = unique_questions
    else:
        # 随机打乱 question
        random.seed(seed)
        shuffled_questions = unique_questions.copy()
        random.shuffle(shuffled_questions)
        
        if mode == "both":
            # 分割 question：前一部分给 SFT，后一部分给 RL
            sft_questions = shuffled_questions[:sft_samples]
            rl_questions = shuffled_questions[sft_samples:total_needed]
        elif mode == "sft":
            sft_questions = shuffled_questions[:sft_samples]
            rl_questions = []
        else:  # rl
            sft_questions = []
            rl_questions = shuffled_questions[:rl_samples]

    # 根据 question 获取对应的所有数据（保留完整数据格式）
    sft_data = []
    rl_data = []
    
    for q in sft_questions:
        sft_data.extend(question_groups[q])
    
    for q in rl_questions:
        rl_data.extend(question_groups[q])

    logger.info(f"  {domain_name}: SFT={len(sft_data)} 条 ({len(sft_questions)} 个 question), RL={len(rl_data)} 条 ({len(rl_questions)} 个 question)")

    return sft_data, rl_data


def verify_data_isolation(
    math_sft: List[Dict[str, Any]],
    math_rl: List[Dict[str, Any]],
    code_sft: List[Dict[str, Any]],
    code_rl: List[Dict[str, Any]],
) -> bool:
    """
    验证 SFT 和 RL 数据集之间没有重叠（基于 question 字段）

    Args:
        math_sft: Math SFT 数据
        math_rl: Math RL 数据
        code_sft: Code SFT 数据
        code_rl: Code RL 数据

    Returns:
        是否验证通过
    """
    # 使用 question 字段作为唯一标识
    def get_question_set(data: List[Dict[str, Any]]) -> set:
        return {item.get("question", "") for item in data}

    math_sft_questions = get_question_set(math_sft)
    math_rl_questions = get_question_set(math_rl)
    code_sft_questions = get_question_set(code_sft)
    code_rl_questions = get_question_set(code_rl)

    # 检查 Math 内部是否有重叠
    math_overlap = math_sft_questions & math_rl_questions
    if math_overlap:
        logger.error(f"❌ Math 数据集中 SFT 和 RL 有重叠！重叠数: {len(math_overlap)}")
        return False

    # 检查 Code 内部是否有重叠
    code_overlap = code_sft_questions & code_rl_questions
    if code_overlap:
        logger.error(f"❌ Code 数据集中 SFT 和 RL 有重叠！重叠数: {len(code_overlap)}")
        return False

    logger.info("✅ 数据隔离验证通过：SFT 和 RL 数据集之间没有重叠的 question")
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
    if args.mode in ("rl", "both"):
        logger.info(f"📁 RL 输出: {args.rl_output}")
    logger.info(f"📊 SFT 每领域抽取: {args.sft_samples_per_domain} 条")
    logger.info(f"📊 RL 每领域抽取: {args.rl_samples_per_domain} 条")
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

    math_filtered, math_stats = filter_data_by_rv_cd(
        math_data, args.rv_min, args.rv_max, args.cd_min, args.cd_max, "Math", filter_thoughts=True, min_thoughts=args.min_thoughts
    )
    code_filtered, code_stats = filter_data_by_rv_cd(
        code_data, args.rv_min, args.rv_max, args.cd_min, args.cd_max, "Code", filter_thoughts=True, min_thoughts=args.min_thoughts
    )

    if not math_filtered and not code_filtered:
        logger.error("没有符合 RV/CD 范围要求的数据！")
        return 1

    # 步骤 3: 分割数据
    logger.info("\n" + "-" * 60)
    if args.mode == "both":
        logger.info("步骤 3: 分割 SFT 和 RL 数据（保证数据隔离）...")
    elif args.mode == "sft":
        logger.info("步骤 3: 抽取 SFT 数据...")
    else:  # rl
        logger.info("步骤 3: 抽取 RL 数据...")
    logger.info("-" * 60)

    math_sft, math_rl = split_data_for_sft_rl(
        math_filtered, args.sft_samples_per_domain, args.rl_samples_per_domain, 
        args.seed, "Math", mode=args.mode
    )

    # Code 使用不同的种子，确保与 Math 的随机性不同
    code_seed = args.seed + 1000
    code_sft, code_rl = split_data_for_sft_rl(
        code_filtered, args.sft_samples_per_domain, args.rl_samples_per_domain, 
        code_seed, "Code", mode=args.mode
    )

    # 步骤 4: 合并数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 4: 合并数据...")
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

    # 步骤 5: 验证数据隔离（仅在 both 模式下）
    if args.mode == "both":
        logger.info("\n" + "-" * 60)
        logger.info("步骤 5: 验证数据隔离...")
        logger.info("-" * 60)

        if not verify_data_isolation(math_sft, math_rl, code_sft, code_rl):
            logger.error("数据隔离验证失败！")
            return 1

    # 步骤 6: 保存数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 6: 保存数据...")
    logger.info("-" * 60)

    if sft_data:
        save_jsonl(sft_data, args.sft_output)
    if rl_data:
        save_jsonl(rl_data, args.rl_output)

    # 步骤 7: 打印统计信息
    logger.info("\n" + "-" * 60)
    logger.info("步骤 7: 统计信息...")
    logger.info("-" * 60)

    logger.info("原始数据统计:")
    print_statistics(math_data, "Math 原始")
    print_statistics(code_data, "Code 原始")
    
    logger.info("\n筛选后数据统计 (RV/CD 筛选):")
    print_statistics(math_filtered, "Math 筛选后")
    print_statistics(code_filtered, "Code 筛选后")

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
            "sft_samples_per_domain": args.sft_samples_per_domain,
            "rl_samples_per_domain": args.rl_samples_per_domain,
            "rv_min": args.rv_min,
            "rv_max": args.rv_max,
            "cd_min": args.cd_min,
            "cd_max": args.cd_max,
            "seed": args.seed,
            "mode": args.mode,
        },
        "statistics": {
            "math_total": len(math_data),
            "code_total": len(code_data),
            "math_filtered": len(math_filtered),
            "code_filtered": len(code_filtered),
            "math_stats": math_stats,
            "code_stats": code_stats,
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
    logger.info(f"   - Math: {len(math_filtered)}/{len(math_data)} 条符合 ({len(math_filtered)/len(math_data)*100:.1f}%)")
    logger.info(f"   - Code: {len(code_filtered)}/{len(code_data)} 条符合 ({len(code_filtered)/len(code_data)*100:.1f}%)")
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
# 示例1：传入文件夹路径（推荐，会自动在目录下创建 train.jsonl）
python reasoning_graph/select_sft_data_and_rl_data.py \
    --rv-min 3 --rv-max 5 \
    --cd-min 0 --cd-max 6 \
    --min-thoughts 2 \
    --sft-samples-per-domain 25000 \
    --rl-samples-per-domain 5000 \
    --sft-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_50k/train.jsonl \
    --rl-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_10k/train.jsonl

# 示例2：传入具体文件路径（兼容旧用法）
python reasoning_graph/select_sft_data_and_rl_data.py \
    --sft-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_50k/train.jsonl \
    --rl-output /fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_10k/train.jsonl

"""
