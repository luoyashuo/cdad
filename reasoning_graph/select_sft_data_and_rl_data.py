#!/usr/bin/env python3
"""
从筛选后的数据中分别抽取 SFT 和 RL 数据集

功能：
1. 从 math 和 code 两个数据源分别随机抽取指定数量的数据
2. 合并生成 SFT 数据集（math + code）
3. 保证数据隔离，再从剩余数据中抽取 RL 数据集（math + code）

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
from typing import List, Dict, Any, Tuple
from loguru import logger
from datetime import datetime


# 默认配置
DEFAULT_MATH_INPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_math_all/train_merge.jsonl"
DEFAULT_CODE_INPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_code_all/train_merge.jsonl"
DEFAULT_SFT_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_sft_50k/train.jsonl"
DEFAULT_RL_OUTPUT = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rl_50k/train.jsonl"
DEFAULT_SAMPLES_PER_DOMAIN = 25000
DEFAULT_SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从 math 和 code 数据源抽取 SFT 和 RL 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置（各抽取10k，共20k）
  python select_sft_data_and_rl_data.py

  # 只生成 SFT 数据集
  python select_sft_data_and_rl_data.py --mode sft

  # 只生成 RL 数据集
  python select_sft_data_and_rl_data.py --mode rl

  # 自定义抽取数量
  python select_sft_data_and_rl_data.py \
      --math-input /path/to/math.jsonl \
      --code-input /path/to/code.jsonl \
      --samples-per-domain 5000 \
      --seed 123

  # 自定义输出路径
  python select_sft_data_and_rl_data.py \
      --sft-output /path/to/sft.jsonl \
      --rl-output /path/to/rl.jsonl
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
        "--samples-per-domain",
        type=int,
        default=DEFAULT_SAMPLES_PER_DOMAIN,
        help=f"每个领域抽取的样本数（默认: {DEFAULT_SAMPLES_PER_DOMAIN}）",
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


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    保存数据到 JSONL 文件

    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✅ 已保存 {len(data)} 条数据到: {file_path}")


def split_data_for_sft_rl(
    data: List[Dict[str, Any]],
    samples_per_domain: int,
    seed: int,
    domain_name: str,
    mode: str = "both",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将数据分割为 SFT 和 RL 两部分，保证 question 级别的数据隔离
    （SFT 和 RL 之间不会有相同的 question）

    Args:
        data: 原始数据
        samples_per_domain: 每个领域需要抽取的样本数（SFT 和 RL 各这么多）
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
        total_needed = samples_per_domain
    elif mode == "rl":
        total_needed = samples_per_domain
    else:  # both
        total_needed = samples_per_domain * 2

    if len(unique_questions) < total_needed:
        logger.warning(
            f"{domain_name} 唯一 question 数量不足！需要 {total_needed} 个，实际只有 {len(unique_questions)} 个"
        )
        if mode == "both":
            # 按比例分配
            sft_q_size = len(unique_questions) // 2
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
            # 分割 question：前一半给 SFT，后一半给 RL
            sft_questions = shuffled_questions[:samples_per_domain]
            rl_questions = shuffled_questions[samples_per_domain:total_needed]
        elif mode == "sft":
            sft_questions = shuffled_questions[:samples_per_domain]
            rl_questions = []
        else:  # rl
            sft_questions = []
            rl_questions = shuffled_questions[:samples_per_domain]

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
    logger.info(f"📊 每领域抽取: {args.samples_per_domain} 条")
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

    # 步骤 2: 分割数据
    logger.info("\n" + "-" * 60)
    if args.mode == "both":
        logger.info("步骤 2: 分割 SFT 和 RL 数据（保证数据隔离）...")
    elif args.mode == "sft":
        logger.info("步骤 2: 抽取 SFT 数据...")
    else:  # rl
        logger.info("步骤 2: 抽取 RL 数据...")
    logger.info("-" * 60)

    math_sft, math_rl = split_data_for_sft_rl(
        math_data, args.samples_per_domain, args.seed, "Math", mode=args.mode
    )

    # Code 使用不同的种子，确保与 Math 的随机性不同
    code_seed = args.seed + 1000
    code_sft, code_rl = split_data_for_sft_rl(
        code_data, args.samples_per_domain, code_seed, "Code", mode=args.mode
    )

    # 步骤 3: 合并数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 3: 合并数据...")
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

    # 步骤 4: 验证数据隔离（仅在 both 模式下）
    if args.mode == "both":
        logger.info("\n" + "-" * 60)
        logger.info("步骤 4: 验证数据隔离...")
        logger.info("-" * 60)

        if not verify_data_isolation(math_sft, math_rl, code_sft, code_rl):
            logger.error("数据隔离验证失败！")
            return 1

    # 步骤 5: 保存数据
    logger.info("\n" + "-" * 60)
    logger.info("步骤 5: 保存数据...")
    logger.info("-" * 60)

    if sft_data:
        save_jsonl(sft_data, args.sft_output)
    if rl_data:
        save_jsonl(rl_data, args.rl_output)

    # 步骤 6: 打印统计信息
    logger.info("\n" + "-" * 60)
    logger.info("步骤 6: 统计信息...")
    logger.info("-" * 60)

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
            "math_input": args.math_input,
            "code_input": args.code_input,
            "sft_output": args.sft_output,
            "rl_output": args.rl_output,
            "samples_per_domain": args.samples_per_domain,
            "seed": args.seed,
            "mode": args.mode,
        },
        "statistics": {
            "math_total": len(math_data),
            "code_total": len(code_data),
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
        sft_log_path = Path(args.sft_output).parent / "selection_log.json"
        with open(sft_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"\n日志已保存: {sft_log_path}")

    # 保存日志到 RL 输出目录
    if rl_data:
        rl_log_path = Path(args.rl_output).parent / "selection_log.json"
        with open(rl_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"日志已保存: {rl_log_path}")

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("  处理完成！")
    logger.info("=" * 70)
    if sft_data:
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
使用示例:

# 默认配置（同时生成 SFT 和 RL，各抽取 10k，共 20k）
python reasoning_graph/select_sft_data_and_rl_data.py

# 只生成 SFT 数据集
python reasoning_graph/select_sft_data_and_rl_data.py --mode sft

# 只生成 RL 数据集
python reasoning_graph/select_sft_data_and_rl_data.py --mode rl

# 自定义配置
python reasoning_graph/select_sft_data_and_rl_data.py \
    --math-input /path/to/math.jsonl \
    --code-input /path/to/code.jsonl \
    --sft-output /path/to/sft.jsonl \
    --rl-output /path/to/rl.jsonl \
    --samples-per-domain 10000 \
    --seed 42

# 只生成 SFT 并指定输出路径
python reasoning_graph/select_sft_data_and_rl_data.py \
    --mode sft \
    --sft-output /path/to/sft.jsonl \
    --samples-per-domain 10000
"""
