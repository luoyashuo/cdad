#!/usr/bin/env python3
"""
将 select_diff_data.py 筛选后的数据转换为 LLamaFactory 支持的 sharegpt 格式

支持多种数据选择策略（参考 OmniThought 论文）:
  1. cd_optimal: 选择 CD 在指定范围内的 CoT（默认 0-6，适合 7B 模型）
  2. rv_optimal: 选择 RV 在指定范围内的 CoT（默认 3-5，适合 7B 模型）
  3. combined: 双规则概率采样（P1: CD匹配模型能力 + P2: CD-RV差距惩罚）
  4. 旧策略: selected_by_cognitive_difficulty, selected_by_reasoning_verbosity, selected_by_lowest_fsf

额外参数:
  --fsf_lowest: 在候选范围内直接选择 FSF 最低的 CoT（而非概率采样）

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
    -s cd_optimal \\
    -o /path/to/output.jsonl \\
    -r LlamaFactory/data/dataset_info.json
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import random
from loguru import logger


# ============================================================================
# 论文中 7B 模型的默认参数设置
# ============================================================================
DEFAULT_CD_RANGE = (0, 6)       # CD 范围: 0-6（适合 7B 模型）
DEFAULT_RV_RANGE = (3, 5)       # RV 范围: 3-5（中等冗长度）
DEFAULT_MU_CD = 6               # 模型认知能力阈值 μ_CD
DEFAULT_BETA = 0.5              # P1 和 P2 的权重系数


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="将筛选后的数据转换为 sharegpt 格式（支持 OmniThought 论文中的数据选择策略）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 CD Optimal 策略（论文中 7B 模型配置）
  python build_sharegpt_dataset.py \\
    -i /path/to/train_merge_level1.jsonl \\
    -s cd_optimal \\
    -o /path/to/output_dir

  # 使用 RV Optimal 策略
  python build_sharegpt_dataset.py \\
    -i /path/to/input.jsonl \\
    -s rv_optimal \\
    -o /path/to/output_dir

  # 使用 Combined 双规则概率采样
  python build_sharegpt_dataset.py \\
    -i input.jsonl \\
    -s combined \\
    -o /path/to/output_dir

  # 在候选范围内选择 FSF 最低的（而非概率采样）
  python build_sharegpt_dataset.py \\
    -i input.jsonl \\
    -s combined \\
    --fsf_lowest \\
    -o /path/to/output_dir

  # 自定义 CD/RV 范围（适配不同规模模型）
  python build_sharegpt_dataset.py \\
    -i input.jsonl \\
    -s combined \\
    --cd_min 0 --cd_max 7 \\
    --rv_min 3 --rv_max 5 \\
    --mu_cd 7 \\
    -o /path/to/output_dir

  # 旧策略（直接使用预选字段）
  python build_sharegpt_dataset.py \\
    -i /path/to/input.jsonl \\
    -s selected_by_cognitive_difficulty \\
    -o /path/to/output_dir
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
        choices=[
            # 新策略（论文方法）
            "cd_optimal",           # CD 范围筛选
            "rv_optimal",           # RV 范围筛选
            "combined",             # 双规则概率采样
            # 旧策略（直接使用预选字段）
            "selected_by_cognitive_difficulty",
            "selected_by_lowest_fsf",
            "selected_by_reasoning_verbosity"
        ],
        help="选择策略: cd_optimal/rv_optimal/combined（论文方法）或旧字段选择器"
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

    # ========== 论文策略相关参数 ==========
    parser.add_argument(
        "--fsf_lowest",
        action="store_true",
        help="在候选 CoT 中直接选择 FSF 最低的（而非概率采样）"
    )
    parser.add_argument(
        "--cd_min",
        type=int,
        default=DEFAULT_CD_RANGE[0],
        help=f"CD 范围下限（默认: {DEFAULT_CD_RANGE[0]}）"
    )
    parser.add_argument(
        "--cd_max",
        type=int,
        default=DEFAULT_CD_RANGE[1],
        help=f"CD 范围上限（默认: {DEFAULT_CD_RANGE[1]}，论文 7B 模型配置）"
    )
    parser.add_argument(
        "--rv_min",
        type=int,
        default=DEFAULT_RV_RANGE[0],
        help=f"RV 范围下限（默认: {DEFAULT_RV_RANGE[0]}）"
    )
    parser.add_argument(
        "--rv_max",
        type=int,
        default=DEFAULT_RV_RANGE[1],
        help=f"RV 范围上限（默认: {DEFAULT_RV_RANGE[1]}，论文 7B 模型配置）"
    )
    parser.add_argument(
        "--mu_cd",
        type=int,
        default=DEFAULT_MU_CD,
        help=f"模型认知能力阈值 μ_CD（默认: {DEFAULT_MU_CD}，用于 combined 策略）"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"P1 和 P2 的权重系数 β（默认: {DEFAULT_BETA}，用于 combined 策略）"
    )

    return parser.parse_args()


# ============================================================================
# 论文数据选择策略实现
# ============================================================================

def get_thought_metrics(thought: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    从单个 thought 中提取 CD、RV、FSF 指标

    Returns:
        (cd, rv, fsf) 三元组
    """
    cd = thought.get("Cognitive_Difficulty", {}).get("level", 0)
    rv = thought.get("Reasoning_Verbosity", {}).get("level", 0)

    reasoning_graph = thought.get("reasoning_graph", [])
    fsf = reasoning_graph[0].get("fsf", 0.0) if reasoning_graph else 0.0

    return cd, rv, fsf


def select_cd_optimal(
    reasoning: List[Dict[str, Any]],
    cd_min: int,
    cd_max: int,
    fsf_lowest: bool = False
) -> Optional[Dict[str, Any]]:
    """
    CD Optimal 策略：选择 CD 在指定范围内的 CoT

    Args:
        reasoning: CoT 候选列表
        cd_min: CD 范围下限
        cd_max: CD 范围上限
        fsf_lowest: 是否在候选中选择 FSF 最低的

    Returns:
        选中的 thought，如果没有候选则返回 None
    """
    # 筛选 CD 在范围内的候选
    candidates = []
    for thought in reasoning:
        cd, rv, fsf = get_thought_metrics(thought)
        if cd_min <= cd <= cd_max:
            candidates.append({
                'thought': thought,
                'cd': cd,
                'rv': rv,
                'fsf': fsf
            })

    if not candidates:
        return None

    if fsf_lowest:
        # 选择 FSF 最低的
        selected = min(candidates, key=lambda x: x['fsf'])
    else:
        # 随机选择一个（等概率）
        selected = random.choice(candidates)

    return selected['thought']


def select_rv_optimal(
    reasoning: List[Dict[str, Any]],
    rv_min: int,
    rv_max: int,
    fsf_lowest: bool = False
) -> Optional[Dict[str, Any]]:
    """
    RV Optimal 策略：选择 RV 在指定范围内的 CoT

    Args:
        reasoning: CoT 候选列表
        rv_min: RV 范围下限
        rv_max: RV 范围上限
        fsf_lowest: 是否在候选中选择 FSF 最低的

    Returns:
        选中的 thought，如果没有候选则返回 None
    """
    # 筛选 RV 在范围内的候选
    candidates = []
    for thought in reasoning:
        cd, rv, fsf = get_thought_metrics(thought)
        if rv_min <= rv <= rv_max:
            candidates.append({
                'thought': thought,
                'cd': cd,
                'rv': rv,
                'fsf': fsf
            })

    if not candidates:
        return None

    if fsf_lowest:
        # 选择 FSF 最低的
        selected = min(candidates, key=lambda x: x['fsf'])
    else:
        # 随机选择一个（等概率）
        selected = random.choice(candidates)

    return selected['thought']


def compute_p1(
    candidates: List[Dict[str, Any]],
    mu_cd: int
) -> np.ndarray:
    """
    计算规则 P1：CD 匹配模型认知能力

    论文公式:
    f1(y_CoT) =
      若 S_CD ≤ μ_CD: max|S^(i)_CD - μ_CD|
      若 S_CD > μ_CD: max|S^(i)_CD - μ_CD| - (S_CD - μ_CD)

    Args:
        candidates: 候选列表，每项包含 'cd' 字段
        mu_cd: 模型认知能力阈值

    Returns:
        归一化后的概率分布
    """
    if not candidates:
        return np.array([])

    # 计算 max|S^(i)_CD - μ_CD|
    max_deviation = max(abs(c['cd'] - mu_cd) for c in candidates)

    f1_values = []
    for c in candidates:
        cd = c['cd']
        if cd <= mu_cd:
            # CD 在模型能力范围内，等概率高选中
            f1 = max_deviation
        else:
            # CD 超出模型能力，惩罚
            f1 = max_deviation - (cd - mu_cd)
        f1_values.append(max(f1, 0))  # 确保非负

    f1_array = np.array(f1_values, dtype=np.float64)

    # 归一化为概率分布
    total = f1_array.sum()
    if total > 0:
        return f1_array / total
    else:
        # 如果全为 0，返回均匀分布
        return np.ones(len(candidates)) / len(candidates)


def compute_p2(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """
    计算规则 P2：CD 与 RV 应匹配

    论文公式:
    f2(y_CoT) = max|S^(i)_CD - S^(i)_RV| - |S_CD - S_RV|

    Args:
        candidates: 候选列表，每项包含 'cd' 和 'rv' 字段

    Returns:
        归一化后的概率分布
    """
    if not candidates:
        return np.array([])

    # 计算 max|S^(i)_CD - S^(i)_RV|
    max_cd_rv_gap = max(abs(c['cd'] - c['rv']) for c in candidates)

    f2_values = []
    for c in candidates:
        gap = abs(c['cd'] - c['rv'])
        f2 = max_cd_rv_gap - gap
        f2_values.append(max(f2, 0))  # 确保非负

    f2_array = np.array(f2_values, dtype=np.float64)

    # 归一化为概率分布
    total = f2_array.sum()
    if total > 0:
        return f2_array / total
    else:
        # 如果全为 0，返回均匀分布
        return np.ones(len(candidates)) / len(candidates)


def select_combined(
    reasoning: List[Dict[str, Any]],
    cd_min: int,
    cd_max: int,
    rv_min: int,
    rv_max: int,
    mu_cd: int,
    beta: float = 0.5,
    fsf_lowest: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Combined 双规则概率采样策略

    论文公式:
    Pr(y_CoT) = β * P1(y_CoT) + (1-β) * P2(y_CoT)

    Args:
        reasoning: CoT 候选列表
        cd_min: CD 范围下限
        cd_max: CD 范围上限
        rv_min: RV 范围下限
        rv_max: RV 范围上限
        mu_cd: 模型认知能力阈值
        beta: P1 和 P2 的权重系数
        fsf_lowest: 是否在候选中选择 FSF 最低的（而非概率采样）

    Returns:
        选中的 thought，如果没有候选则返回 None
    """
    # 筛选同时满足 CD 和 RV 范围的候选
    candidates = []
    for idx, thought in enumerate(reasoning):
        cd, rv, fsf = get_thought_metrics(thought)
        if cd_min <= cd <= cd_max and rv_min <= rv <= rv_max:
            candidates.append({
                'idx': idx,
                'thought': thought,
                'cd': cd,
                'rv': rv,
                'fsf': fsf
            })

    if not candidates:
        return None

    if fsf_lowest:
        # 直接选择 FSF 最低的
        selected = min(candidates, key=lambda x: x['fsf'])
        return selected['thought']

    # 计算 P1 和 P2
    p1 = compute_p1(candidates, mu_cd)
    p2 = compute_p2(candidates)

    # 组合概率
    final_prob = beta * p1 + (1 - beta) * p2

    # 归一化
    total = final_prob.sum()
    if total > 0:
        final_prob = final_prob / total
    else:
        final_prob = np.ones(len(candidates)) / len(candidates)

    # 按概率采样
    selected_idx = np.random.choice(len(candidates), p=final_prob)
    return candidates[selected_idx]['thought']


def select_thought_by_strategy(
    item: Dict[str, Any],
    strategy: str,
    cd_min: int,
    cd_max: int,
    rv_min: int,
    rv_max: int,
    mu_cd: int,
    beta: float,
    fsf_lowest: bool
) -> Optional[Dict[str, Any]]:
    """
    根据策略选择 thought

    Args:
        item: 输入数据项（包含 reasoning 字段）
        strategy: 选择策略
        cd_min, cd_max: CD 范围
        rv_min, rv_max: RV 范围
        mu_cd: 模型认知能力阈值
        beta: P1/P2 权重系数
        fsf_lowest: 是否选择 FSF 最低的

    Returns:
        选中的 thought，如果没有则返回 None
    """
    reasoning = item.get("reasoning", [])
    if not reasoning:
        return None

    if strategy == "cd_optimal":
        return select_cd_optimal(reasoning, cd_min, cd_max, fsf_lowest)
    elif strategy == "rv_optimal":
        return select_rv_optimal(reasoning, rv_min, rv_max, fsf_lowest)
    elif strategy == "combined":
        return select_combined(
            reasoning, cd_min, cd_max, rv_min, rv_max, mu_cd, beta, fsf_lowest
        )
    else:
        # 旧策略：直接使用预选字段
        return item.get(strategy, None)


def convert_to_sharegpt(item: Dict[str, Any], selector: str) -> Dict[str, Any]:
    """
    将单个数据项转换为 sharegpt 格式（旧策略，使用预选字段）

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


def convert_to_sharegpt_with_strategy(
    item: Dict[str, Any],
    strategy: str,
    cd_min: int,
    cd_max: int,
    rv_min: int,
    rv_max: int,
    mu_cd: int,
    beta: float,
    fsf_lowest: bool
) -> Optional[Dict[str, Any]]:
    """
    使用论文策略将单个数据项转换为 sharegpt 格式

    Args:
        item: 输入数据项
        strategy: 选择策略
        cd_min, cd_max: CD 范围
        rv_min, rv_max: RV 范围
        mu_cd: 模型认知能力阈值
        beta: P1/P2 权重系数
        fsf_lowest: 是否选择 FSF 最低的

    Returns:
        sharegpt 格式的数据项，如果没有候选则返回 None
    """
    question = item.get("question", "")

    # 使用策略选择 thought
    selected_thought = select_thought_by_strategy(
        item, strategy, cd_min, cd_max, rv_min, rv_max, mu_cd, beta, fsf_lowest
    )

    if selected_thought is None:
        return None

    # 提取 thought 和 full_response
    thought = selected_thought.get("thought", "")
    solution = selected_thought.get("full_response", "")

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


def is_new_strategy(selector: str) -> bool:
    """判断是否为新策略（论文方法）"""
    return selector in ["cd_optimal", "rv_optimal", "combined"]


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

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("="*70)
    logger.info("  数据格式转换工具（支持 OmniThought 论文策略）")
    logger.info("="*70)
    if len(args.input) == 1:
        logger.info(f"📁 输入文件: {args.input[0]}")
    else:
        logger.info(f"📁 输入文件 ({len(args.input)} 个):")
        for idx, input_file in enumerate(args.input, 1):
            logger.info(f"   {idx}. {input_file}")
    logger.info(f"🎯 选择策略: {args.selector}")

    # 如果是新策略，显示参数配置
    if is_new_strategy(args.selector):
        logger.info(f"   📊 CD 范围: [{args.cd_min}, {args.cd_max}]")
        logger.info(f"   📊 RV 范围: [{args.rv_min}, {args.rv_max}]")
        if args.selector == "combined":
            logger.info(f"   📊 μ_CD (模型能力阈值): {args.mu_cd}")
            logger.info(f"   📊 β (P1/P2 权重): {args.beta}")
        logger.info(f"   📊 FSF 最低优先: {args.fsf_lowest}")

    logger.info(f"📁 输出目录: {args.output}")
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
    skipped_count = 0

    if is_new_strategy(args.selector):
        # 新策略：使用论文方法选择 CoT
        for item in data:
            try:
                sharegpt_item = convert_to_sharegpt_with_strategy(
                    item,
                    args.selector,
                    args.cd_min,
                    args.cd_max,
                    args.rv_min,
                    args.rv_max,
                    args.mu_cd,
                    args.beta,
                    args.fsf_lowest
                )
                if sharegpt_item is not None:
                    sharegpt_data.append(sharegpt_item)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.warning(f"转换数据失败: {e}")
                skipped_count += 1
                continue
    else:
        # 旧策略：直接使用预选字段
        for item in data:
            try:
                sharegpt_item = convert_to_sharegpt(item, args.selector)
                sharegpt_data.append(sharegpt_item)
            except Exception as e:
                logger.warning(f"转换数据失败: {e}")
                skipped_count += 1
                continue

    logger.info(f"✅ 转换完成，共 {len(sharegpt_data)} 条数据")
    if skipped_count > 0:
        logger.info(f"⚠️  跳过 {skipped_count} 条数据（无符合条件的候选 CoT）")

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

# ============================================================================
# 使用示例
# ============================================================================

# --- 新策略（OmniThought 论文方法）---

# 1. CD Optimal 策略（7B 模型配置：CD 范围 0-6）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s cd_optimal \
    --cd_min 0 --cd_max 6 \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_cd_optimal \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 2. RV Optimal 策略（7B 模型配置：RV 范围 3-5）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s rv_optimal \
    --rv_min 3 --rv_max 5 \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_rv_optimal \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 3. Combined 双规则概率采样（7B 模型配置）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --mu_cd 6 \
    --beta 0.5 \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_combined \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 4. Combined + FSF 最低（在候选范围内选择 FSF 最低的）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --fsf_lowest \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_combined_fsf \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# --- 旧策略（直接使用预选字段）---

# 5. 使用预选的 CG 最高字段
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s selected_by_cognitive_difficulty \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_cg \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 6. 使用预选的 RV 最高字段
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s selected_by_reasoning_verbosity \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_rv \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 7. 使用预选的 FSF 最低字段
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s selected_by_lowest_fsf \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_level1_fsf \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""