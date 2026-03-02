#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sharegpt_dataset.py
Generate train / val split from OmniThought dataset with various thought selection strategies.

Usage:
    # Random strategy: filter by rv_min/rv_max, then randomly sample
    python build_sharegpt_dataset.py \
        -i /path/to/train.jsonl \
        -s random \
        -o /path/to/output \
        --train-ratio 0.99 \
        -r ../LlamaFactory/data/dataset_info.json \
        --rv_min 3 --rv_max 5

    # FSF strategy: use FSF scores to select top-k thoughts
    python build_sharegpt_dataset.py \
        -i /path/to/train.jsonl \
        -s fsf \
        -o /path/to/output \
        --train-ratio 0.99 \
        -r ../LlamaFactory/data/dataset_info.json \
        --fsf-topk 20000
"""

from __future__ import annotations
import os
import sys
import json
import random
import argparse
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# --------------------------------------------------
# 1. 统一配置（@dataclass）
# --------------------------------------------------
@dataclass
class Config:
    # I/O
    src_jsonl: str = ""
    out_dir: str = "./output"
    
    # 采样策略
    strategy: str = "random"  # "random", "fsf"
    
    # 过滤条件 (Reasoning_Verbosity level 范围)
    rv_min: int = 0
    rv_max: int = 10
    
    # FSF 相关参数
    fsf_topk: Optional[int] = None  # 如果设置，则按FSF分数选择top-k
    fsf_score_file: Optional[str] = None  # FSF分数文件路径
    
    # 数据集划分
    train_ratio: float = 0.9
    random_seed: int = 42
    
    # dataset_info.json 注册路径
    register_path: Optional[str] = None
    
    # 数据集名称
    dataset_name: str = "omnithought_dataset"

    def __post_init__(self):
        random.seed(self.random_seed)
        os.makedirs(self.out_dir, exist_ok=True)


# --------------------------------------------------
# 2. 工具函数
# --------------------------------------------------

def generate_question_id(question: str) -> str:
    """生成问题的唯一标识ID"""
    if not isinstance(question, str):
        raise TypeError("输入必须是字符串类型")
    question_bytes = question.encode('utf-8')
    md5_hash = hashlib.md5(question_bytes)
    return md5_hash.hexdigest()


def categorize_level(level: float) -> str:
    """将难度等级分类"""
    if 0 <= level <= 1:
        return "0-1"
    elif 2 <= level <= 3:
        return "2-3"
    elif 4 <= level <= 5:
        return "4-5"
    elif 6 <= level <= 7:
        return "6-7"
    elif 8 <= level <= 9:
        return "8-9"
    return "其他"


def build_sharegpt_entry(question: str, full_response: str, system: str = "") -> Dict:
    """构建 ShareGPT 格式的数据条目"""
    entry_id = generate_question_id(question)
    entry = {
        "id": entry_id,
        "conversations": [
            {
                "from": "human",
                "value": f"Solve the following math problem. The final answer MUST BE put in \\boxed{{}}. \nProblem: {question}\nAnswer: "
            },
            {
                "from": "gpt",
                "value": f"{full_response}"
            },
        ],
    }
    if system:
        entry["system"] = system
    return entry


def load_fsf_scores(fsf_score_file: str) -> Dict[str, float]:
    """加载FSF分数文件"""
    fsf_scores = {}
    if not os.path.exists(fsf_score_file):
        print(f"Warning: FSF score file not found: {fsf_score_file}")
        return fsf_scores
    
    with open(fsf_score_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 假设格式为 {"question_id": score} 或 {"question": ..., "score": ...}
                if "question_id" in data and "score" in data:
                    fsf_scores[data["question_id"]] = float(data["score"])
                elif "question" in data and "score" in data:
                    qid = generate_question_id(data["question"])
                    fsf_scores[qid] = float(data["score"])
                elif len(data) == 1:
                    # 格式: {question_id: score}
                    key = list(data.keys())[0]
                    fsf_scores[key] = float(data[key])
            except Exception as e:
                continue
    return fsf_scores


def select_thought_random(reasoning_list: List[Dict], rv_min: int, rv_max: int) -> Optional[Dict]:
    """
    随机策略：根据 Reasoning_Verbosity level 过滤后随机选择
    
    Args:
        reasoning_list: 推理列表
        rv_min: Reasoning_Verbosity 最小等级
        rv_max: Reasoning_Verbosity 最大等级
    
    Returns:
        选中的推理条目或 None
    """
    # 过滤符合 rv_min/rv_max 条件的推理
    filtered = []
    for r in reasoning_list:
        rv_level = r.get("Reasoning_Verbosity", {}).get("level", 5)
        if rv_min <= rv_level <= rv_max:
            filtered.append(r)
    
    if not filtered:
        return None
    
    # 随机选择一个
    return random.choice(filtered)


def select_thought_fsf(
    question_id: str, 
    reasoning_list: List[Dict], 
    fsf_scores: Dict[str, float],
    topk: int
) -> Optional[Dict]:
    """
    FSF策略：根据FSF分数选择top-k中的thought
    
    Args:
        question_id: 问题ID
        reasoning_list: 推理列表
        fsf_scores: FSF分数字典
        topk: 选择的top-k数量
    
    Returns:
        选中的推理条目或 None
    """
    # 为每个thought计算FSF分数（如果有的话）
    scored_reasoning = []
    for r in reasoning_list:
        # 获取thought的标识
        thought = r.get("thought", "")
        # 使用question_id + thought_hash作为key
        thought_id = generate_question_id(question_id + thought[:100])
        score = fsf_scores.get(thought_id, fsf_scores.get(question_id, 0.0))
        scored_reasoning.append((score, r))
    
    # 按分数排序，选择top-k
    scored_reasoning.sort(key=lambda x: x[0], reverse=True)
    topk_reasoning = scored_reasoning[:topk]
    
    if not topk_reasoning:
        return None
    
    # 在top-k中随机选择一个
    return random.choice([r for _, r in topk_reasoning])


# --------------------------------------------------
# 3. 主流程
# --------------------------------------------------

def load_and_filter(cfg: Config) -> List[Dict]:
    """
    加载数据并根据策略选择thought
    
    策略说明：
    1. random: 先按 rv_min/rv_max 过滤，再随机选择thought
    2. fsf: 使用FSF分数选择top-k thought
    
    注意：如果 --fsf-topk 未设置（为None），即使strategy=fsf也会回退到random策略
    """
    selected = []
    
    # 如果策略是fsf且设置了fsf_topk，加载FSF分数
    fsf_scores = {}
    if cfg.strategy == "fsf" and cfg.fsf_topk is not None:
        if cfg.fsf_score_file:
            fsf_scores = load_fsf_scores(cfg.fsf_score_file)
            print(f"Loaded {len(fsf_scores)} FSF scores")
    
    with open(cfg.src_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total samples: {len(lines)}")
    
    for line in tqdm(lines, total=len(lines), desc=f"Processing with {cfg.strategy} strategy..."):
        try:
            row = json.loads(line)
            question = row.get("question", "")
            reasoning_list = row.get("reasoning", [])
            
            if not question or not reasoning_list:
                continue
            
            # 确保solution包含\boxed（MATH问题）
            has_math_solution = any(
                r'\boxed' in r.get('solution', '') 
                for r in reasoning_list
            )
            if not has_math_solution:
                continue
            
            question_id = generate_question_id(question)
            
            # 根据策略选择thought
            if cfg.strategy == "fsf" and cfg.fsf_topk is not None:
                # FSF策略：使用FSF分数选择top-k
                selected_reasoning = select_thought_fsf(
                    question_id, reasoning_list, fsf_scores, cfg.fsf_topk
                )
            else:
                # Random策略：按rv_min/rv_max过滤后随机选择
                # 当--fsf-topk未设置时，即使strategy=fsf也会回退到random
                selected_reasoning = select_thought_random(
                    reasoning_list, cfg.rv_min, cfg.rv_max
                )
            
            if selected_reasoning is None:
                continue
            
            full_response = selected_reasoning.get('full_response', '')
            if not full_response:
                continue
            
            entry = build_sharegpt_entry(question, full_response)
            
            # 添加元信息
            entry["_level"] = selected_reasoning.get("Cognitive_Difficulty", {}).get("level", 5)
            entry["_rv_level"] = selected_reasoning.get("Reasoning_Verbosity", {}).get("level", 5)
            entry["_strategy"] = cfg.strategy
            
            selected.append(entry)
            
        except Exception as e:
            continue
    
    print(f"Selected {len(selected)} samples")
    return selected


def split_dataset(full: List[Dict], cfg: Config) -> Tuple[List[Dict], List[Dict]]:
    """划分训练集和验证集"""
    random.shuffle(full)
    split_idx = int(len(full) * cfg.train_ratio)
    return full[:split_idx], full[split_idx:]


def save_jsonl(data: List[Dict], path: str):
    """保存为JSONL格式"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} lines to {path}")


def save_json(data: List[Dict], path: str):
    """保存为JSON格式（数组）"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} lines to {path}")


def update_dataset_info(cfg: Config, train_path: str, val_path: str):
    """更新 LlamaFactory 的 dataset_info.json"""
    if not cfg.register_path:
        return
    
    dataset_info = {}
    if os.path.exists(cfg.register_path):
        try:
            with open(cfg.register_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load dataset_info.json: {e}")
    
    # 获取相对路径（相对于 LlamaFactory/data）
    abs_register_dir = os.path.dirname(os.path.abspath(cfg.register_path))
    abs_train = os.path.abspath(train_path)
    abs_val = os.path.abspath(val_path)
    
    try:
        train_rel = os.path.relpath(abs_train, abs_register_dir)
        val_rel = os.path.relpath(abs_val, abs_register_dir)
    except ValueError:
        # Windows上不同盘符的情况
        train_rel = abs_train
        val_rel = abs_val
    
    # 添加数据集信息
    dataset_info[cfg.dataset_name] = {
        "file_name": train_rel,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
    }
    
    # 如果有验证集，添加eval_dataset
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        dataset_info[f"{cfg.dataset_name}_eval"] = {
            "file_name": val_rel,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    
    # 保存
    os.makedirs(os.path.dirname(cfg.register_path), exist_ok=True)
    with open(cfg.register_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"Updated dataset_info.json: {cfg.dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Build ShareGPT format dataset from OmniThought with various thought selection strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random strategy (filter by rv_min/rv_max, then random sample)
  python build_sharegpt_dataset.py -i input.jsonl -s random -o output_dir --rv_min 3 --rv_max 5
  
  # FSF strategy (select top-k by FSF score)
  python build_sharegpt_dataset.py -i input.jsonl -s fsf -o output_dir --fsf-topk 20000
  
Note:
  If --fsf-topk is not set, even with -s fsf, it will fall back to random strategy.
        """
    )
    
    # 输入输出参数
    parser.add_argument("-i", "--input", required=True, help="输入JSONL文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("-s", "--strategy", default="random", 
                        choices=["random", "fsf"],
                        help="Thought选择策略 (default: random)")
    
    # 过滤参数
    parser.add_argument("--rv_min", type=int, default=0,
                        help="Reasoning_Verbosity最小等级 (default: 0)")
    parser.add_argument("--rv_max", type=int, default=10,
                        help="Reasoning_Verbosity最大等级 (default: 10)")
    
    # FSF参数
    parser.add_argument("--fsf-topk", type=int, default=None,
                        help="FSF分数top-k选择 (如果设置则使用FSF策略)")
    parser.add_argument("--fsf-score-file", type=str, default=None,
                        help="FSF分数文件路径")
    
    # 数据集划分参数
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="训练集比例 (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    
    # 注册参数
    parser.add_argument("-r", "--register", default=None,
                        help="LlamaFactory dataset_info.json路径，用于注册数据集")
    parser.add_argument("--dataset-name", default="omnithought_dataset",
                        help="数据集名称 (default: omnithought_dataset)")
    
    args = parser.parse_args()
    
    # 创建配置
    cfg = Config(
        src_jsonl=args.input,
        out_dir=args.output,
        strategy=args.strategy,
        rv_min=args.rv_min,
        rv_max=args.rv_max,
        fsf_topk=args.fsf_topk,
        fsf_score_file=args.fsf_score_file,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        register_path=args.register,
        dataset_name=args.dataset_name
    )
    
    print("=" * 60)
    print("Config:")
    print(json.dumps({k: v for k, v in asdict(cfg).items() if k != 'tokenizer'}, 
                     indent=2, default=str))
    print("=" * 60)
    
    # 检查策略逻辑
    if args.strategy == "fsf" and args.fsf_topk is None:
        print("\n[Note] Strategy is 'fsf' but --fsf-topk is not set.")
        print("       Will fall back to 'random' strategy with rv_min/rv_max filtering.\n")
    
    # 1. 加载并选择thought
    samples = load_and_filter(cfg)
    
    if not samples:
        print("No samples selected! Please check your filtering criteria.")
        sys.exit(1)
    
    # 2. 划分训练集和验证集
    train, val = split_dataset(samples, cfg)
    print(f"Train: {len(train)}, Val: {len(val)}")
    
    # 3. 保存
    train_path = os.path.join(cfg.out_dir, "train.jsonl")
    val_path = os.path.join(cfg.out_dir, "val.jsonl")
    
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    
    # 4. 更新 dataset_info.json
    if cfg.register_path:
        update_dataset_info(cfg, train_path, val_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
