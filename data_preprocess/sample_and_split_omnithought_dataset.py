#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_qwen_dataset.py
Generate balanced train / val split (9:1) from OmniThought-R1-0528.jsonl
using Qwen2.5-3B-Instruct tokenizer and draw token-length distribution.
"""

from __future__ import annotations
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
import hashlib
# --------------------------------------------------
# 1. 统一配置（@dataclass）
# --------------------------------------------------
@dataclass
class Config:
    # I/O
    src_jsonl: str = "/root/autodl-tmp/luoyashuo/SoftCED/data/alibaba-pai/OmniThought-0528/data/OmniThought-R1-0528.jsonl"
    out_dir: str = "./output"
    tokenizer_path: str = "/root/autodl-tmp/luoyashuo/SoftCED/model/Qwen/Qwen2.5-3B-Instruct"

    # 采样
    samples_per_bin: int = 10_000        # 每个难度档上限
    train_ratio: float = 0.9             # 9:1 划分
    random_seed: int = 42
    max_token_len: int = 16384

    # 绘图
    fig_dpi: int = 300
    fig_font: str = "DejaVu Sans"

    # 内部字段（运行时会填充）
    tokenizer = None                     # 初始化后置入

    def __post_init__(self):
        random.seed(self.random_seed)
        os.makedirs(self.out_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=False
        )

# --------------------------------------------------
# 2. 工具函数
# --------------------------------------------------


def generate_question_id(question: str) -> str:
    """
    生成问题的唯一标识ID
    
    参数:
        question: 输入的问题字符串
        
    返回:
        32位长度的唯一标识字符串
    """
    # 确保输入是字符串类型
    if not isinstance(question, str):
        raise TypeError("输入必须是字符串类型")
    
    # 使用UTF-8编码转换为字节流
    question_bytes = question.encode('utf-8')
    
    # 计算MD5哈希值
    md5_hash = hashlib.md5(question_bytes)
    
    # 返回32位十六进制字符串
    return md5_hash.hexdigest()
def categorize_level(level: float) -> str:
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

def build_sharegpt_entry(question: str,full_response:str) -> Dict:
    # TODO:根据question生成hash值作为id
    id = generate_question_id(question)
    return {
        "id": id,
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
        # "system": (
        #     "You FIRST think about the reasoning process as an internal monologue "
        #     "and then provide the final answer. "
        #     "The reasoning process MUST BE enclosed within <think> </think> tags. "
        #     "The final answer MUST BE put in \boxed{}."
        # )
    }

# --------------------------------------------------
# 3. 主流程
# --------------------------------------------------
def load_and_balance(cfg: Config) -> List[Dict]:
    """返回 list[dict]，已平衡抽样"""
    level_counter = defaultdict(int)
    selected = []

    with open(cfg.src_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 随机打乱（固定种子）
    random.shuffle(lines)
    total_lines = len(lines)
    for line in tqdm(lines,total=total_lines,desc="均匀采样中..."):
        try:
            row = json.loads(line)
            level = float(row["reasoning"][0]["Cognitive_Difficulty"]["level"])
            group = categorize_level(level)
            if group == "其他":
                continue
            if not r'\boxed' in row['reasoning'][0]['solution']:
                # 用于过滤，仅保留MATH
                continue
            if level_counter[group] < cfg.samples_per_bin:
                entry = build_sharegpt_entry(row["question"],row["reasoning"][0]['full_response'])
                entry["_level"] = level        # 后续用于绘图
                selected.append(entry)
                level_counter[group] += 1
                if all(v >= cfg.samples_per_bin for v in level_counter.values()):
                    break
        except Exception:
            continue
    print(f"Balanced sampling done. Samples:{len(selected)}")
    return selected

def split_dataset(full: List[Dict], cfg: Config) -> Tuple[List[Dict], List[Dict]]:
    random.shuffle(full)
    split_idx = int(len(full) * cfg.train_ratio)
    return full[:split_idx], full[split_idx:]

def save_json(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False,indent=4)
    print(f"Saved {len(data)} lines to {path}")

# --------------------------------------------------
# 4. 绘图
# --------------------------------------------------
def plot_length_dist(lengths: List[int], cfg: Config, split_name: str):
    plt.rcParams["font.family"] = cfg.fig_font
    fig, ax = plt.subplots(figsize=(6, 4))

    # 统计
    arr = np.array(lengths)
    mean = arr.mean()
    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99])
    mini, maxi = arr.min(), arr.max()

    ax.hist(arr, bins=50, color="#4c72b0", alpha=0.75, edgecolor="k")
    ax.axvline(mean, color="red", lw=2, label=f"Mean={mean:.1f}")

    # 标注
    text = (
        f"Config: seed={cfg.random_seed}, bins={cfg.samples_per_bin}, ratio={cfg.train_ratio}\n"
        f"Min={mini}, Max={maxi}\n"
        f"Median={p50:.1f}, P90={p90:.1f}, P95={p95:.1f}, P99={p99:.1f}\n"
        f"Total samples={len(arr)}"
    )
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Tokens (Qwen tokenizer)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Token Length Distribution ({split_name})")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(cfg.out_dir, f"token_dist_{split_name}.png")
    fig.savefig(out_path, dpi=cfg.fig_dpi)
    plt.close(fig)
    print(f"Plot saved to {out_path}")

# --------------------------------------------------
# 5. 入口
# --------------------------------------------------
def main(cfg):
    print("Config:\n", json.dumps(asdict(cfg), indent=2, default=str))

    # 1. 采样
    samples = load_and_balance(cfg)

    # 2. 划分
    train, val = split_dataset(samples, cfg)
    dict_data = {"train": train, "val": val}
    # 3. 计算 token 长度，过滤
    for split, data in dict_data.items():
        lengths = [
            len(cfg.tokenizer.encode(item["conversations"][0]["value"]+item["conversations"][1]["value"]+item.get('system',''), add_special_tokens=False))
            for item in tqdm(data,total=len(data),desc=f"计算 {split} token 长度中...")
        ]
        plot_length_dist(lengths, cfg, split)
        index_lt_max_len = [i for i, l in enumerate(lengths) if l <= cfg.max_token_len]
        datas = [data[i] for i in index_lt_max_len]
        dict_data[split] = datas
    # 4. 保存
    save_json(dict_data['train'], os.path.join(cfg.out_dir, "train.jsonl"))
    save_json(dict_data['val'], os.path.join(cfg.out_dir, "val.jsonl"))
if __name__ == "__main__":
    
    # 0818：过滤了Code，仅保留MATH问题，同时取消了system prompt
    cfg = Config(
        src_jsonl="/root/autodl-tmp/luoyashuo/SoftCED/data/alibaba-pai/OmniThought-0528/data/OmniThought-R1-0528.jsonl",
        out_dir="/root/autodl-tmp/luoyashuo/SoftCED/data/alibaba-pai/OmniThought-0528/processed/OmniThought-0818-math",
        samples_per_bin=3000,
        random_seed=307,
        train_ratio=0.9,
        max_token_len=16384,
    )
    main(cfg)