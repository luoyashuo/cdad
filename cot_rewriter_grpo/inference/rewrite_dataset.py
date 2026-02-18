#!/usr/bin/env python3
"""
数据集改写脚本

使用改写模型对原有数据集进行改造，主要流程：
1. 读取配置，转换改写模型格式
2. 读取原始数据集，构建prompt
3. 调用改写模型生成n次结果（支持断点续传）
4. 计算similarity_penalty过滤
5. 计算ppl_relative_improvement过滤（支持断点续传）
6. 计算fsf过滤（支持断点续传）
7. 选择最优结果，生成新数据集
"""

import argparse
import json
import os
import re
import sys
import math
import yaml
import gc
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import numpy as np

# 添加项目路径
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo')

import torch
import ray
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig

# 尝试导入tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # fallback


# =============================================================================
# 配置相关
# =============================================================================

@dataclass
class RewriteConfig:
    """改写任务配置"""
    # 模型路径
    rewrite_model_path: str
    student_model_path: str
    fsf_judge_model_path: str

    # 数据路径
    input_data_path: str
    output_dir: str

    # 改写参数
    n_rewrites: int = 4  # 每个样本调用n次改写模型

    # 阈值参数
    similarity_penalty_threshold: float = 0.9  # 相似度阈值，高于此值则丢弃
    cot_effect_ratio_threshold: float = 1.0    # CoT Effect Ratio阈值
    cot_effect_ratio_ceiling: float = 0.95     # CoT Effect Ratio上限
    fsf_threshold: float = 0.0                 # FSF阈值（已废弃）

    # 模型生成参数
    max_new_tokens: int = 15000
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # 批处理参数
    num_workers: int = 8           # 多进程 worker 数（用于 similarity 计算）
    process_batch_size: int = 32   # 每处理这么多样本后保存一次 ckpt

    # vLLM 参数 (用于改写模型)
    vllm_tensor_parallel_size: int = 2      # 每个实例的张量并行大小
    vllm_gpu_memory_utilization: float = 0.85
    vllm_micro_batch_size: int = 1          # 每个 actor 每次处理的任务数

    # SGLang 参数 (用于 student 和 fsf judge 模型)
    sglang_tensor_parallel_size: int = 1    # TP=1 数据并行
    sglang_micro_batch_size: int = 4

    # 可选：跳过某些步骤
    skip_rewrite: bool = False
    skip_similarity: bool = False
    skip_ppl: bool = False
    skip_fsf: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RewriteConfig':
        """从字典创建配置，处理向后兼容"""
        # 处理 ppl_relative_improvement_threshold -> cot_effect_ratio_threshold
        if 'ppl_relative_improvement_threshold' in config_dict and 'cot_effect_ratio_threshold' not in config_dict:
            config_dict['cot_effect_ratio_threshold'] = config_dict['ppl_relative_improvement_threshold'] + 1.0

        # 处理旧的 sgland_micro_batch_size 拼写
        if 'sgland_micro_batch_size' in config_dict and 'sglang_micro_batch_size' not in config_dict:
            config_dict['sglang_micro_batch_size'] = config_dict['sgland_micro_batch_size']

        return cls(**config_dict)


def load_config(config_path: str) -> RewriteConfig:
    """从yaml文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return RewriteConfig.from_dict(config_dict)


# =============================================================================
# 统计收集器
# =============================================================================

@dataclass
class StepStatistics:
    """单个步骤的统计信息"""
    step_name: str
    total_samples: int = 0
    total_items: int = 0  # 对于 rewrite 步骤是改写尝试次数
    passed: int = 0
    filtered: int = 0
    pass_rate: float = 0.0
    # 数值型指标的分布
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    # 额外信息
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportStatistics:
    """完整的报告统计信息"""
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    # 步骤统计
    step_rewrite: StepStatistics = field(default_factory=lambda: StepStatistics("rewrite"))
    step_similarity: StepStatistics = field(default_factory=lambda: StepStatistics("similarity"))
    step_ppl: StepStatistics = field(default_factory=lambda: StepStatistics("ppl"))
    step_fsf: StepStatistics = field(default_factory=lambda: StepStatistics("fsf"))

    # 最终统计
    final_stats: Dict[str, Any] = field(default_factory=dict)


class StatisticsCollector:
    """统计信息收集器"""

    def __init__(self, config: RewriteConfig):
        self.config = config
        self.report = ReportStatistics()
        self.report.timestamp = datetime.now().isoformat()
        self.report.config = {
            'rewrite_model_path': config.rewrite_model_path,
            'student_model_path': config.student_model_path,
            'fsf_judge_model_path': config.fsf_judge_model_path,
            'input_data_path': config.input_data_path,
            'output_dir': config.output_dir,
            'n_rewrites': config.n_rewrites,
            'similarity_penalty_threshold': config.similarity_penalty_threshold,
            'cot_effect_ratio_threshold': config.cot_effect_ratio_threshold,
            'cot_effect_ratio_ceiling': config.cot_effect_ratio_ceiling,
            'fsf_threshold': config.fsf_threshold,
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'presence_penalty': config.presence_penalty,
            'frequency_penalty': config.frequency_penalty,
        }

    def update_rewrite_stats(self, total_samples: int, success_count: int, total_rewrites: int):
        """更新改写步骤统计"""
        self.report.step_rewrite.total_samples = total_samples
        self.report.step_rewrite.total_items = total_rewrites
        self.report.step_rewrite.passed = success_count
        self.report.step_rewrite.filtered = total_rewrites - success_count
        self.report.step_rewrite.pass_rate = success_count / total_rewrites if total_rewrites > 0 else 0.0

    def update_similarity_stats(self, samples: List[Dict]):
        """更新相似度过滤统计"""
        total_rewrites = sum(len(s['rewrite_results']) for s in samples)
        filtered_count = sum(
            sum(1 for r in s['rewrite_results'] if r.get('filtered', False))
            for s in samples
        )
        passed_count = total_rewrites - filtered_count

        # 收集所有相似度值
        similarities = []
        for s in samples:
            for r in s['rewrite_results']:
                sim = r.get('similarity_penalty')
                if sim is not None:
                    similarities.append(sim)

        self.report.step_similarity.total_items = total_rewrites
        self.report.step_similarity.passed = passed_count
        self.report.step_similarity.filtered = filtered_count
        self.report.step_similarity.pass_rate = passed_count / total_rewrites if total_rewrites > 0 else 0.0
        self.report.step_similarity.values = similarities

        if similarities:
            self.report.step_similarity.mean = float(np.mean(similarities))
            self.report.step_similarity.std = float(np.std(similarities))
            self.report.step_similarity.min = float(np.min(similarities))
            self.report.step_similarity.max = float(np.max(similarities))
            self.report.step_similarity.median = float(np.median(similarities))
            self.report.step_similarity.p25 = float(np.percentile(similarities, 25))
            self.report.step_similarity.p75 = float(np.percentile(similarities, 75))

    def update_ppl_stats(self, cot_effect_ratios: List[float]):
        """更新PPL过滤统计"""
        passed = sum(1 for x in cot_effect_ratios if x > self.config.cot_effect_ratio_threshold)
        total = len(cot_effect_ratios)

        self.report.step_ppl.total_items = total
        self.report.step_ppl.passed = passed
        self.report.step_ppl.filtered = total - passed
        self.report.step_ppl.pass_rate = passed / total if total > 0 else 0.0
        self.report.step_ppl.values = cot_effect_ratios.copy()

        if cot_effect_ratios:
            self.report.step_ppl.mean = float(np.mean(cot_effect_ratios))
            self.report.step_ppl.std = float(np.std(cot_effect_ratios))
            self.report.step_ppl.min = float(np.min(cot_effect_ratios))
            self.report.step_ppl.max = float(np.max(cot_effect_ratios))
            self.report.step_ppl.median = float(np.median(cot_effect_ratios))
            self.report.step_ppl.p25 = float(np.percentile(cot_effect_ratios, 25))
            self.report.step_ppl.p75 = float(np.percentile(cot_effect_ratios, 75))
            self.report.step_ppl.extra = {
                'cot_effect_ratio_threshold': self.config.cot_effect_ratio_threshold
            }

    def update_fsf_stats(self, original_fsfs: List[float], rewrite_better_count: int, total_samples: int):
        """更新FSF过滤统计"""
        self.report.step_fsf.total_samples = total_samples
        self.report.step_fsf.total_items = len(original_fsfs)
        self.report.step_fsf.values = original_fsfs.copy()
        self.report.step_fsf.extra = {
            'samples_with_better_rewrite': rewrite_better_count,
            'samples_with_better_rewrite_rate': rewrite_better_count / total_samples if total_samples > 0 else 0.0
        }

        if original_fsfs:
            self.report.step_fsf.mean = float(np.mean(original_fsfs))
            self.report.step_fsf.std = float(np.std(original_fsfs))
            self.report.step_fsf.min = float(np.min(original_fsfs))
            self.report.step_fsf.max = float(np.max(original_fsfs))
            self.report.step_fsf.median = float(np.median(original_fsfs))
            self.report.step_fsf.p25 = float(np.percentile(original_fsfs, 25))
            self.report.step_fsf.p75 = float(np.percentile(original_fsfs, 75))

    def update_final_stats(self, stats: Dict[str, Any]):
        """更新最终统计"""
        self.report.final_stats = stats

    def save_to_json(self, output_dir: str):
        """保存统计信息到JSON文件"""
        report_path = os.path.join(output_dir, 'report.json')

        # 转换为可序列化的dict
        def to_dict(obj):
            if isinstance(obj, StepStatistics):
                d = asdict(obj)
                # 清空 values 列表以减小文件大小（只保留统计摘要）
                d['values_count'] = len(obj.values)
                d['values'] = obj.values[:100] if len(obj.values) <= 100 else obj.values[:100]
                return d
            elif isinstance(obj, ReportStatistics):
                d = asdict(obj)
                return d
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        report_dict = to_dict(self.report)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"\n  [Report] Detailed report saved to: {report_path}")
        return report_path


def save_distribution_plots(stats: StatisticsCollector, output_dir: str):
    """绘制分布图并保存"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Rewrite Dataset Statistics', fontsize=16, fontweight='bold')

        # 1. Similarity Penalty 分布
        if stats.report.step_similarity.values:
            ax = axes[0, 0]
            similarities = stats.report.step_similarity.values
            ax.hist(similarities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(stats.config.similarity_penalty_threshold, color='red',
                      linestyle='--', label=f'Threshold: {stats.config.similarity_penalty_threshold}')
            ax.axvline(stats.report.step_similarity.mean, color='green',
                      linestyle='--', label=f'Mean: {stats.report.step_similarity.mean:.3f}')
            ax.set_xlabel('Similarity Penalty')
            ax.set_ylabel('Count')
            ax.set_title('Similarity Penalty Distribution')
            ax.legend()
            # 添加统计信息文本
            stats_text = f'Mean: {stats.report.step_similarity.mean:.3f}\n'
            stats_text += f'Std: {stats.report.step_similarity.std:.3f}\n'
            stats_text += f'Median: {stats.report.step_similarity.median:.3f}\n'
            stats_text += f'Pass Rate: {stats.report.step_similarity.pass_rate:.1%}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. CoT Effect Ratio 分布
        if stats.report.step_ppl.values:
            ax = axes[0, 1]
            ratios = stats.report.step_ppl.values
            ax.hist(ratios, bins=50, alpha=0.7, color='seagreen', edgecolor='black')
            ax.axvline(stats.config.cot_effect_ratio_threshold, color='red',
                      linestyle='--', label=f'Threshold: {stats.config.cot_effect_ratio_threshold}')
            ax.axvline(stats.report.step_ppl.mean, color='green',
                      linestyle='--', label=f'Mean: {stats.report.step_ppl.mean:.3f}')
            ax.axvline(1.0, color='orange', linestyle=':', label='Baseline (1.0)')
            ax.set_xlabel('CoT Effect Ratio')
            ax.set_ylabel('Count')
            ax.set_title('CoT Effect Ratio Distribution')
            ax.legend()
            stats_text = f'Mean: {stats.report.step_ppl.mean:.3f}\n'
            stats_text += f'Std: {stats.report.step_ppl.std:.3f}\n'
            stats_text += f'Median: {stats.report.step_ppl.median:.3f}\n'
            stats_text += f'Pass Rate: {stats.report.step_ppl.pass_rate:.1%}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 3. Original FSF 分布
        if stats.report.step_fsf.values:
            ax = axes[1, 0]
            fsfs = stats.report.step_fsf.values
            ax.hist(fsfs, bins=50, alpha=0.7, color='coral', edgecolor='black')
            ax.axvline(stats.report.step_fsf.mean, color='green',
                      linestyle='--', label=f'Mean: {stats.report.step_fsf.mean:.3f}')
            ax.set_xlabel('Original FSF')
            ax.set_ylabel('Count')
            ax.set_title('Original FSF Distribution')
            ax.legend()
            stats_text = f'Mean: {stats.report.step_fsf.mean:.3f}\n'
            stats_text += f'Std: {stats.report.step_fsf.std:.3f}\n'
            stats_text += f'Median: {stats.report.step_fsf.median:.3f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. 过滤步骤通过率对比
        ax = axes[1, 1]
        steps = ['Similarity', 'CoT Effect\nRatio', 'FSF']
        pass_rates = [
            stats.report.step_similarity.pass_rate * 100,
            stats.report.step_ppl.pass_rate * 100,
            0.0  # FSF 没有传统意义上的通过率
        ]
        colors = ['steelblue', 'seagreen', 'lightgray']
        bars = ax.bar(steps, pass_rates, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('Filter Pass Rate Comparison')
        ax.set_ylim(0, 100)

        # 在柱状图上显示数值
        for bar, rate in zip(bars, pass_rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rate:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'distribution_plots.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Report] Distribution plots saved to: {plot_path}")

    except ImportError as e:
        print(f"  [Report] Warning: matplotlib/seaborn not available, skipping plots: {e}")
        print(f"  [Report] Install with: pip install matplotlib seaborn")


# =============================================================================
# 工具函数
# =============================================================================

def extract_think_from_rv(content: str) -> str:
    """从RV格式的content中提取think部分"""
    return content.split("<think>")[1].split("</think>")[0].strip()


def extract_answer_from_rv(content: str) -> str:
    """从RV格式的content中提取answer部分（think之后的内容）"""
    parts = content.split("</think>")
    if len(parts) > 1:
        return parts[1].strip()
    return ""


def compute_similarity_penalty(original: str, rewrite: str) -> float:
    """
    计算相似度惩罚值
    
    使用简单的字符级相似度计算，可以替换为更复杂的算法
    返回值范围 [0, 1]，越接近1表示越相似
    """
    # 简单的字符级相似度（可以使用更复杂的算法如TF-IDF、BERT embedding等）
    original_clean = re.sub(r'\s+', '', original.lower())
    rewrite_clean = re.sub(r'\s+', '', rewrite.lower())
    
    if len(original_clean) == 0 or len(rewrite_clean) == 0:
        return 1.0 if original_clean == rewrite_clean else 0.0
    
    # 计算最长公共子序列(LCS)相似度
    m, n = len(original_clean), len(rewrite_clean)
    
    # 使用简单的序列匹配（作为近似）
    # 实际可以使用difflib.SequenceMatcher
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, original_clean, rewrite_clean).ratio()
    
    return similarity


def save_jsonl(data: List[Dict], path: str, append: bool = False):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: str) -> List[Dict]:
    """从JSONL文件加载数据"""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def get_checkpoint_path(output_dir: str, step: int) -> str:
    """获取检查点文件路径"""
    return os.path.join(output_dir, 'ckpt', f'ckpt_{step}.jsonl')


def get_progress_path(output_dir: str, step: int) -> str:
    """获取进度文件路径"""
    return os.path.join(output_dir, 'ckpt', f'progress_{step}.json')


# =============================================================================
# 步骤1: 模型格式转换
# =============================================================================

def convert_model_to_hf(model_path: str) -> str:
    """
    将FSDP checkpoint转换为Huggingface格式 (使用VERL的model_merger库)

    输入: /path/to/model/global_step_100
    输出: /path/to/model/global_step_100/hf

    如果hf目录已存在且包含必要文件，则跳过转换
    """
    hf_path = os.path.join(model_path, 'hf')

    # 检查是否已存在转换后的模型
    if os.path.exists(hf_path) and os.path.isdir(hf_path):
        files = os.listdir(hf_path)
        has_config = 'config.json' in files
        has_tokenizer = 'tokenizer.json' in files or 'tokenizer_config.json' in files
        has_model = any(f.endswith('.bin') or f.endswith('.safetensors') for f in files)

        if has_config and has_tokenizer and has_model:
            print(f"  [Model Convert] HF format already exists at {hf_path}")
            return hf_path

    print(f"  [Model Convert] Converting FSDP checkpoint to HF format using VERL model_merger...")
    print(f"    Source: {model_path}")
    print(f"    Target: {hf_path}")

    try:
        os.makedirs(hf_path, exist_ok=True)

        # 使用 VERL 的 FSDPModelMerger
        actor_dir = os.path.join(model_path, 'actor')
        hf_config_path = os.path.join(actor_dir, 'huggingface')

        # 检查必要的文件是否存在
        fsdp_config_path = os.path.join(actor_dir, 'fsdp_config.json')
        if not os.path.exists(fsdp_config_path):
            raise FileNotFoundError(f"FSDP config not found: {fsdp_config_path}")

        # 创建 ModelMergerConfig
        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=actor_dir,
            hf_model_config_path=hf_config_path,  # 必须设置，否则会尝试从 HF 下载
            target_dir=hf_path,
            trust_remote_code=False,
        )

        # 创建 merger 并执行转换
        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        merger.cleanup()

        print(f"  [Model Convert] Conversion completed: {hf_path}")
        return hf_path

    except Exception as e:
        import traceback
        print(f"  [Model Convert] Error during conversion: {e}")
        print(traceback.format_exc())
        print(f"  [Model Convert] Please manually convert the model to HF format")
        raise


# =============================================================================
# 步骤2: 构建Prompt
# =============================================================================

def build_prompt(question: str, thought: str) -> Tuple[str, str]:
    """
    构建 system_prompt 和 user_prompt
    """
    system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' is a verified and valid derivation. Using it as a reference, rewrite the reasoning with a better logical structure and a more clear delivery.

Follow these guidelines:
1. **Logical Optimization**: Keep logic direct and coherent; eliminate redundancy without losing necessary complexity.
2. **Step-by-Step Delivery**: Present the reasoning in a clear, sequential, step-by-step manner.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags. Output nothing else."""

    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}"""

    return system_prompt, user_prompt


# =============================================================================
# Ray 数据并行改写客户端
# =============================================================================

class vLLMWorkerActor:
    """
    Ray Actor: 每个 actor 运行一个 vLLM 实例
    """

    def __init__(
        self,
        worker_id: int,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int = 16384,
    ):
        import os
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.worker_id = worker_id
        self.model_path = model_path

        print(f"[vLLMWorkerActor {worker_id}] Initializing...")

        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 初始化 vLLM engine
        self.llm_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )

        # System prompt (固定不变)
        self.system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' is a verified and valid derivation. Using it as a reference, rewrite the reasoning with a better logical structure and a more clear delivery.

Follow these guidelines:
1. **Logical Optimization**: Keep logic direct and coherent; eliminate redundancy without losing necessary complexity.
2. **Step-by-Step Delivery**: Present the reasoning in a clear, sequential, step-by-step manner.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags. Output nothing else."""

        print(f"[vLLMWorkerActor {worker_id}] Initialization complete")

    def _build_prompt(self, task: Dict) -> str:
        """构建单个任务的 prompt"""
        question = task['question']
        thought = task['thought']

        user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _extract_rewrite_reasoning(self, generated_text: str) -> Optional[str]:
        """从生成文本中提取 rewrite_reasoning"""
        if '<rewrite_reasoning>' in generated_text and '</rewrite_reasoning>' in generated_text:
            return generated_text.split('<rewrite_reasoning>')[1].split('</rewrite_reasoning>')[0].strip()
        return None

    def process_tasks(
        self,
        tasks: List[Dict],
        sampling_params_dict: Dict[str, Any],
    ) -> List[Dict]:
        """
        处理一批任务，支持格式提取失败时重试（最多3次）

        Args:
            tasks: 任务列表，每个任务包含 task_id, sample_idx, question, thought, answer
            sampling_params_dict: 采样参数字典

        Returns:
            处理结果列表
        """
        if not tasks:
            return []

        from vllm import SamplingParams

        # 构建 SamplingParams
        sp_params = SamplingParams(**sampling_params_dict)

        # 首次处理所有任务
        results = {}
        retry_counts = {task['task_id']: 0 for task in tasks}

        # 构建 prompts
        prompts = [self._build_prompt(task) for task in tasks]

        # vLLM 批量推理
        try:
            outputs = self.llm_engine.generate(prompts, sp_params)

            for task, output in zip(tasks, outputs):
                generated_text = output.outputs[0].text if output.outputs else ""
                rewrite_reasoning = self._extract_rewrite_reasoning(generated_text)

                results[task['task_id']] = {
                    'task_id': task['task_id'],
                    'sample_idx': task['sample_idx'],
                    'question': task['question'],
                    'original_thought': task['thought'],
                    'answer': task.get('answer', ''),
                    'rewrite_output': generated_text,
                    'rewrite_reasoning': rewrite_reasoning,
                    'success': rewrite_reasoning is not None
                }

        except Exception as e:
            # vLLM 推理异常，所有任务标记为失败
            for task in tasks:
                results[task['task_id']] = {
                    'task_id': task['task_id'],
                    'sample_idx': task['sample_idx'],
                    'question': task['question'],
                    'original_thought': task['thought'],
                    'answer': task.get('answer', ''),
                    'rewrite_output': str(e),
                    'rewrite_reasoning': None,
                    'success': False,
                    'error': str(e)
                }
            return list(results.values())

        # 对格式提取失败的任务进行重试（最多3次）
        max_retries = 3

        for retry_round in range(max_retries):
            # 找出需要重试的任务：格式提取失败 且 未达到最大重试次数
            tasks_to_retry = [
                task for task in tasks
                if not results[task['task_id']]['success']
                and retry_counts[task['task_id']] < max_retries
            ]

            if not tasks_to_retry:
                break

            for task in tasks_to_retry:
                retry_counts[task['task_id']] += 1

            print(f"[vLLMWorkerActor {self.worker_id}] Retry round {retry_round + 1}/{max_retries} for {len(tasks_to_retry)} format-extraction failed tasks")

            # 构建重试 prompts
            retry_prompts = [self._build_prompt(task) for task in tasks_to_retry]

            try:
                retry_outputs = self.llm_engine.generate(retry_prompts, sp_params)

                for task, output in zip(tasks_to_retry, retry_outputs):
                    generated_text = output.outputs[0].text if output.outputs else ""
                    rewrite_reasoning = self._extract_rewrite_reasoning(generated_text)

                    if rewrite_reasoning is not None:
                        # 重试成功，更新结果
                        results[task['task_id']] = {
                            'task_id': task['task_id'],
                            'sample_idx': task['sample_idx'],
                            'question': task['question'],
                            'original_thought': task['thought'],
                            'answer': task.get('answer', ''),
                            'rewrite_output': generated_text,
                            'rewrite_reasoning': rewrite_reasoning,
                            'success': True,
                            'retried': retry_counts[task['task_id']]
                        }
                        print(f"[vLLMWorkerActor {self.worker_id}] Task {task['task_id']} succeeded on retry {retry_counts[task['task_id']]}")
                    else:
                        # 重试仍然失败，更新输出文本
                        results[task['task_id']]['rewrite_output'] = generated_text
                        results[task['task_id']]['retried'] = retry_counts[task['task_id']]

            except Exception as e:
                # 重试也失败了，记录错误但继续
                for task in tasks_to_retry:
                    results[task['task_id']]['error'] = str(e)
                    results[task['task_id']]['retried'] = retry_counts[task['task_id']]
                print(f"[vLLMWorkerActor {self.worker_id}] Retry round {retry_round + 1} failed: {e}")

        return list(results.values())

    def ping(self) -> str:
        """健康检查"""
        return f"Worker {self.worker_id} is alive"

    def shutdown(self):
        """清理资源"""
        if hasattr(self, 'llm_engine'):
            del self.llm_engine
        print(f"[vLLMWorkerActor {self.worker_id}] Shutdown complete")


class RayRewriteClient:
    """
    Ray 改写客户端 - 使用 Ray Actor 实现数据并行

    将所有 GPU 分成多组，每组运行一个独立的 vLLM Actor，处理不同的数据子集。

    例如：8 张卡，4 个实例，每组 2 张卡 (TP=2)
        - Actor 0: GPU 0,1 (num_gpus=2)
        - Actor 1: GPU 2,3 (num_gpus=2)
        - Actor 2: GPU 4,5 (num_gpus=2)
        - Actor 3: GPU 6,7 (num_gpus=2)
    """

    def __init__(
        self,
        model_path: str,
        num_instances: int,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.85,
        total_gpus: int = 8,
        micro_batch_size: int = 1,
        max_model_len: int = 16384,
    ):
        """
        Args:
            model_path: 模型路径
            num_instances: vLLM Actor 数量
            tensor_parallel_size: 每个 Actor 的张量并行大小
            gpu_memory_utilization: GPU 显存利用率
            total_gpus: 总 GPU 数量
            micro_batch_size: 每次调用处理的任务数
            max_model_len: 最大模型长度
        """
        self.model_path = model_path
        self.num_instances = num_instances
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.total_gpus = total_gpus
        self.micro_batch_size = micro_batch_size
        self.max_model_len = max_model_len

        # 验证配置
        if num_instances * tensor_parallel_size > total_gpus:
            raise ValueError(
                f"num_instances ({num_instances}) * tensor_parallel_size ({tensor_parallel_size}) "
                f"cannot exceed total_gpus ({total_gpus})"
            )

        self.actors = []
        self.is_initialized = False

        # Ray 需要根据 tensor_parallel_size 动态设置 num_gpus
        # 创建 Actor 类的动态版本
        self.actor_class = ray.remote(num_gpus=tensor_parallel_size)(vLLMWorkerActor)

    def initialize(self):
        """初始化 Ray Actors"""
        if self.is_initialized:
            return

        # 检查 Ray 是否已初始化
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        print(f"\n  [RayRewriteClient] Initializing...")
        print(f"    Total GPUs: {self.total_gpus}")
        print(f"    Number of actors: {self.num_instances}")
        print(f"    GPUs per actor (TP): {self.tensor_parallel_size}")
        print(f"    Micro batch size: {self.micro_batch_size}")

        # 创建 Actors
        for i in range(self.num_instances):
            actor = self.actor_class.remote(
                worker_id=i,
                model_path=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            self.actors.append(actor)

        # 等待所有 actor 初始化完成
        print(f"  [RayRewriteClient] Waiting for actors to initialize...")
        for i, actor in enumerate(self.actors):
            ray.get(actor.ping.remote())

        self.is_initialized = True
        print(f"  [RayRewriteClient] All {self.num_instances} actors ready")

    def process_samples(
        self,
        samples: List[Dict],
        n_rewrites: int,
        max_new_tokens: int = 15000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = -1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        ckpt_path: Optional[str] = None,
        progress_path: Optional[str] = None,
        process_batch_size: int = 32,
    ) -> List[Dict]:
        """
        处理多个样本，每个样本生成 n_rewrites 次改写

        Args:
            samples: 样本列表，每个样本包含 sample_idx, question, thought, answer
            n_rewrites: 每个样本生成的改写次数
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            presence_penalty: 存在惩罚
            frequency_penalty: 频率惩罚
            ckpt_path: 检查点文件路径
            progress_path: 进度文件路径
            process_batch_size: 每处理这么多样本后保存一次 ckpt

        Returns:
            处理后的样本列表
        """
        self.initialize()

        # 构建采样参数
        sampling_params = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k > 0:
            sampling_params["top_k"] = top_k
        if presence_penalty != 0.0:
            sampling_params["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            sampling_params["frequency_penalty"] = frequency_penalty

        # 准备所有任务
        all_tasks = []
        task_id = 0
        for sample in samples:
            for _ in range(n_rewrites):
                all_tasks.append({
                    'task_id': task_id,
                    'sample_idx': sample['sample_idx'],
                    'question': sample['question'],
                    'thought': sample['thought'],
                    'answer': sample.get('answer', ''),
                })
                task_id += 1

        total_tasks = len(all_tasks)
        num_actors = len(self.actors)

        # 将任务分配给各个 actors (round-robin)
        actor_tasks = [[] for _ in range(num_actors)]
        for i, task in enumerate(all_tasks):
            actor_tasks[i % num_actors].append(task)

        # 按 micro_batch_size 分批
        actor_batches = []
        for tasks in actor_tasks:
            batches = []
            for i in range(0, len(tasks), self.micro_batch_size):
                batch = tasks[i:i + self.micro_batch_size]
                batches.append(batch)
            actor_batches.append(batches)

        print(f"  [RayRewriteClient] Distributed {total_tasks} tasks across {num_actors} actors")

        # 收集结果
        results = {i: [] for i in range(len(samples))}
        sample_task_counts = {i: 0 for i in range(len(samples))}
        completed_samples = []
        write_batch = []

        # 获取起始样本索引
        start_sample_idx = samples[0].get('sample_idx', 0) if samples else 0

        # 使用异步方式提交任务并收集结果
        pending_refs = []
        actor_batch_indices = [0 for _ in range(num_actors)]  # 每个 actor 当前处理到第几个 batch

        # 初始提交：每个 actor 提交第一批
        for actor_idx, batches in enumerate(actor_batches):
            if batches:
                ref = self.actors[actor_idx].process_tasks.remote(
                    batches[0], sampling_params
                )
                pending_refs.append((actor_idx, 0, ref))
                actor_batch_indices[actor_idx] = 1

        with tqdm(total=total_tasks, desc="Rewrite (Ray)", unit="task", dynamic_ncols=True) as pbar:
            while pending_refs:
                # 等待任意一个任务完成
                # 提取所有 ObjectRef 用于 ray.wait
                pending_object_refs = [r for _, _, r in pending_refs]
                ready_refs, remaining_object_refs = ray.wait(pending_object_refs, num_returns=1)

                for ready_ref in ready_refs:
                    # 在 pending_refs 中找到对应的元组
                    found_idx = None
                    for i, (actor_idx, batch_idx, ref) in enumerate(pending_refs):
                        if ref == ready_ref:
                            found_idx = i
                            break

                    if found_idx is not None:
                        actor_idx, batch_idx, _ = pending_refs.pop(found_idx)
                    else:
                        # 不应该发生，但作为保险
                        print(f"  Warning: Cannot find pending ref for {ready_ref}")
                        continue

                    # 获取结果
                    try:
                        batch_results = ray.get(ready_ref)

                        # 处理结果
                        for result in batch_results:
                            sample_idx = result['sample_idx']
                            results[sample_idx].append(result)
                            sample_task_counts[sample_idx] += 1
                            pbar.update(1)

                            # 检查样本是否完成
                            if sample_task_counts[sample_idx] == n_rewrites:
                                write_batch.append(sample_idx)
                                completed_samples.append(sample_idx)

                        # 提交该 actor 的下一批
                        next_batch_idx = actor_batch_indices[actor_idx]
                        if next_batch_idx < len(actor_batches[actor_idx]):
                            next_ref = self.actors[actor_idx].process_tasks.remote(
                                actor_batches[actor_idx][next_batch_idx], sampling_params
                            )
                            pending_refs.append((actor_idx, next_batch_idx, next_ref))
                            actor_batch_indices[actor_idx] = next_batch_idx + 1

                        # 批量写入
                        completed_tasks = sum(sample_task_counts.values())
                        pbar.set_postfix({'completed': f'{completed_tasks}/{total_tasks}', 'samples': f'{len(completed_samples)}/{len(samples)}'})

                        if len(write_batch) >= process_batch_size or completed_tasks == total_tasks:
                            batch_write_results = []
                            for idx in sorted(write_batch):
                                sample = samples[idx]
                                rewrite_results = []
                                for r in results.get(idx, []):
                                    rewrite_results.append({
                                        'rewrite_output': r['rewrite_output'],
                                        'rewrite_reasoning': r['rewrite_reasoning'],
                                        'success': r['success']
                                    })

                                result_item = {
                                    'sample_idx': sample['sample_idx'],
                                    'question': sample['question'],
                                    'original_thought': sample['thought'],
                                    'answer': sample.get('answer', ''),
                                    'rewrite_results': rewrite_results
                                }
                                batch_write_results.append(result_item)

                            # 写入文件
                            if ckpt_path:
                                is_first_batch = (start_sample_idx == samples[0].get('sample_idx', 0) and len(completed_samples) == len(write_batch))
                                save_jsonl(batch_write_results, ckpt_path, append=(not is_first_batch))

                            # 更新进度
                            if progress_path:
                                completed_count = len(completed_samples)
                                last_idx = start_sample_idx + completed_count
                                with open(progress_path, 'w') as f:
                                    json.dump({'last_completed_idx': last_idx}, f)

                            pbar.write(f"  Saved batch of {len(write_batch)} samples ({len(completed_samples)}/{len(samples)} completed)")
                            write_batch.clear()

                    except Exception as e:
                        print(f"  Error processing batch: {e}")
                        # 继续提交下一批
                        next_batch_idx = actor_batch_indices[actor_idx]
                        if next_batch_idx < len(actor_batches[actor_idx]):
                            next_ref = self.actors[actor_idx].process_tasks.remote(
                                actor_batches[actor_idx][next_batch_idx], sampling_params
                            )
                            pending_refs.append((actor_idx, next_batch_idx, next_ref))
                            actor_batch_indices[actor_idx] = next_batch_idx + 1


        # 整理最终结果
        final_results = []
        for i, sample in enumerate(samples):
            rewrite_results = []
            for r in results.get(i, []):
                rewrite_results.append({
                    'rewrite_output': r['rewrite_output'],
                    'rewrite_reasoning': r['rewrite_reasoning'],
                    'success': r['success']
                })

            result_item = {
                'sample_idx': sample['sample_idx'],
                'question': sample['question'],
                'original_thought': sample['thought'],
                'answer': sample.get('answer', ''),
                'rewrite_results': rewrite_results
            }
            final_results.append(result_item)

        return final_results

    def shutdown(self):
        """关闭所有 Actors"""
        if self.actors:
            # 调用每个 actor 的 shutdown 方法
            shutdown_refs = [actor.shutdown.remote() for actor in self.actors]
            ray.get(shutdown_refs)
            self.actors = []

        self.is_initialized = False


def extract_rewrite_reasoning(output: str) -> Optional[str]:
    """从输出中提取改写后的reasoning"""
    try:
        if '<rewrite_reasoning>' in output and '</rewrite_reasoning>' in output:
            reasoning = output.split('<rewrite_reasoning>')[1].split('</rewrite_reasoning>')[0].strip()
            return reasoning
    except Exception:
        pass
    return None


def step_rewrite(
    config: RewriteConfig,
    samples: List[Dict],
    rewrite_client: 'RayRewriteClient'
) -> List[Dict]:
    """
    步骤3: 调用改写模型生成n次结果

    使用 Ray 数据并行模式。

    支持断点续传。
    """
    print("\n" + "=" * 60)
    print("Step 3: Rewrite Generation")
    print("=" * 60)

    ckpt_path = get_checkpoint_path(config.output_dir, 1)
    progress_path = get_progress_path(config.output_dir, 1)

    # 检查是否有断点
    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_completed_idx', 0)
        print(f"  [Resume] Found checkpoint, resuming from sample {start_idx}")

    # 加载已保存的结果
    results = []
    if os.path.exists(ckpt_path):
        results = load_jsonl(ckpt_path)
        print(f"  [Resume] Loaded {len(results)} existing results")

    total_samples = len(samples)
    remaining_samples = total_samples - start_idx

    print(f"  Total samples: {total_samples}")
    print(f"  Already processed: {start_idx}")
    print(f"  Remaining: {remaining_samples}")
    print(f"  N rewrites per sample: {config.n_rewrites}")
    print(f"  Mode: Ray data parallel")
    print(f"  Num actors: {rewrite_client.num_instances}")
    print(f"  Process batch size: {config.process_batch_size}")

    # 获取待处理的样本
    remaining_samples_list = samples[start_idx:]

    if not remaining_samples_list:
        print("  No samples to process.")
        return results

    # 为样本添加 sample_idx
    samples_to_process = [
        {**s, 'sample_idx': start_idx + i}
        for i, s in enumerate(remaining_samples_list)
    ]

    # 批量处理（使用 process_batch_size 控制保存策略）
    new_results = rewrite_client.process_samples(
        samples=samples_to_process,
        n_rewrites=config.n_rewrites,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        ckpt_path=ckpt_path,
        progress_path=progress_path,
        process_batch_size=config.process_batch_size,
    )

    # 结果已在 process_samples 中写入文件，这里只用于返回
    results.extend(new_results)

    print(f"  Completed. Results saved to: {ckpt_path}")
    return results


# =============================================================================
# 步骤4: Similarity Penalty过滤
# =============================================================================

def compute_similarity_worker(args: Tuple[int, Dict, float]) -> Tuple[int, List[Dict]]:
    """多进程worker：计算单个样本的similarity，不删除结果只标记"""
    idx, sample, threshold = args

    original_thought = sample['original_thought']
    rewrite_results = sample.get('rewrite_results', [])

    # 不删除结果，只添加标记
    for result in rewrite_results:
        rewrite_reasoning = result.get('rewrite_reasoning')

        if not rewrite_reasoning:
            result['similarity_penalty'] = None
            result['similarity_passed'] = False
            result['filtered'] = True
            continue

        similarity = compute_similarity_penalty(original_thought, rewrite_reasoning)

        result['similarity_penalty'] = similarity
        passed = similarity < threshold
        result['similarity_passed'] = passed
        # 如果之前没有过滤标记，或者当前过滤失败，则标记为 filtered
        if passed:
            result['filtered'] = result.get('filtered', False)
        else:
            result['filtered'] = True

    return idx, rewrite_results


def step_similarity_filter(
    config: RewriteConfig,
    samples: List[Dict],
    stats_collector: Optional[StatisticsCollector] = None
) -> List[Dict]:
    """
    步骤4: 计算similarity_penalty并标记过滤

    不删除结果，只添加 filtered 标记
    """
    if config.skip_similarity:
        print("\n  [Skip] Similarity filter skipped")
        return samples

    print("\n" + "=" * 60)
    print("Step 4: Similarity Penalty Filter")
    print("=" * 60)
    print(f"  Threshold: {config.similarity_penalty_threshold}")
    print(f"  Workers: {config.num_workers}")

    # 准备多进程参数
    args_list = [
        (i, sample, config.similarity_penalty_threshold)
        for i, sample in enumerate(samples)
    ]

    results = [None] * len(samples)

    # 多进程处理，使用tqdm显示进度
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {executor.submit(compute_similarity_worker, args): args[0] for args in args_list}

        pbar = tqdm(total=len(samples), desc="Similarity", unit="sample", dynamic_ncols=True)
        for future in as_completed(futures):
            idx, rewrite_results = future.result()
            results[idx] = rewrite_results
            pbar.update(1)
        pbar.close()

    # 更新样本
    filtered_samples = []
    for i, sample in enumerate(samples):
        sample_copy = sample.copy()
        sample_copy['rewrite_results'] = results[i] if results[i] else []
        filtered_samples.append(sample_copy)

    # 统计
    total_rewrites = sum(len(s['rewrite_results']) for s in filtered_samples)
    filtered_count = sum(
        sum(1 for r in s['rewrite_results'] if r.get('filtered', False))
        for s in filtered_samples
    )
    passed_count = total_rewrites - filtered_count

    print(f"  Total rewrites: {total_rewrites}")
    print(f"  Passed: {passed_count} ({passed_count / total_rewrites * 100:.1f}%)")
    print(f"  Filtered: {filtered_count} ({filtered_count / total_rewrites * 100:.1f}%)")

    # 收集统计信息
    if stats_collector:
        stats_collector.update_similarity_stats(filtered_samples)

    return filtered_samples


# =============================================================================
# 步骤5: PPL过滤
# =============================================================================

from verl.experimental.reward.cot_rewriter.sglang_student_reward import SGLangStudentRewardClient


def step_ppl_filter(
    config: RewriteConfig,
    samples: List[Dict],
    stats_collector: Optional[StatisticsCollector] = None
) -> List[Dict]:
    """
    步骤5: 计算 CoT Effect Ratio 并过滤

    cot_effect_ratio = cot_effect(rewrite) / cot_effect(original)
    其中 cot_effect(cot) = avg_prob(Q,CoT,A) - avg_prob(Q,A)

    支持断点续传
    """
    if config.skip_ppl:
        print("\n  [Skip] PPL filter skipped")
        return samples

    print("\n" + "=" * 60)
    print("Step 5: CoT Effect Ratio Filter")
    print("=" * 60)
    print(f"  Student Model: {config.student_model_path}")
    print(f"  CoT Effect Ratio Threshold: {config.cot_effect_ratio_threshold} (>{config.cot_effect_ratio_threshold} to pass)")
    print(f"  CoT Effect Ratio Ceiling: {config.cot_effect_ratio_ceiling}")

    ckpt_path = get_checkpoint_path(config.output_dir, 2)
    progress_path = get_progress_path(config.output_dir, 2)

    # 检查断点
    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_completed_idx', 0)
        print(f"  [Resume] Found checkpoint, resuming from sample {start_idx}")

    # 加载已处理的结果
    processed_samples = []
    if os.path.exists(ckpt_path):
        processed_samples = load_jsonl(ckpt_path)
        print(f"  [Resume] Loaded {len(processed_samples)} existing results")

    # 初始化SGLang客户端，使用 cot_effect_ratio 模式
    print("  Initializing SGLang student reward client (cot_effect_ratio mode)...")
    student_client = SGLangStudentRewardClient(
        model_path=config.student_model_path,
        tensor_parallel_size=config.sglang_tensor_parallel_size,
        reward_mode="cot_effect_ratio",
        cot_effect_ratio_ceiling=config.cot_effect_ratio_ceiling,
        use_reward_delta=True,
        devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        micro_batch_size=config.sglang_micro_batch_size,
    )
    student_client.initialize()

    total_samples = len(samples)
    all_cot_effect_ratios = []

    # 创建进度条
    pbar = tqdm(
        range(start_idx, total_samples),
        initial=start_idx,
        total=total_samples,
        desc="CoT Effect Ratio Filter",
        unit="sample",
        dynamic_ncols=True
    )

    for idx in pbar:
        sample = samples[idx]
        question = sample['question']
        original_thought = sample['original_thought']
        answer = sample.get('answer', '')
        rewrite_results = sample.get('rewrite_results', [])

        # 更新进度条
        pbar.set_postfix({'sample': f'{idx}/{total_samples}'})

        # 提取有效的rewrite reasoning
        rewrite_cots = []
        result_indices = []

        for r_idx, result in enumerate(rewrite_results):
            if result.get('rewrite_reasoning'):
                rewrite_cots.append(result['rewrite_reasoning'])
                result_indices.append(r_idx)

        if rewrite_cots:
            # 计算 cognitive reward (cot_effect_ratio 模式)
            questions_batch = [question] * len(rewrite_cots)
            answers_batch = [answer] * len(rewrite_cots)
            original_cots_batch = [original_thought] * len(rewrite_cots)

            try:
                reward_result = student_client.compute_cognitive_reward_batch(
                    questions=questions_batch,
                    rewrite_cots=rewrite_cots,
                    answers=answers_batch,
                    original_cots=original_cots_batch,
                )

                # cot_effect_ratio = rewards (在 cot_effect_ratio 模式下)
                cot_effect_ratios = reward_result.get('rewards', [])

                # 更新结果
                for i, r_idx in enumerate(result_indices):
                    if i < len(cot_effect_ratios):
                        ratio = cot_effect_ratios[i]
                        sample['rewrite_results'][r_idx]['cot_effect_ratio'] = ratio
                        # ratio > 1 表示 rewrite 比 original 更有帮助
                        # ratio > threshold 则通过过滤
                        passed = ratio > config.cot_effect_ratio_threshold
                        sample['rewrite_results'][r_idx]['ppl_passed'] = passed
                        # 同时保留 ppl_relative_improvement 字段用于兼容性
                        sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = ratio - 1.0  # 转换为改进率
                        # 更新 filtered 标记：如果之前未过滤且当前通过，则保持未过滤；否则标记为过滤
                        if not passed:
                            sample['rewrite_results'][r_idx]['filtered'] = True
                        else:
                            # 如果之前已经被过滤，保持过滤状态
                            sample['rewrite_results'][r_idx]['filtered'] = sample['rewrite_results'][r_idx].get('filtered', False)
                        all_cot_effect_ratios.append(ratio)
                    else:
                        sample['rewrite_results'][r_idx]['cot_effect_ratio'] = 0.0
                        sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = -float('inf')
                        sample['rewrite_results'][r_idx]['ppl_passed'] = False
                        sample['rewrite_results'][r_idx]['filtered'] = True

            except Exception as e:
                pbar.set_postfix({'sample': f'{idx}/{total_samples}', 'error': str(e)[:30]})
                for r_idx in result_indices:
                    sample['rewrite_results'][r_idx]['cot_effect_ratio'] = 0.0
                    sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = -float('inf')
                    sample['rewrite_results'][r_idx]['ppl_passed'] = False
                    sample['rewrite_results'][r_idx]['filtered'] = True

        processed_samples.append(sample)

        # 保存进度
        save_jsonl([sample], ckpt_path, append=(idx > 0))
        with open(progress_path, 'w') as f:
            json.dump({'last_completed_idx': idx + 1}, f)

        if (idx + 1) % 10 == 0 or idx == total_samples - 1:
            print(f"  Progress: {idx + 1}/{total_samples}")

    # 输出 CoT Effect Ratio 分布统计
    if all_cot_effect_ratios:
        print("\n  CoT Effect Ratio Distribution:")
        print(f"    Mean: {np.mean(all_cot_effect_ratios):.4f}")
        print(f"    Std: {np.std(all_cot_effect_ratios):.4f}")
        print(f"    Min: {np.min(all_cot_effect_ratios):.4f}")
        print(f"    Max: {np.max(all_cot_effect_ratios):.4f}")
        print(f"    Median: {np.median(all_cot_effect_ratios):.4f}")
        print(f"    Passed: {sum(1 for x in all_cot_effect_ratios if x > config.cot_effect_ratio_threshold)}/{len(all_cot_effect_ratios)}")

    print(f"  Completed. Results saved to: {ckpt_path}")

    # 收集统计信息
    if stats_collector:
        stats_collector.update_ppl_stats(all_cot_effect_ratios)

    # 清理
    student_client.shutdown()

    return processed_samples


# =============================================================================
# 步骤6: FSF过滤
# =============================================================================

from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient


def step_fsf_filter(
    config: RewriteConfig,
    samples: List[Dict],
    stats_collector: Optional[StatisticsCollector] = None
) -> List[Dict]:
    """
    步骤6: 计算FSF并过滤

    过滤条件：fsf_rewrite < fsf_original（改写比原文失败步骤更少）
    支持断点续传
    """
    if config.skip_fsf:
        print("\n  [Skip] FSF filter skipped")
        return samples

    print("\n" + "=" * 60)
    print("Step 6: FSF Filter (Compare with Original)")
    print("=" * 60)
    print(f"  FSF Judge Model: {config.fsf_judge_model_path}")
    print(f"  Filter: Keep rewrites with fsf_rewrite < fsf_original")

    ckpt_path = get_checkpoint_path(config.output_dir, 3)
    progress_path = get_progress_path(config.output_dir, 3)

    # 检查断点
    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_completed_idx', 0)
        print(f"  [Resume] Found checkpoint, resuming from sample {start_idx}")

    # 加载已处理的结果
    processed_samples = []
    if os.path.exists(ckpt_path):
        processed_samples = load_jsonl(ckpt_path)
        print(f"  [Resume] Loaded {len(processed_samples)} existing results")

    # 初始化FSF客户端
    print("  Initializing SGLang FSF client...")
    fsf_client = SGLangFSFClient(
        model_path=config.fsf_judge_model_path,
        tensor_parallel_size=config.sglang_tensor_parallel_size,
        devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        micro_batch_size=config.sglang_micro_batch_size,
    )
    fsf_client.initialize()

    total_samples = len(samples)
    original_fsfs_recorded = []  # 记录所有 original FSF 值
    rewrite_better_count = 0      # 记录有多少 rewrite 比 original 好

    # 创建进度条
    pbar = tqdm(
        range(start_idx, total_samples),
        initial=start_idx,
        total=total_samples,
        desc="FSF Filter",
        unit="sample",
        dynamic_ncols=True
    )

    for idx in pbar:
        sample = samples[idx]
        original_thought = sample.get('original_thought', '')
        rewrite_results = sample.get('rewrite_results', [])

        # 更新进度条
        pbar.set_postfix({'sample': f'{idx}/{total_samples}'})

        # 提取有效的rewrite reasoning
        rewrite_cots = []
        result_indices = []

        for r_idx, result in enumerate(rewrite_results):
            if result.get('rewrite_reasoning'):
                rewrite_cots.append(result['rewrite_reasoning'])
                result_indices.append(r_idx)

        if rewrite_cots and original_thought:
            try:
                # 使用 compute_logic_reward_batch 同时计算 original 和 rewrite 的 FSF
                # 这个方法会返回 original_fsfs, rewrite_fsfs, logic_rewards
                # logic_rewards = original_fsf - rewrite_fsf，正值表示 rewrite 更好
                fsf_result = fsf_client.compute_logic_reward_batch(
                    questions=[sample['question']] * len(rewrite_cots),
                    original_cots=[original_thought] * len(rewrite_cots),
                    rewrite_cots=rewrite_cots,
                )

                original_fsfs = fsf_result.get('original_fsfs', [])
                rewrite_fsfs = fsf_result.get('rewrite_fsfs', [])
                logic_rewards = fsf_result.get('logic_rewards', [])

                # 记录第一个 original FSF（同一个样本的所有 original FSF 应该相同）
                if original_fsfs and original_fsfs[0] is not None:
                    original_fsfs_recorded.append(original_fsfs[0])

                # 更新结果
                sample_has_better_rewrite = False
                for i, r_idx in enumerate(result_indices):
                    if i < len(rewrite_fsfs) and rewrite_fsfs[i] is not None:
                        rewrite_fsf = rewrite_fsfs[i]
                        original_fsf = original_fsfs[i] if i < len(original_fsfs) else None

                        sample['rewrite_results'][r_idx]['fsf'] = rewrite_fsf
                        sample['rewrite_results'][r_idx]['original_fsf'] = original_fsf

                        # 过滤条件：fsf_rewrite < fsf_original
                        # 即 logic_reward = original_fsf - rewrite_fsf > 0
                        if original_fsf is not None:
                            passed = rewrite_fsf < original_fsf
                        else:
                            passed = False  # original FSF 计算失败，不通过

                        sample['rewrite_results'][r_idx]['fsf_passed'] = passed
                        # 更新 filtered 标记
                        if not passed:
                            sample['rewrite_results'][r_idx]['filtered'] = True
                        else:
                            # 如果之前已经被过滤，保持过滤状态
                            sample['rewrite_results'][r_idx]['filtered'] = sample['rewrite_results'][r_idx].get('filtered', False)

                        if passed:
                            sample_has_better_rewrite = True
                    else:
                        sample['rewrite_results'][r_idx]['fsf'] = float('inf')
                        sample['rewrite_results'][r_idx]['original_fsf'] = None
                        sample['rewrite_results'][r_idx]['fsf_passed'] = False
                        sample['rewrite_results'][r_idx]['filtered'] = True

                if sample_has_better_rewrite:
                    rewrite_better_count += 1

            except Exception as e:
                pbar.set_postfix({'sample': f'{idx}/{total_samples}', 'error': str(e)[:30]})
                for r_idx in result_indices:
                    sample['rewrite_results'][r_idx]['fsf'] = float('inf')
                    sample['rewrite_results'][r_idx]['original_fsf'] = None
                    sample['rewrite_results'][r_idx]['fsf_passed'] = False
                    sample['rewrite_results'][r_idx]['filtered'] = True
        else:
            # 没有 rewrite 或 original 为空，标记所有为未通过
            for r_idx in result_indices:
                sample['rewrite_results'][r_idx]['fsf'] = None
                sample['rewrite_results'][r_idx]['original_fsf'] = None
                sample['rewrite_results'][r_idx]['fsf_passed'] = False
                sample['rewrite_results'][r_idx]['filtered'] = True

        processed_samples.append(sample)

        # 保存进度
        save_jsonl([sample], ckpt_path, append=(idx > 0))
        with open(progress_path, 'w') as f:
            json.dump({'last_completed_idx': idx + 1}, f)

    # 输出 FSF 统计
    if original_fsfs_recorded:
        print("\n  Original FSF Distribution:")
        print(f"    Mean: {np.mean(original_fsfs_recorded):.4f}")
        print(f"    Std: {np.std(original_fsfs_recorded):.4f}")
        print(f"    Min: {np.min(original_fsfs_recorded):.4f}")
        print(f"    Max: {np.max(original_fsfs_recorded):.4f}")
        print(f"    Median: {np.median(original_fsfs_recorded):.4f}")

    print(f"\n  Samples with at least one better rewrite: {rewrite_better_count}/{total_samples}")

    print(f"  Completed. Results saved to: {ckpt_path}")

    # 收集统计信息
    if stats_collector:
        stats_collector.update_fsf_stats(original_fsfs_recorded, rewrite_better_count, total_samples)

    # 清理
    fsf_client.shutdown()

    return processed_samples


# =============================================================================
# 步骤7: 选择最优结果并生成数据集
# =============================================================================

def select_best_rewrite(results: List[Dict]) -> Optional[Dict]:
    """
    选择最优的改写结果

    过滤条件：filtered 标记为 false（未过滤）

    优先级（按以下顺序排序）:
    1. cot_effect_ratio 最高（如果没有则用 ppl_relative_improvement）
    2. fsf 最低（相对于 original 的改进，即 original_fsf - fsf 最大）
    3. similarity_penalty 最低
    """
    if not results:
        return None

    # 首先过滤掉被标记为 filtered 的结果
    valid_results = []
    for r in results:
        # 必须成功提取reasoning
        if not r.get('rewrite_reasoning'):
            continue
        # 检查 filtered 标记（如果存在）
        if r.get('filtered', False):
            continue
        valid_results.append(r)

    if not valid_results:
        return None

    # 按优先级排序
    # 优先级: cot_effect_ratio (高->低), fsf_improvement (高->低), similarity_penalty (低->高)
    def sort_key(x):
        # 首选 cot_effect_ratio，如果没有则用 ppl_relative_improvement + 1 转换回 ratio
        cot_effect_ratio = x.get('cot_effect_ratio', None)
        if cot_effect_ratio is None:
            ppl_rel_imp = x.get('ppl_relative_improvement', -float('inf'))
            cot_effect_ratio = ppl_rel_imp + 1.0 if ppl_rel_imp > -float('inf') else 0.0

        # FSF 改进程度（original_fsf - fsf），越大越好
        fsf_improvement = 0.0
        original_fsf = x.get('original_fsf')
        fsf = x.get('fsf')
        if original_fsf is not None and fsf is not None:
            fsf_improvement = original_fsf - fsf  # 正值表示 rewrite 更好

        similarity = x.get('similarity_penalty', float('inf'))

        return (
            -cot_effect_ratio,      # 越高越好（用负号因为 sorted 是升序）
            -fsf_improvement,       # 越高越好（用负号）
            similarity              # 越低越好
        )

    sorted_results = sorted(valid_results, key=sort_key)

    return sorted_results[0]


def step_select_and_generate(
    config: RewriteConfig,
    samples: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    步骤7: 选择最优结果并生成新数据集
    
    返回新数据集和统计信息
    """
    print("\n" + "=" * 60)
    print("Step 7: Select Best Rewrite and Generate Dataset")
    print("=" * 60)
    
    new_dataset = []
    stats = {
        'total_samples': len(samples),
        'using_rewrite': 0,
        'using_original': 0,
        'rewrite_stats': {
            'total_rewrite_attempts': 0,
            'passed_similarity': 0,
            'passed_ppl': 0,
            'passed_fsf': 0,
        }
    }
    
    # 创建进度条
    pbar = tqdm(samples, desc="Generate", unit="sample", dynamic_ncols=True)
    
    for sample in pbar:
        question = sample['question']
        original_thought = sample['original_thought']
        answer = sample.get('answer', '')
        rewrite_results = sample.get('rewrite_results', [])
        
        stats['rewrite_stats']['total_rewrite_attempts'] += len(rewrite_results)
        stats['rewrite_stats']['passed_similarity'] += sum(
            1 for r in rewrite_results if r.get('similarity_passed', True)
        )
        stats['rewrite_stats']['passed_ppl'] += sum(
            1 for r in rewrite_results if r.get('ppl_passed', True)
        )
        stats['rewrite_stats']['passed_fsf'] += sum(
            1 for r in rewrite_results if r.get('fsf_passed', True)
        )
        
        # 选择最佳改写
        best_rewrite = select_best_rewrite(rewrite_results)
        
        if best_rewrite:
            # 使用改写结果
            rewrite_cot = best_rewrite['rewrite_reasoning']
            using_original = False
            stats['using_rewrite'] += 1
        else:
            # 没有符合要求的改写，使用原始thought
            rewrite_cot = original_thought
            using_original = True
            stats['using_original'] += 1
        
        # 构建新的conversations格式
        new_item = {
            'conversations': [
                {
                    'role': 'user',
                    'content': question
                },
                {
                    'role': 'assistant',
                    'content': f"<think>\n{rewrite_cot.strip()}\n</think>\n\n{answer.strip()}"
                }
            ],
            'metadata': {
                'using_original': using_original,
                'original_thought': original_thought,
                'answer': answer,
            }
        }
        
        # 如果使用了改写，添加改写相关的统计
        if best_rewrite:
            new_item['metadata']['rewrite_metrics'] = {
                'cot_effect_ratio': best_rewrite.get('cot_effect_ratio'),
                'ppl_relative_improvement': best_rewrite.get('ppl_relative_improvement'),
                'fsf': best_rewrite.get('fsf'),
                'original_fsf': best_rewrite.get('original_fsf'),
                'fsf_improvement': (best_rewrite.get('original_fsf') - best_rewrite.get('fsf'))
                    if best_rewrite.get('original_fsf') is not None and best_rewrite.get('fsf') is not None
                    else None,
                'similarity_penalty': best_rewrite.get('similarity_penalty'),
            }
        
        new_dataset.append(new_item)
    
    # 保存新数据集
    output_path = os.path.join(config.output_dir, 'rewritten_dataset.jsonl')
    save_jsonl(new_dataset, output_path)
    
    # 保存统计信息
    stats_path = os.path.join(config.output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Dataset Statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Using rewrite: {stats['using_rewrite']} ({stats['using_rewrite']/stats['total_samples']*100:.1f}%)")
    print(f"    Using original: {stats['using_original']} ({stats['using_original']/stats['total_samples']*100:.1f}%)")
    print(f"\n  Rewrite Filter Statistics:")
    print(f"    Total rewrite attempts: {stats['rewrite_stats']['total_rewrite_attempts']}")
    print(f"    Passed similarity: {stats['rewrite_stats']['passed_similarity']}")
    print(f"    Passed PPL: {stats['rewrite_stats']['passed_ppl']}")
    print(f"    Passed FSF: {stats['rewrite_stats']['passed_fsf']}")
    print(f"\n  Output saved to: {output_path}")
    print(f"  Stats saved to: {stats_path}")
    
    return new_dataset, stats


# =============================================================================
# 主函数
# =============================================================================

def load_and_parse_input_data(input_path: str) -> List[Dict]:
    """加载并解析输入数据"""
    print(f"\n  Loading input data from: {input_path}")
    
    data = load_jsonl(input_path)
    samples = []
    
    for item in data:
        conversations = item.get('conversations', [])
        if len(conversations) < 2:
            continue
        
        question = conversations[0].get('content', '')
        assistant_content = conversations[1].get('content', '')
        
        try:
            thought = extract_think_from_rv(assistant_content)
            answer = extract_answer_from_rv(assistant_content)
        except (IndexError, AttributeError):
            continue
        
        samples.append({
            'question': question,
            'thought': thought,
            'answer': answer,
            'original_item': item
        })
    
    print(f"  Loaded {len(samples)} valid samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Rewrite dataset using rewrite model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'rewrite', 'similarity', 'ppl', 'fsf', 'select'],
                        help='Run specific step only')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating distribution plots')
    args = parser.parse_args()

    # 加载配置
    print("=" * 60)
    print("Loading Configuration")
    print("=" * 60)
    config = load_config(args.config)
    print(f"  Config loaded from: {args.config}")
    print(f"  Output directory: {config.output_dir}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'ckpt'), exist_ok=True)

    # 保存配置副本
    config_copy_path = os.path.join(config.output_dir, 'config_used.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(vars(config), f)

    # 初始化统计收集器
    stats_collector = StatisticsCollector(config)

    # 步骤0: 转换模型格式
    print("\n" + "=" * 60)
    print("Step 0: Model Format Conversion")
    print("=" * 60)
    rewrite_model_hf_path = convert_model_to_hf(config.rewrite_model_path)
    print(f"  Rewrite model (HF): {rewrite_model_hf_path}")

    # 初始化改写模型客户端
    rewrite_client = None
    if args.step in ['all', 'rewrite']:
        # 自动计算实例数和总 GPU 数
        total_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(','))
        vllm_num_instances = total_gpus // config.vllm_tensor_parallel_size

        print(f"\n  [GPU Allocation]")
        print(f"    Total GPUs: {total_gpus}")
        print(f"    vLLM TP size: {config.vllm_tensor_parallel_size}")
        print(f"    Ray actors: {vllm_num_instances} (auto-calculated)")
        print(f"    Micro batch size: {config.vllm_micro_batch_size}")
        print(f"    Process batch size: {config.process_batch_size}")

        # 创建 Ray 改写客户端
        rewrite_client = RayRewriteClient(
            model_path=rewrite_model_hf_path,
            num_instances=vllm_num_instances,
            tensor_parallel_size=config.vllm_tensor_parallel_size,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            total_gpus=total_gpus,
            micro_batch_size=config.vllm_micro_batch_size,
        )

    # 加载输入数据
    samples = load_and_parse_input_data(config.input_data_path)

    # 根据步骤执行
    if args.step == 'all' or args.step == 'rewrite':
        # 步骤3: 改写生成
        samples = step_rewrite(config, samples, rewrite_client)
        rewrite_client.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # 从检查点加载
        ckpt_path = get_checkpoint_path(config.output_dir, 1)
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)
            print(f"\n  Loaded {len(samples)} samples from checkpoint 1")

    if args.step == 'all' or args.step == 'similarity':
        # 步骤4: Similarity过滤
        samples = step_similarity_filter(config, samples, stats_collector)
        # 保存中间结果
        save_jsonl(samples, get_checkpoint_path(config.output_dir, '1_5'))
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, '1_5')
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)

    if args.step == 'all' or args.step == 'ppl':
        # 步骤5: PPL过滤
        samples = step_ppl_filter(config, samples, stats_collector)
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, 2)
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)

    if args.step == 'all' or args.step == 'fsf':
        # 步骤6: FSF过滤
        samples = step_fsf_filter(config, samples, stats_collector)
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, 3)
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)

    if args.step == 'all' or args.step == 'select':
        # 步骤7: 选择和生成
        new_dataset, stats = step_select_and_generate(config, samples)
        # 更新最终统计
        stats_collector.update_final_stats(stats)

    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)

    # 生成报告
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)

    # 保存详细统计信息到 JSON
    stats_collector.save_to_json(config.output_dir)

    # 绘制分布图
    if not args.no_plots:
        save_distribution_plots(stats_collector, config.output_dir)


if __name__ == '__main__':
    main()
