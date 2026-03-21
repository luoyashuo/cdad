#!/usr/bin/env python3
"""
EvalScope V2 - 跨 Benchmark 流式推理

核心特性：
- 将所有 benchmark 的 prompts 混合进一个全局队列
- GPU 持续满载推理，benchmark 推理完成后立即在 CPU 线程触发评分
- 不阻塞其他 benchmark 的推理

用法：
    python eval_v2.py eval_configs/eval_all_sft_v2.yaml
    python eval_v2.py eval_configs/eval_all_sft_v2.yaml --resume
"""

import os
import sys
import json
import copy
import time
import logging
import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import yaml

# 添加 evalscope 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evalscope'))
from evalscope.config import TaskConfig
from evalscope.run import run_task
from evalscope.constants import EvalBackend, JudgeStrategy
from evalscope.utils.logger import get_logger
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.dataset import Dataset, Sample
from evalscope.api.registry import get_benchmark

from vllm import SamplingParams

# 从 eval.py 复用组件
from eval import (
    VLLMOfflineEngine,
    VLLMOfflineModelAPI,
    _extract_model_name,
    _resolve_lora,
    _extract_report_subsets,
)

logger = get_logger()


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class PromptItem:
    """单个 Prompt 项，包含推理所需的所有信息"""
    bench_name: str          # e.g. "AIME25"
    subset_name: str         # e.g. "default"
    sample_id: int           # evalscope sample.id (含 repeat 展开)
    group_id: int            # evalscope sample.group_id
    prompt_text: str         # apply_chat_template 后的最终 prompt
    sampling_params: SamplingParams  # 该 benchmark 的生成参数
    sample: Sample           # evalscope Sample 引用 (用于写 JSONL)


@dataclass
class BenchState:
    """单个 Benchmark 的状态"""
    bench_name: str
    dataset_name: str
    subsets: Dict[str, Dataset]   # subset_name -> Dataset
    benchmark: Any                # evalscope benchmark adapter
    task_config: TaskConfig       # evalscope TaskConfig
    repeats: int
    total_prompts: int
    completed: int = 0
    # subset -> {sample_id -> result}
    results: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    status: str = "pending"       # pending / inferring / scoring / done / failed
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 流式持久化：每个 subset 的 JSONL 文件句柄
    _jsonl_files: Dict[str, Any] = field(default_factory=dict)
    # 评分结果
    per_repeat_scores: List[float] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    error_msg: str = ""


# ============================================================
# 跨 Benchmark 评测器
# ============================================================

class CrossBenchmarkEvaluator:
    """跨 Benchmark 流式推理评测器
    
    工作流程：
    1. 加载所有 benchmark 的 dataset，提取 prompts
    2. 混合所有 prompts 到全局队列，用 VLLMOfflineEngine 推理
    3. 按 benchmark 收集结果，写入 evalscope 兼容的 predictions JSONL
    4. benchmark 推理完成后，异步调用 run_task(use_cache=True) 评分
    5. 汇总所有 benchmark 结果到统一表格
    """

    def __init__(self, config_file: str, resume: bool = False,
                 resume_dir: Optional[str] = None,
                 output_dir_override: Optional[str] = None,
                 force_rescore: bool = False):
        self.config_file = config_file
        self.config = self._load_config()
        self.model_path = self.config['model_path']
        self.model_name = _extract_model_name(self.model_path)
        
        # 输出目录
        if output_dir_override:
            self.output_dir = Path(output_dir_override)
            self.version = self.output_dir.name
        elif resume_dir:
            self.output_dir = Path(resume_dir)
            self.version = self.output_dir.name
        elif resume:
            # 自动检测最新 version_* 目录续跑
            evalscope_dir = Path(self.model_path) / "evalscope"
            latest = self._find_latest_version(evalscope_dir)
            if latest:
                self.output_dir = latest
                self.version = latest.name
                logger.info(f"自动检测到最新版本目录: {latest}")
            else:
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = evalscope_dir / self.version
                logger.info(f"未检测到已有版本，全新运行: {self.version}")
        else:
            self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
            self.output_dir = Path(self.model_path) / "evalscope" / self.version
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 时间戳（用于 predictions 目录）
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Benchmark 状态
        self.bench_states: Dict[str, BenchState] = {}
        
        # Resume 支持
        self.resume = resume or (resume_dir is not None)
        self.completed_benchmarks: Set[str] = set()
        
        # 是否强制重新评分（即使 repeat 数量相同）
        self.force_rescore = force_rescore
        
        # vLLM 引擎
        self.engine: Optional[VLLMOfflineEngine] = None
        
        # 评分线程池
        self._scoring_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scorer")
        
        # 进度显示
        self._progress: Optional['ProgressDisplay'] = None
        
        logger.info(f"CrossBenchmarkEvaluator 初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"版本: {self.version}")
        logger.info(f"强制重新评分: {force_rescore}")

    def _load_config(self) -> Dict:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _find_latest_version(evalscope_dir: Path) -> Optional[Path]:
        """返回 evalscope_dir 下按名称排序最新的 version_* 目录，不存在则返回 None"""
        if not evalscope_dir.exists():
            return None
        candidates = sorted(evalscope_dir.glob("version_*"), reverse=True)
        return candidates[0] if candidates else None

    def _load_existing_predictions(self, bench: BenchState) -> int:
        """加载已有 predictions，最大化利用已完成数据。

        策略：
        - 统计每个 subset 中连续完成的 repeat 轮数 complete_repeats
          （repeat r 完整 ⟺ 该 subset 所有题目都有 q*old_repeats+r 的条目）
        - effective_repeats = max(complete_repeats, new_repeats)
          * complete_repeats > new_repeats：直接用全部已完成数据评分，不浪费
          * complete_repeats < new_repeats：加载已有部分，剩余缺口补推理
        - 只加载 bench.subsets 中存在的 subset（过滤旧版本多余 subset）
        - 复用已有评分：仅当 subsets 完全一致且 effective_repeats == old_repeats 时跳过评分

        Returns: 加载的条目数（已填入 bench.results 和 bench.completed）
        """
        bench_dir = self.output_dir / bench.bench_name
        if not bench_dir.exists():
            return 0

        jsonl_pattern = f"*/predictions/{self.model_name}/{bench.dataset_name}_*.jsonl"
        jsonl_files = sorted(bench_dir.glob(jsonl_pattern))
        if not jsonl_files:
            return 0

        new_repeats = bench.repeats
        # 每个 subset 的原始题目数（非 repeat 展开），基于当前 dataset
        num_q_per_subset: Dict[str, int] = {
            s: len(ds) // new_repeats for s, ds in bench.subsets.items()
        }

        # 从 _result.json 获取 old_repeats 和旧 subsets 信息
        result_path = self.output_dir / f"{bench.bench_name}_result.json"
        old_repeats = None
        old_result_data = None
        if result_path.exists():
            try:
                with open(result_path) as f:
                    old_result_data = json.load(f)
                old_repeats = old_result_data.get('repeats')
            except Exception:
                pass

        # 加载所有 JSONL 条目（同一 index 以最后出现的为准）
        raw_entries: Dict[str, Dict[int, dict]] = {}
        for jsonl_path in jsonl_files:
            stem = jsonl_path.stem
            prefix = f"{bench.dataset_name}_"
            subset_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            if subset_name not in raw_entries:
                raw_entries[subset_name] = {}
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        raw_entries[subset_name][entry['index']] = entry
                    except Exception:
                        continue

        # 只保留 bench.subsets 中存在的 subset
        if bench.subsets:
            raw_entries = {k: v for k, v in raw_entries.items() if k in bench.subsets}

        total_raw = sum(len(v) for v in raw_entries.values())
        if total_raw == 0:
            return 0

        # 推断 old_repeats（若 result.json 未提供）
        if old_repeats is None:
            for subset_name, entries in raw_entries.items():
                nq = num_q_per_subset.get(subset_name, 1)
                if entries:
                    old_repeats = max(1, (max(entries.keys()) + 1 + nq - 1) // nq)
                    break
        if old_repeats is None:
            old_repeats = new_repeats

        # 统计每个 subset 中完整的 repeat 轮数
        # repeat r 完整 ⟺ 所有题目 q 都有 q*old_repeats+r 的条目
        def _count_complete(entries: Dict[int, dict], nq: int, nr: int) -> int:
            return sum(
                1 for r in range(nr)
                if all(q * nr + r in entries for q in range(nq))
            )

        complete_per_subset = []
        for subset_name in bench.subsets:
            entries = raw_entries.get(subset_name, {})
            nq = num_q_per_subset.get(subset_name, 1)
            cr = _count_complete(entries, nq, old_repeats) if entries else 0
            complete_per_subset.append(cr)

        complete_repeats = min(complete_per_subset) if complete_per_subset else 0
        # 最大化利用已完成数据：若 complete_repeats > new_repeats 则全部用上
        effective_repeats = max(complete_repeats, new_repeats)

        logger.info(f"[{bench.bench_name}] Resume: 发现 {total_raw} 条已有预测, "
                    f"old_repeats={old_repeats}, complete_repeats={complete_repeats}, "
                    f"new_repeats={new_repeats} → effective_repeats={effective_repeats}")

        # 构建 sample_id → Sample 映射（以 effective_repeats 为步长）
        # bench.subsets 中 sample.id 基于 new_repeats；超出 new_repeats 的 repeat 无 Sample 引用（仅缺 metadata，可接受）
        sample_map: Dict[int, Sample] = {}
        for subset_name, dataset in bench.subsets.items():
            for sample in dataset:
                q = sample.id // new_repeats
                r = sample.id % new_repeats
                sample_map[q * effective_repeats + r] = sample

        # 重映射 old sample_ids → new sample_ids（以 effective_repeats 为步长）
        loaded = 0
        for subset_name, entries in raw_entries.items():
            nq = num_q_per_subset.get(subset_name, 1)
            if subset_name not in bench.results:
                bench.results[subset_name] = {}

            for old_sid, entry in entries.items():
                question_id = old_sid // old_repeats
                old_repeat_idx = old_sid % old_repeats

                if old_repeat_idx >= effective_repeats:
                    continue
                if question_id >= nq:
                    continue

                new_sid = question_id * effective_repeats + old_repeat_idx

                choices = entry.get('model_output', {}).get('choices', [])
                if choices:
                    text = choices[0].get('message', {}).get('content', '')
                    finish_reason = choices[0].get('stop_reason', 'stop')
                else:
                    text = ''
                    finish_reason = 'unknown'

                bench.results[subset_name][new_sid] = {
                    'text': text,
                    'finish_reason': finish_reason,
                    'sample': sample_map.get(new_sid),
                }
                loaded += 1

        bench.completed = loaded

        # 若 effective_repeats 不同于 new_repeats，更新 bench 的 repeats 和 total_prompts
        if effective_repeats != new_repeats:
            bench.repeats = effective_repeats
            bench.total_prompts = sum(
                num_q_per_subset[s] * effective_repeats for s in bench.subsets
            )

        if loaded >= bench.total_prompts:
            # 预测充足，检查是否需要重新评分
            # 默认：如果 repeats 相同且未指定 force_rescore，则跳过评分
            should_rescore = self.force_rescore or (effective_repeats != old_repeats)
            
            if should_rescore:
                bench.status = "scoring"
                reason = "强制重新评分" if self.force_rescore else f"repeats 变化 ({old_repeats}→{effective_repeats})"
                logger.info(f"[{bench.bench_name}] Resume: 预测充足 ({loaded}/{bench.total_prompts})，"
                            f"将重新评分（原因: {reason}）")
            else:
                # 直接复用已有评分结果
                if old_result_data and 'per_repeat_scores' in old_result_data:
                    bench.per_repeat_scores = old_result_data['per_repeat_scores']
                    bench.stats = {
                        'mean': old_result_data.get('mean', 0),
                        'std': old_result_data.get('std', 0),
                        'min': old_result_data.get('min', 0),
                        'max': old_result_data.get('max', 0),
                    }
                    bench.status = "done"
                    self.completed_benchmarks.add(bench.bench_name)
                    logger.info(f"[{bench.bench_name}] Resume: 预测充足且 repeats 相同 ({effective_repeats})，"
                                f"跳过评分，直接复用已有结果 (mean={bench.stats['mean']:.4f})")
                else:
                    # 没有旧评分结果，需要重新评分
                    bench.status = "scoring"
                    logger.info(f"[{bench.bench_name}] Resume: 预测充足但无旧评分结果，将重新评分")
        else:
            logger.info(f"[{bench.bench_name}] Resume: 需补齐 "
                        f"{bench.total_prompts - loaded}/{bench.total_prompts} 条预测")

        return loaded

    def _save_progress(self):
        """保存进度文件"""
        progress = {
            'version': self.version,
            'config_file': self.config_file,
            'timestamp': self._timestamp,
            'completed_benchmarks': list(self.completed_benchmarks),
            'updated_at': datetime.now().isoformat(),
        }
        progress_path = self.output_dir / "progress_v2.json"
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _create_task_config(self, bench_name: str) -> TaskConfig:
        """创建 TaskConfig（复用 eval.py 的配置解析逻辑）"""
        bench_cfg = self.config['benchmarks'][bench_name]
        
        task_config = TaskConfig()
        task_config.model_id = self.model_name
        task_config.model = self.model_name  # 避免 "No model is provided" 警告
        task_config.eval_backend = EvalBackend.NATIVE
        
        # 支持指定数据源 (huggingface/modelscope)
        if 'data_source' in bench_cfg:
            from evalscope.constants import HubType
            hub_type = bench_cfg['data_source']
            if hub_type == 'huggingface':
                task_config.dataset_hub = HubType.HUGGINGFACE
            elif hub_type == 'modelscope':
                task_config.dataset_hub = HubType.MODELSCOPE
            logger.info(f"[{bench_name}] 使用数据源: {task_config.dataset_hub}")
        
        # 数据集
        task_config.datasets = [bench_cfg['dataset_name']]
        if 'limit' in bench_cfg:
            task_config.limit = bench_cfg['limit']
        task_config.repeats = bench_cfg.get('repeats', 1)
        
        dataset_args = {
            'dataset_name': bench_cfg.get('subset', None),
            'few_shot_num': bench_cfg.get('few_shot_num', 0)
        }
        # 若指定了 subset，通过 subset_list 限制 evalscope 只加载该 subset
        # （BenchmarkMeta._update 会 setattr subset_list，避免评分时把其他 subset 算作 0 分）
        if bench_cfg.get('subset'):
            dataset_args['subset_list'] = [bench_cfg['subset']]
        if 'extra_params' in bench_cfg:
            dataset_args['extra_params'] = bench_cfg['extra_params']
        # 支持指定数据源 (huggingface/modelscope)
        if 'data_source' in bench_cfg:
            dataset_args['data_source'] = bench_cfg['data_source']
        task_config.dataset_args = {bench_cfg['dataset_name']: dataset_args}
        
        # 生成参数
        generation_config_dict = {
            'temperature': bench_cfg.get('temperature', 0.0),
            'top_p': bench_cfg.get('top_p', 1.0),
            'top_k': bench_cfg.get('top_k', 50),
            'max_tokens': bench_cfg.get('max_tokens', 2048),
            'repetition_penalty': bench_cfg.get('repetition_penalty', 1.0)
        }
        gen_config = bench_cfg.get('gen_config', {})
        if gen_config:
            generation_config_dict.update(gen_config)
        task_config.generation_config = GenerateConfig.model_validate(generation_config_dict)
        
        # 工作目录
        task_config.work_dir = str(self.output_dir / bench_name)
        
        # Sandbox 配置
        if 'use_sandbox' in bench_cfg:
            task_config.use_sandbox = bench_cfg['use_sandbox']
            sandbox_workers = bench_cfg.get('sandbox_workers', 5)
            task_config.judge_worker_num = sandbox_workers
            
            sandbox_config = self.config.get('sandbox', {})
            remote_url = sandbox_config.get('remote_url')
            if remote_url:
                remote_config = {'base_url': remote_url}
                if sandbox_config.get('timeout'):
                    remote_config['timeout'] = sandbox_config['timeout']
                task_config.sandbox_manager_config = remote_config
            elif sandbox_config.get('work_dir'):
                task_config.sandbox_manager_config = {'work_dir': sandbox_config['work_dir']}
        
        # Judge 配置
        bench_judge_strategy = bench_cfg.get('judge_strategy', JudgeStrategy.RULE)
        task_config.judge_strategy = bench_judge_strategy
        use_llm = 'llm' in str(bench_judge_strategy).lower()
        
        if use_llm:
            api_key = self.config.get('API_KEY', '')
            base_url = self.config.get('BASE_URL', '').replace('/chat/completions', '').rstrip('/')
            judge_model = bench_cfg.get('judge_model', 'Qwen/Qwen3-8B')
            if not api_key or not base_url:
                raise ValueError(f"[{bench_name}] LLM judge 需要 API_KEY 和 BASE_URL")
            task_config.judge_model_args = {
                'api_key': api_key,
                'api_url': base_url,
                'model_id': judge_model,
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 1024,
                    'extra_body': {'enable_thinking': False}
                },
            }
            judge_parallel = bench_cfg.get('judge_parallel', 1)
            task_config.judge_worker_num = judge_parallel
        
        return task_config

    @staticmethod
    def _config_to_sampling_params(config: GenerateConfig) -> SamplingParams:
        """将 GenerateConfig 转换为 SamplingParams"""
        kwargs = {}
        if config.temperature is not None:
            kwargs['temperature'] = config.temperature
        if config.top_p is not None:
            kwargs['top_p'] = config.top_p
        if config.top_k is not None and config.top_k > 0:
            kwargs['top_k'] = config.top_k
        if config.max_tokens is not None:
            kwargs['max_tokens'] = config.max_tokens
        if getattr(config, 'repetition_penalty', None) is not None:
            kwargs['repetition_penalty'] = config.repetition_penalty
        return SamplingParams(**kwargs)

    def _load_all_benchmarks(self):
        """加载所有 benchmark 的 dataset"""
        logger.info("=" * 60)
        logger.info("加载所有 benchmark 数据...")
        
        for bench_name, bench_cfg in self.config['benchmarks'].items():
            logger.info(f"[{bench_name}] 加载中...")

            try:
                # 1. 构建 TaskConfig
                task_config = self._create_task_config(bench_name)

                # 2. 通过 evalscope API 加载 benchmark + dataset
                benchmark = get_benchmark(bench_cfg['dataset_name'], task_config)
                dataset_dict = {k: v for k, v in benchmark.load_dataset().items()}

                # 若配置指定了 subset，过滤只保留该 subset
                requested_subset = bench_cfg.get('subset')
                if requested_subset:
                    if requested_subset in dataset_dict:
                        dataset_dict = {requested_subset: dataset_dict[requested_subset]}
                    else:
                        logger.warning(f"[{bench_name}] 指定 subset '{requested_subset}' 不在数据集 "
                                       f"{list(dataset_dict.keys())} 中，使用全量数据")

                # 3. 记录 BenchState
                total = sum(len(ds) for ds in dataset_dict.values())
                bench_state = BenchState(
                    bench_name=bench_name,
                    dataset_name=bench_cfg['dataset_name'],
                    subsets=dataset_dict,
                    benchmark=benchmark,
                    task_config=task_config,
                    repeats=task_config.repeats,
                    total_prompts=total,
                )
                self.bench_states[bench_name] = bench_state

                logger.info(f"[{bench_name}] 加载完成: {total} prompts, {task_config.repeats} repeats")

                # 4. Resume：加载已有 predictions，自动处理 repeats 变更
                if self.resume:
                    loaded = self._load_existing_predictions(bench_state)
                    if bench_state.status == "done":
                        self.completed_benchmarks.add(bench_name)

            except Exception as e:
                logger.error(f"[{bench_name}] 加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.bench_states[bench_name] = BenchState(
                    bench_name=bench_name,
                    dataset_name=bench_cfg.get('dataset_name', 'unknown'),
                    subsets={},
                    benchmark=None,
                    task_config=None,
                    repeats=0,
                    total_prompts=0,
                    status="failed",
                    error_msg=str(e),
                )
        
        total_all = sum(b.total_prompts for b in self.bench_states.values() if b.status != "failed")
        logger.info(f"总计: {len(self.bench_states)} benchmarks, {total_all} prompts")
        logger.info("=" * 60)

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """将 ChatMessage 列表转换为 prompt 文本"""
        msg_dicts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else msg.text
            msg_dicts.append({"role": msg.role, "content": content})
        return self.engine.tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True)

    def _extract_all_prompts(self) -> List[PromptItem]:
        """从所有 BenchState 提取 PromptItem 列表。

        排列策略：
        1. 按 benchmark 分组（小 benchmark 优先），同一 benchmark 的 prompts 连续排列。
           这样小 benchmark 先完成推理，可以尽早开始评分（CPU 线程）。
        2. benchmark 内部按 repeat 轮次排序：先完成 repeat 0（所有题目），再 repeat 1，...
           这样中途中断时，已完成的数据是若干完整 repeat，便于 resume 和评分。
        3. 每个 repeat 轮内部 shuffle，配合 max_dedup_n 限制，
           确保 Worker 内有足够的 unique prompts 填满 GPU batch。
        """
        from collections import defaultdict

        # 按 benchmark 分组收集
        bench_items: List[Tuple[int, str, List[PromptItem]]] = []

        for bench in self.bench_states.values():
            # 跳过已完成（done）、已有足够预测待评分（scoring）、加载失败的
            if bench.status in ("failed", "done", "scoring"):
                continue

            sp = self._config_to_sampling_params(bench.task_config.generation_config)
            existing_results = bench.results  # resume 加载的已有结果
            repeats = bench.repeats

            # 按 repeat 轮次分组收集
            repeat_groups: Dict[int, List[PromptItem]] = defaultdict(list)
            skipped = 0
            for subset_name, dataset in bench.subsets.items():
                for sample in dataset:
                    # 跳过 resume 已加载的 sample
                    if existing_results.get(subset_name, {}).get(sample.id) is not None:
                        skipped += 1
                        continue
                    prompt_text = self._messages_to_prompt(sample.input)
                    item = PromptItem(
                        bench_name=bench.bench_name,
                        subset_name=subset_name,
                        sample_id=sample.id,
                        group_id=sample.group_id,
                        prompt_text=prompt_text,
                        sampling_params=sp,
                        sample=sample,
                    )
                    repeat_idx = sample.id % repeats
                    repeat_groups[repeat_idx].append(item)

            if skipped > 0:
                logger.info(f"  {bench.bench_name}: 跳过 {skipped} 已有, 需补齐 "
                            f"{sum(len(g) for g in repeat_groups.values())}")

            # 按 repeat 轮次排序：repeat 0 → repeat 1 → ...
            # 每个 repeat 轮内部 shuffle，打散局部性
            items = []
            for r in sorted(repeat_groups.keys()):
                group = repeat_groups[r]
                random.shuffle(group)
                items.extend(group)

            bench_items.append((len(items), bench.bench_name, items))

        # 按 benchmark 大小排序：小的先跑，先完成先评分
        bench_items.sort(key=lambda x: x[0])

        all_items = []
        for size, name, items in bench_items:
            all_items.extend(items)
            logger.info(f"  {name}: {size} prompts")

        logger.info(f"Prompt 提取完成: {len(all_items)} prompts, "
                    f"来自 {len(bench_items)} benchmarks, 小 benchmark 优先, repeat 轮次优先")
        return all_items

    def _run_global_inference(self, all_prompts: List[PromptItem]):
        """将所有 prompts 提交到 VLLMOfflineEngine，收集结果
        
        使用 ThreadPoolExecutor 并发提交所有 prompts，
        engine 的 _batch_loop 自动批处理 + 去重。
        """
        total = len(all_prompts)
        logger.info(f"开始全局推理: {total} prompts")

        # 计算 resume 已加载的基线数量（不计入速度估算）
        resumed_count = sum(
            b.completed for b in self.bench_states.values() if b.status != "failed"
        )

        # 初始化 StatusPrinter，接管 logger 的 StreamHandler
        self._progress = StatusPrinter(self.bench_states, resumed_done=resumed_count)
        _hijacked: List[logging.Handler] = []
        _status_handler: Optional[_StatusLogHandler] = None
        root_log = logging.getLogger()
        for h in list(root_log.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                orig_fmt = h.formatter
                root_log.removeHandler(h)
                _hijacked.append(h)
        if _hijacked:
            _status_handler = _StatusLogHandler(self._progress, _hijacked[0].formatter)
            _status_handler.setLevel(logging.DEBUG)
            root_log.addHandler(_status_handler)

        # 注册 TPS callback（引擎每次触发 TPS 打印时同步更新状态块）
        if self.engine:
            self.engine._tps_callback = self._progress.record_tps

        # 提交所有 prompts
        vllm_cfg = self.config.get('vllm', {})
        eval_batch_size = vllm_cfg.get('eval_batch_size', 512)
        
        with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
            futures = {}
            for item in all_prompts:
                fut = executor.submit(
                    self.engine.generate_single,
                    item.prompt_text,
                    item.sampling_params,
                    item.bench_name,  # group_key: 同 benchmark 优先一起分发
                )
                futures[fut] = item
            
            logger.info(f"已提交 {len(futures)} 个推理任务")
            
            # 收集结果
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    result = fut.result()
                    self._on_result(item, result)
                except Exception as e:
                    logger.error(f"推理失败 [{item.bench_name}/{item.sample_id}]: {e}")
                    # 标记失败但继续
                    self._on_result(item, {'text': '', 'finish_reason': 'error'})
        
        # 恢复 logger handlers，清理 TPS callback
        if self.engine:
            self.engine._tps_callback = None
        if _status_handler:
            root_log.removeHandler(_status_handler)
        for h in _hijacked:
            root_log.addHandler(h)
        if self._progress:
            self._progress.close()
            self._progress = None

        logger.info("全局推理完成")

    def _get_jsonl_writer(self, bench: BenchState, subset_name: str):
        """获取或创建 subset 对应的 JSONL 文件句柄（用于流式追加写入）"""
        if subset_name not in bench._jsonl_files:
            work_dir = self.output_dir / bench.bench_name / self._timestamp
            pred_dir = work_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"
            bench._jsonl_files[subset_name] = open(jsonl_path, 'a', encoding='utf-8')
        return bench._jsonl_files[subset_name]

    def _close_jsonl_writers(self, bench: BenchState):
        """关闭所有 JSONL 文件句柄"""
        for f in bench._jsonl_files.values():
            f.close()
        bench._jsonl_files.clear()

    def _on_result(self, item: PromptItem, result: dict):
        """收到推理结果，路由到对应 BenchState，并立即追加写入 JSONL"""
        bench = self.bench_states.get(item.bench_name)
        if bench is None:
            return

        # 调试：检查返回结果是否为空
        if not result.get('text'):
            logger.warning(f"[{item.bench_name}] sample_id={item.sample_id} 返回空结果! finish_reason={result.get('finish_reason')}")

        with bench.lock:
            if item.subset_name not in bench.results:
                bench.results[item.subset_name] = {}

            bench.results[item.subset_name][item.sample_id] = {
                'text': result['text'],
                'finish_reason': result['finish_reason'],
                'sample': item.sample,
            }
            bench.completed += 1

            # 流式持久化：立即追加写入 predictions JSONL（崩溃后可恢复）
            try:
                f = self._get_jsonl_writer(bench, item.subset_name)
                sample = item.sample
                model_result = {
                    "index": item.sample_id,
                    "model": self.model_name,
                    "model_output": {
                        "model": self.model_name,
                        "choices": [{
                            "message": {"role": "assistant", "content": result['text']},
                            "stop_reason": result['finish_reason'] or "stop",
                        }],
                    },
                    "messages": [{"role": "assistant", "content": result['text']}],
                    "metadata": getattr(sample, 'metadata', {}),
                }
                f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
                f.flush()
            except Exception as e:
                logger.warning(f"[{item.bench_name}] 流式写入失败: {e}")

            # 实时刷新进度显示
            if self._progress is not None:
                self._progress.refresh()

            # 检查是否完成
            if bench.completed >= bench.total_prompts:
                if bench.status != "scoring":
                    bench.status = "scoring"
                    # 关闭 JSONL 文件句柄
                    self._close_jsonl_writers(bench)
                    # 异步触发评分
                    self._scoring_pool.submit(self._score_benchmark, bench)

    def _write_predictions(self, bench: BenchState):
        """写入 evalscope 兼容的 predictions JSONL"""
        work_dir = self.output_dir / bench.bench_name / self._timestamp
        
        for subset_name, results in bench.results.items():
            pred_dir = work_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample_id, res in sorted(results.items()):
                    sample = res['sample']
                    # ModelOutput 格式：需要 choices 数组
                    model_result = {
                        "index": sample_id,
                        "model": self.model_name,
                        "model_output": {
                            "model": self.model_name,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": res['text'],
                                    },
                                    "stop_reason": res['finish_reason'] or "stop",
                                }
                            ],
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": res['text'],
                            }
                        ],
                        "metadata": getattr(sample, 'metadata', {}),
                    }
                    f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
        
        logger.info(f"[{bench.bench_name}] Predictions 已写入: {work_dir}")

    def _score_benchmark(self, bench: BenchState):
        """拆分 repeats → 每个 repeat 独立评分 → 汇总统计"""
        logger.info(f"[{bench.bench_name}] 开始评分...")
        
        work_dir = self.output_dir / bench.bench_name / self._timestamp
        scores = []
        
        try:
            for repeat_idx in range(bench.repeats):
                # 1. 拆分: 只取属于该 repeat 的 predictions
                repeat_work_dir = self._write_predictions_for_repeat(bench, repeat_idx)
                
                # 2. 调用 run_task(use_cache=True) 仅评分
                task_cfg = copy.deepcopy(bench.task_config)
                task_cfg.repeats = 1  # 每次只评 1 repeat
                task_cfg.use_cache = str(repeat_work_dir)
                task_cfg.work_dir = str(repeat_work_dir)  # 确保评分结果输出到 repeat 目录
                
                # 使用 model_id 而不是 model（避免加载模型）
                task_cfg.model = self.model_name
                
                logger.info(f"[{bench.bench_name}] 评分 repeat {repeat_idx + 1}/{bench.repeats}...")
                result = run_task(task_cfg)
                
                # 调试：检查 result 类型
                logger.info(f"  result type: {type(result)}")
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                    logger.info(f"  converted to dict")
                elif hasattr(result, 'model_dump'):
                    result = result.model_dump()
                    logger.info(f"  model_dump to dict")
                
                # 3. 提取分数
                score = self._extract_score(result, bench)
                scores.append(score)
                logger.info(f"[{bench.bench_name}] repeat {repeat_idx + 1}: score={score:.4f}")
            
            # 4. 汇总统计
            bench.per_repeat_scores = scores
            bench.stats = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
            bench.status = "done"
            
            # 保存结果
            self._save_benchmark_result(bench)
            self.completed_benchmarks.add(bench.bench_name)
            self._save_progress()
            
            logger.info(f"[{bench.bench_name}] 评分完成: mean={bench.stats['mean']:.4f}, std={bench.stats['std']:.4f}")
            
        except Exception as e:
            logger.error(f"[{bench.bench_name}] 评分失败: {e}")
            bench.status = "failed"
            bench.error_msg = str(e)

    def _write_predictions_for_repeat(self, bench: BenchState, repeat_idx: int) -> Path:
        """为指定 repeat 写入 predictions JSONL
        
        evalscope repeat 展开后，sample_id = original_id * repeats + repeat_idx
        每个 repeat 包含所有原始题目的一次回答。
        """
        work_dir = self.output_dir / bench.bench_name / self._timestamp
        repeat_dir = work_dir / f"repeat_{repeat_idx}"
        
        total_written = 0
        empty_count = 0
        
        for subset_name, results in bench.results.items():
            pred_dir = repeat_dir / "predictions" / self.model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = pred_dir / f"{bench.dataset_name}_{subset_name}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample_id, res in sorted(results.items()):
                    # 只取属于该 repeat 的 samples
                    if sample_id % bench.repeats != repeat_idx:
                        continue
                    
                    # 重映射 index 为无 repeat 的连续 id
                    new_index = sample_id // bench.repeats
                    sample = res.get('sample')
                    
                    # 统计空结果
                    if not res['text']:
                        empty_count += 1
                    
                    # ModelOutput 格式：需要 choices 数组
                    model_result = {
                        "index": new_index,
                        "model": self.model_name,
                        "model_output": {
                            "model": self.model_name,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": res['text'],
                                    },
                                    "stop_reason": res['finish_reason'] or "stop",
                                }
                            ],
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": res['text'],
                            }
                        ],
                        "metadata": getattr(sample, 'metadata', {}) if sample else {},
                    }
                    f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
                    total_written += 1

        logger.info(f"[{bench.bench_name}] repeat_{repeat_idx}: 写入 {total_written} 条 predictions, 其中空结果 {empty_count} 条")
        return repeat_dir

    def _extract_score(self, result, bench: Optional[BenchState] = None) -> float:
        """从 evalscope 结果中提取分数

        result 可能是:
        - dict: {dataset_name: report_dict, ...}
        - Report 对象
        - List[Report]

        当 bench 指定了 subset 时，只取该 subset 的分数（避免被其他 subset 的 0 分拉低均值）
        """
        # 获取当前 benchmark 实际使用的 subset 集合
        valid_subsets = set(bench.subsets.keys()) if bench and bench.subsets else None

        # 1. 处理 Report 对象
        if hasattr(result, 'score'):
            score = result.score
            logger.info(f"  Report.score = {score}")
            return float(score) if score is not None else 0.0

        # 2. 处理 List[Report]
        if isinstance(result, list):
            scores = []
            for r in result:
                if hasattr(r, 'score') and r.score is not None:
                    # 如果 Report 有 name/dataset_name 属性，按 valid_subsets 过滤
                    r_name = getattr(r, 'name', None) or getattr(r, 'dataset_name', None)
                    if valid_subsets and r_name and r_name not in valid_subsets:
                        continue
                    scores.append(float(r.score))
            final_score = np.mean(scores) if scores else 0.0
            logger.info(f"  List[Report] scores = {scores}, final = {final_score}")
            return final_score

        # 3. 处理 dict
        if not isinstance(result, dict):
            logger.warning(f"_extract_score: 未知类型 {type(result)}")
            return 0.0

        # run_task 返回的是 {dataset_name: report_dict} 格式
        # 首先检查是否是单 dataset 结果（最常见情况）
        if len(result) == 1:
            dataset_result = list(result.values())[0]
            if isinstance(dataset_result, dict):
                score = dataset_result.get('score', dataset_result.get('accuracy', 0))
                if score is not None:
                    final_score = float(score)
                    logger.info(f"  Single dataset result: score = {final_score}")
                    return final_score
            elif hasattr(dataset_result, 'score') and dataset_result.score is not None:
                final_score = float(dataset_result.score)
                logger.info(f"  Single dataset Report result: score = {final_score}")
                return final_score

        # 遍历 dataset 结果（多 dataset 情况）
        # 注意：dict 的 key 是 dataset_name（如 'aime25'），不是 subset_name
        # 所以不应该用 valid_subsets 去过滤，valid_subsets 是子集名称（如 'AIME2025-I'）
        scores = []
        for dataset_name, dataset_result in result.items():
            if isinstance(dataset_result, dict):
                score = dataset_result.get('score', dataset_result.get('accuracy', 0))
                if score is not None:
                    scores.append(float(score))
            elif hasattr(dataset_result, 'score'):
                if dataset_result.score is not None:
                    scores.append(float(dataset_result.score))

        final_score = np.mean(scores) if scores else 0.0
        logger.info(f"  Dict result: scores = {scores}, final = {final_score}")
        return final_score

    def _save_benchmark_result(self, bench: BenchState):
        """保存 benchmark 结果到 JSON"""
        result = {
            'benchmark': bench.bench_name,
            'dataset': bench.dataset_name,
            'subsets': list(bench.subsets.keys()),
            'repeats': bench.repeats,
            'per_repeat_scores': bench.per_repeat_scores,
            'mean': bench.stats.get('mean', 0),
            'std': bench.stats.get('std', 0),
            'min': bench.stats.get('min', 0),
            'max': bench.stats.get('max', 0),
        }
        
        result_path = self.output_dir / f"{bench.bench_name}_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[{bench.bench_name}] 结果已保存: {result_path}")

    def _compile_summary(self):
        """汇总所有 benchmark 结果到统一表格"""
        rows = []
        
        for bench in self.bench_states.values():
            if bench.status != "done":
                continue
            
            row = {
                'benchmark': bench.bench_name,
                'dataset': bench.dataset_name,
                'repeats': bench.repeats,
                'mean': bench.stats.get('mean', 0),
                'std': bench.stats.get('std', 0),
                'min': bench.stats.get('min', 0),
                'max': bench.stats.get('max', 0),
            }
            
            # 添加每个 repeat 的分数
            for i, score in enumerate(bench.per_repeat_scores):
                row[f'run_{i + 1}'] = round(score, 4)
            
            rows.append(row)
        
        if not rows:
            logger.warning("没有完成的 benchmark 可以汇总")
            return
        
        df = pd.DataFrame(rows)
        
        # 保存 Excel
        excel_path = self.output_dir / "unified_summary.xlsx"
        df.to_excel(excel_path, index=False)
        logger.info(f"汇总结果已保存: {excel_path}")
        
        # 保存 JSON
        json_path = self.output_dir / "unified_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info(f"汇总结果已保存: {json_path}")
        
        # 打印摘要
        logger.info("\n" + "=" * 80)
        logger.info("评测结果汇总:")
        logger.info("-" * 80)
        for row in rows:
            logger.info(f"{row['benchmark']:20s} | mean={row['mean']:.4f}, std={row['std']:.4f}, "
                       f"range=[{row['min']:.4f}, {row['max']:.4f}]")
        logger.info("=" * 80)

    def run(self):
        """主流程"""
        logger.info("\n" + "=" * 80)
        logger.info("CrossBenchmarkEvaluator V2 开始运行")
        logger.info(f"模型: {self.model_name}")
        logger.info(f"配置文件: {self.config_file}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 80 + "\n")
        
        # 1. 加载所有 benchmark
        self._load_all_benchmarks()
        
        # 过滤掉失败的和已完成的
        active_benches = {k: v for k, v in self.bench_states.items()
                         if v.status not in ("failed", "done")}
        # 保留 done 的用于汇总
        done_benches = {k: v for k, v in self.bench_states.items() if v.status == "done"}

        if not active_benches and not done_benches:
            logger.error("没有有效的 benchmark 可以评估")
            return

        # 先同步完成需要重新评分的 benchmark（避免与推理引擎 / StatusPrinter 并发导致 hang）
        scoring_benches = {k: v for k, v in active_benches.items() if v.status == "scoring"}
        if scoring_benches:
            logger.info(f"先完成 {len(scoring_benches)} 个 benchmark 的评分（同步）...")
            scoring_futures = []
            for bench in scoring_benches.values():
                logger.info(f"[{bench.bench_name}] 触发重新评分...")
                scoring_futures.append(
                    self._scoring_pool.submit(self._score_benchmark, bench))
            # 等待所有评分完成，再启动推理引擎
            for fut in scoring_futures:
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"评分异常: {e}")
            logger.info("所有 pending 评分已完成")

        # 需要推理的 benchmark
        need_inference = {k: v for k, v in active_benches.items()
                         if v.status == "pending"}

        # 合并所有参与本轮的 bench（需要推理 + 正在评分 + 已完成）
        self.bench_states = {**done_benches, **active_benches}

        try:
            if need_inference:
                # 2. 初始化 vLLM 引擎（仅在有推理需求时）
                vllm_cfg = self.config.get('vllm', {})
                dp = vllm_cfg.get('dp', 1)
                chunk_size = vllm_cfg.get('worker_chunk_size', 0)
                # worker_chunk_size=0 表示全量分发，但会导致进度长时间无更新
                # dp>1 时自动设置合理的 chunk_size，确保 Worker 更频繁地返回结果
                if chunk_size == 0 and dp > 1:
                    chunk_size = 64
                    logger.info(f"worker_chunk_size 未设置，dp={dp}，自动调整为 {chunk_size} 以提高进度更新频率")
                self.engine = VLLMOfflineEngine(
                    model_path=self.model_path,
                    tp=vllm_cfg.get('tp', 1),
                    dp=dp,
                    gpu_memory_utilization=vllm_cfg.get('gpu_memory_utilization', 0.9),
                    max_model_len=vllm_cfg.get('max_model_len', None),
                    enforce_eager=vllm_cfg.get('enforce_eager', True),
                    worker_chunk_size=chunk_size,
                    max_dedup_n=vllm_cfg.get('max_dedup_n', 8),
                )

                tps_interval = vllm_cfg.get('tps_interval', 30)
                self.engine._tps_interval = tps_interval
                self.engine._tps_reset()
                logger.info(f"TPS 统计已启动，每 {tps_interval} 秒打印一次")

                # 3. 提取需要推理的 prompts（跳过 resume 已加载的）
                all_prompts = self._extract_all_prompts()
                logger.info(f"共提取 {len(all_prompts)} 个 prompts 需要推理")

                if all_prompts:
                    # 4. 全局推理
                    self._run_global_inference(all_prompts)
                else:
                    logger.info("所有 prompts 均已有预测，跳过推理")
            else:
                logger.info("所有 benchmark 均有足够预测，跳过推理引擎初始化")

            # 5. 等待所有评分完成
            pending_scoring = {k for k, v in self.bench_states.items()
                              if v.status not in ("done", "failed")}
            if pending_scoring:
                logger.info(f"等待 {len(pending_scoring)} 个 benchmark 评分完成...")
                self._wait_for_scoring()

            # 6. 汇总结果
            self._compile_summary()

            # 7. 保存配置
            self._save_config()

        finally:
            if self.engine:
                self.engine.shutdown()
                self.engine = None
            self._scoring_pool.shutdown(wait=True)
        
        logger.info("\nCrossBenchmarkEvaluator V2 运行完成!")

    def _wait_for_scoring(self):
        """等待所有 benchmark 的评分完成"""
        pending = set(self.bench_states.keys())
        last_log = time.time()
        while pending:
            done = set()
            for name in list(pending):
                if self.bench_states[name].status in ("done", "failed"):
                    done.add(name)
            for name in done:
                pending.remove(name)
                bench = self.bench_states[name]
                logger.info(f"[{name}] 评分{'完成' if bench.status == 'done' else '失败'}")
            if pending and time.time() - last_log >= 30:
                names = ', '.join(pending)
                logger.info(f"等待评分完成: {names}")
                last_log = time.time()
            if pending:
                time.sleep(1)

    def _save_config(self):
        """保存配置文件副本"""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"配置已保存: {config_path}")


# ============================================================
# 进度显示
# ============================================================

class StatusPrinter:
    """
    通过 ANSI 转义码在 tail -f 时实现"状态块固定在底部"效果。

    文件末尾始终保持固定高度的状态块。每当有日志消息或状态更新时：
      1. 上移 N 行 + 清除到屏幕末尾（擦除旧状态块）
      2. 写入新内容（日志行 / 空）
      3. 重绘状态块

    状态块布局（共 _block_height 行）：
      ──────────────────────────────────────────────────────────
       总进度 [████████░░]  2304/ 8640  26.7%  bench 2/9  ⏱ 00:12:34
      ──────────────────────────────────────────────────────────
       AIME25          [████████████████████]  1280/ 1280 100.0% ✓
       AIME24          [████████████████████]   960/  960 100.0% ✓
       AMC             [██████░░░░░░░░░░░░░░]   368/  736  50.0% →
       ...
      ──────────────────────────────────────────────────────────
       TPS:  1234.5 tok/s (10:41:05)  →   987.3 tok/s (10:41:35)  →  1102.8 tok/s (10:42:05)
      ──────────────────────────────────────────────────────────
    """

    BAR_W        = 20    # 进度条宽度（字符数）
    LINE_W       = 80    # 行宽上限，避免终端折行导致光标偏移
    MAX_TPS_HIST = 3     # 保留最近 N 次 TPS 记录
    REDRAW_SEC   = 5     # 后台定期刷新间隔（秒）

    EMA_ALPHA    = 0.1   # EMA 平滑系数（越小越平滑）

    def __init__(self, bench_states: Dict[str, 'BenchState'], resumed_done: int = 0):
        self._bench_states  = bench_states
        self._lock          = threading.Lock()
        self._tps_history: List[Tuple[float, str]] = []  # (ema_tps, time_str)
        self._ema_tps: Optional[float] = None            # TPS 的 EMA 平滑值
        self._start_time    = time.time()
        self._initialized   = False

        # Resume 基线：推理开始前已完成的数量，不计入速度估算
        self._resumed_done  = resumed_done
        self._last_done     = resumed_done   # 上次采样时的 done 计数
        self._last_sample_t = self._start_time
        self._ema_speed: Optional[float] = None  # EMA 平滑后的推理速度 (samples/sec)

        # 固定行数：分隔线×4 + 总进度×1 + bench×N + TPS×1
        self._block_height  = 4 + 1 + len(bench_states) + 1

        # 启动后台定期刷新线程
        self._stop = threading.Event()
        self._bg   = threading.Thread(target=self._bg_loop, daemon=True, name="StatusPrinter")
        self._bg.start()

    # ── 外部接口 ─────────────────────────────────────────────

    def write_log(self, msg: str):
        """日志消息路由入口，由 _StatusLogHandler 调用。"""
        with self._lock:
            self._render(msg)

    def refresh(self):
        """由 _on_result 调用，限流重绘（最多每 0.5s 一次，避免频繁刷屏）。"""
        now = time.time()
        if now - getattr(self, '_last_refresh', 0) < 0.5:
            return
        self._last_refresh = now
        with self._lock:
            self._render(None)

    def record_tps(self, window_tps: float, _global_tps: float):
        """记录一次 TPS 读数并触发重绘。由 VLLMOfflineEngine._tps_callback 调用。"""
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            # 对 window_tps 做 EMA 平滑，避免大批次完成时出现毛刺
            if self._ema_tps is None:
                self._ema_tps = window_tps
            else:
                self._ema_tps = self.EMA_ALPHA * window_tps + (1 - self.EMA_ALPHA) * self._ema_tps
            self._tps_history.append((self._ema_tps, ts))
            if len(self._tps_history) > self.MAX_TPS_HIST:
                self._tps_history.pop(0)
            self._render(None)

    def close(self):
        """停止后台线程；最后做一次重绘，让状态块留在最终位置。"""
        self._stop.set()
        with self._lock:
            self._render(None)

    # ── 内部实现 ─────────────────────────────────────────────

    def _render(self, log_line: Optional[str]):
        """擦除旧状态块 → 写日志行（若有）→ 重绘状态块。必须在 _lock 内调用。"""
        out = sys.stderr
        if self._initialized:
            out.write(f'\033[{self._block_height}A\033[J')
        if log_line is not None:
            out.write(log_line + '\n')
        for line in self._build_block():
            out.write(line + '\n')
        out.flush()
        self._initialized = True

    def _bg_loop(self):
        while not self._stop.wait(self.REDRAW_SEC):
            with self._lock:
                self._render(None)

    def _build_block(self) -> List[str]:
        """构建当前状态块，返回恰好 _block_height 行的列表。"""
        W   = self.LINE_W
        sep = '─' * W

        # ── 总进度 ──────────────────────────────────────────
        total = sum(b.total_prompts for b in self._bench_states.values()
                    if b.status != "failed")
        done  = sum(b.completed     for b in self._bench_states.values()
                    if b.status != "failed")
        pct   = done / total * 100 if total > 0 else 0.0
        bw    = self.BAR_W
        bar   = '█' * int(bw * done / total) + '░' * (bw - int(bw * done / total)) \
                if total > 0 else '░' * bw
        elap  = time.time() - self._start_time
        etime = f"{int(elap//3600):02d}:{int(elap%3600//60):02d}:{int(elap%60):02d}"
        
        # 计算 ETA（基于 EMA 平滑速度，排除 resume 数据的影响）
        now = time.time()
        new_done = done - self._resumed_done  # 本轮推理实际完成的数量
        eta_str = "计算中..."
        if new_done > 0 and done < total:
            # 更新 EMA 速度：用增量 / 时间差计算瞬时速度
            dt = now - self._last_sample_t
            d_done = done - self._last_done
            if dt > 0 and d_done > 0:
                instant_speed = d_done / dt
                if self._ema_speed is None:
                    self._ema_speed = instant_speed
                else:
                    self._ema_speed = self.EMA_ALPHA * instant_speed + (1 - self.EMA_ALPHA) * self._ema_speed
                self._last_done = done
                self._last_sample_t = now

            speed = self._ema_speed
            if speed is None:
                # 还没采样到增量，用全局平均作为 fallback
                speed = new_done / elap if elap > 0 else 0

            remain = total - done
            eta_sec = remain / speed if speed > 0 else 0
            eta_h = int(eta_sec // 3600)
            eta_m = int((eta_sec % 3600) // 60)
            eta_s = int(eta_sec % 60)
            eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
        elif done >= total:
            eta_str = "已完成"
        
        n_done   = sum(1 for b in self._bench_states.values() if b.status == "done")
        n_total  = len(self._bench_states)
        total_line = (
            f" 总进度 [{bar}] {done:6d}/{total:<6d} {pct:5.1f}%"
            f"  bench {n_done}/{n_total}  ⏱ {etime}  ETA: {eta_str}"
        )

        # ── per-bench ────────────────────────────────────────
        bench_lines = []
        for name, bench in self._bench_states.items():
            bd = bench.completed
            bt = max(bench.total_prompts, 1)
            pb = bd / bt
            bbar = '█' * int(bw * pb) + '░' * (bw - int(bw * pb))
            st = bench.status
            if   st == "done":    icon, c = "✓", "\033[32m"
            elif st == "scoring": icon, c = "⏳", "\033[33m"
            elif st == "failed":  icon, c = "✗", "\033[31m"
            elif st == "pending": icon, c = " ", "\033[90m"
            else:                 icon, c = "→", ""
            rst  = "\033[0m"
            nm   = name[:16].ljust(16)
            bench_lines.append(
                f" {c}{nm}{rst} [{bbar}] {bd:5d}/{bt:5d} {pb*100:5.1f}% {icon}"
            )

        # ── TPS ──────────────────────────────────────────────
        if self._tps_history:
            parts = [f"\033[36m{wt:>8.1f}\033[0m tok/s ({ts})"
                     for wt, ts in self._tps_history]
            tps_line = " TPS: " + "  →  ".join(parts)
        else:
            tps_line = " TPS: 等待数据..."

        return [sep, total_line, sep] + bench_lines + [sep, tps_line, sep]


class _StatusLogHandler(logging.Handler):
    """将 logger 所有 StreamHandler 的输出路由到 StatusPrinter，保证状态块始终在底部。"""

    def __init__(self, printer: StatusPrinter, orig_formatter: Optional[logging.Formatter] = None):
        super().__init__()
        self._printer   = printer
        self._formatter = orig_formatter or logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    def emit(self, record: logging.LogRecord):
        try:
            msg = self._formatter.format(record)
            self._printer.write_log(msg)
        except Exception:
            self.handleError(record)


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EvalScope V2 - 跨 Benchmark 流式推理",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('config', help='YAML 配置文件路径')
    parser.add_argument('--resume', action='store_true',
                        help='断点续传：自动检测已完成的 benchmark 并跳过')
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='指定续传的输出目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='强制指定输出目录（覆盖自动生成的目录）')
    parser.add_argument('--force-rescore', action='store_true',
                        help='强制重新评分：即使 repeats 数量与上次相同也重新评分（默认会跳过）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建 evaluator 并运行
    evaluator = CrossBenchmarkEvaluator(
        config_file=args.config,
        resume=args.resume,
        resume_dir=args.resume_dir,
        output_dir_override=args.output_dir,
        force_rescore=args.force_rescore
    )
    
    evaluator.run()


if __name__ == '__main__':
    main()
