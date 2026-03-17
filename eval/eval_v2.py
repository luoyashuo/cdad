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
                 output_dir_override: Optional[str] = None):
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
        else:
            self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
            self.output_dir = Path(self.model_path) / "evalscope" / self.version
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 时间戳（用于 predictions 目录）
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Benchmark 状态
        self.bench_states: Dict[str, BenchState] = {}
        
        # Resume 支持
        self.resume = resume
        self.completed_benchmarks: Set[str] = set()
        if resume or resume_dir:
            self._scan_completed_benchmarks()
        
        # vLLM 引擎
        self.engine: Optional[VLLMOfflineEngine] = None
        
        # 评分线程池
        self._scoring_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scorer")
        
        # 进度显示
        self._progress: Optional['ProgressDisplay'] = None
        
        logger.info(f"CrossBenchmarkEvaluator 初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"版本: {self.version}")

    def _load_config(self) -> Dict:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _scan_completed_benchmarks(self):
        """扫描已完成的 benchmark"""
        for p in self.output_dir.glob('*_result.json'):
            bench_name = p.stem[:-len('_result')]
            self.completed_benchmarks.add(bench_name)
        logger.info(f"[Resume] 发现 {len(self.completed_benchmarks)} 个已完成 benchmark: {self.completed_benchmarks}")

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
            # 跳过已完成的
            if bench_name in self.completed_benchmarks:
                logger.info(f"[{bench_name}] 已跳过（已完成）")
                continue
            
            logger.info(f"[{bench_name}] 加载中...")
            
            try:
                # 1. 构建 TaskConfig
                task_config = self._create_task_config(bench_name)
                
                # 2. 通过 evalscope API 加载 benchmark + dataset
                benchmark = get_benchmark(bench_cfg['dataset_name'], task_config)
                dataset_dict = benchmark.load_dataset()
                
                # 3. 记录 BenchState
                total = sum(len(ds) for ds in dataset_dict.values())
                self.bench_states[bench_name] = BenchState(
                    bench_name=bench_name,
                    dataset_name=bench_cfg['dataset_name'],
                    subsets=dataset_dict,
                    benchmark=benchmark,
                    task_config=task_config,
                    repeats=task_config.repeats,
                    total_prompts=total,
                )
                
                logger.info(f"[{bench_name}] 加载完成: {total} prompts, {task_config.repeats} repeats")
                
            except Exception as e:
                logger.error(f"[{bench_name}] 加载失败: {e}")
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
        """从所有 BenchState 提取 PromptItem 列表"""
        items = []
        
        for bench in self.bench_states.values():
            if bench.status == "failed":
                continue
            
            sp = self._config_to_sampling_params(bench.task_config.generation_config)
            
            for subset_name, dataset in bench.subsets.items():
                for sample in dataset:
                    # sample.input 是 List[ChatMessage]，需要 apply_chat_template
                    prompt_text = self._messages_to_prompt(sample.input)
                    items.append(PromptItem(
                        bench_name=bench.bench_name,
                        subset_name=subset_name,
                        sample_id=sample.id,
                        group_id=sample.group_id,
                        prompt_text=prompt_text,
                        sampling_params=sp,
                        sample=sample,
                    ))
        
        return items

    def _run_global_inference(self, all_prompts: List[PromptItem]):
        """将所有 prompts 提交到 VLLMOfflineEngine，收集结果
        
        使用 ThreadPoolExecutor 并发提交所有 prompts，
        engine 的 _batch_loop 自动批处理 + 去重。
        """
        total = len(all_prompts)
        logger.info(f"开始全局推理: {total} prompts")
        
        # 初始化进度显示
        self._progress = ProgressDisplay(self.bench_states, total)
        
        # 提交所有 prompts
        vllm_cfg = self.config.get('vllm', {})
        eval_batch_size = vllm_cfg.get('eval_batch_size', 512)
        
        with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
            futures = {}
            for item in all_prompts:
                fut = executor.submit(
                    self.engine.generate_single,
                    item.prompt_text,
                    item.sampling_params
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
        
        logger.info("全局推理完成")

    def _on_result(self, item: PromptItem, result: dict):
        """收到推理结果，路由到对应 BenchState"""
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
            
            # 更新进度
            if self._progress:
                self._progress.update(item.bench_name)
            
            # 检查是否完成
            if bench.completed >= bench.total_prompts:
                if bench.status != "scoring":
                    bench.status = "scoring"
                    # 写入 predictions JSONL
                    self._write_predictions(bench)
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
                score = self._extract_score(result)
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
                    sample = res['sample']
                    
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
                        "metadata": getattr(sample, 'metadata', {}),
                    }
                    f.write(json.dumps(model_result, ensure_ascii=False) + '\n')
                    total_written += 1
        
        logger.info(f"[{bench.bench_name}] repeat_{repeat_idx}: 写入 {total_written} 条 predictions, 其中空结果 {empty_count} 条")
        return repeat_dir

    def _extract_score(self, result) -> float:
        """从 evalscope 结果中提取分数
        
        result 可能是:
        - dict: {dataset_name: report_dict, ...}
        - Report 对象
        - List[Report]
        """
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
                    scores.append(float(r.score))
            final_score = np.mean(scores) if scores else 0.0
            logger.info(f"  List[Report] scores = {scores}, final = {final_score}")
            return final_score
        
        # 3. 处理 dict
        if not isinstance(result, dict):
            logger.warning(f"_extract_score: 未知类型 {type(result)}")
            return 0.0
        
        # 遍历所有 dataset 结果
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
        
        # 过滤掉失败的
        active_benches = {k: v for k, v in self.bench_states.items() if v.status != "failed"}
        if not active_benches:
            logger.error("没有有效的 benchmark 可以评估")
            return
        
        self.bench_states = active_benches
        
        # 2. 初始化 vLLM 引擎
        vllm_cfg = self.config.get('vllm', {})
        self.engine = VLLMOfflineEngine(
            model_path=self.model_path,
            tp=vllm_cfg.get('tp', 1),
            dp=vllm_cfg.get('dp', 1),
            gpu_memory_utilization=vllm_cfg.get('gpu_memory_utilization', 0.9),
            max_model_len=vllm_cfg.get('max_model_len', None),
            enforce_eager=vllm_cfg.get('enforce_eager', True),
            worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
        )
        
        # 启动 TPS 统计（可选：通过 tps_interval 配置打印间隔，默认 30 秒）
        tps_interval = vllm_cfg.get('tps_interval', 30)
        self.engine._tps_interval = tps_interval
        self.engine._tps_reset()
        logger.info(f"TPS 统计已启动，每 {tps_interval} 秒打印一次")
        
        try:
            # 3. 提取所有 prompts
            all_prompts = self._extract_all_prompts()
            logger.info(f"共提取 {len(all_prompts)} 个 prompts")
            
            # 4. 全局推理
            self._run_global_inference(all_prompts)
            
            # 5. 等待所有评分完成
            logger.info("等待所有评分完成...")
            self._wait_for_scoring()
            
            # 6. 汇总结果
            self._compile_summary()
            
            # 7. 保存配置
            self._save_config()
            
        finally:
            # 关闭引擎
            if self.engine:
                self.engine.shutdown()
                self.engine = None
            
            # 关闭评分线程池
            self._scoring_pool.shutdown(wait=True)
        
        logger.info("\nCrossBenchmarkEvaluator V2 运行完成!")

    def _wait_for_scoring(self):
        """等待所有 benchmark 的评分完成"""
        pending = set(self.bench_states.keys())
        
        with tqdm(total=len(pending), desc="等待评分完成") as pbar:
            while pending:
                done = set()
                for name in pending:
                    bench = self.bench_states[name]
                    if bench.status in ("done", "failed"):
                        done.add(name)
                
                for name in done:
                    pending.remove(name)
                    pbar.update(1)
                
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

class ProgressDisplay:
    """多 benchmark 实时进度条"""
    
    def __init__(self, bench_states: Dict[str, BenchState], total_prompts: int):
        self.total_prompts = total_prompts
        self.completed_prompts = 0
        
        # 总进度条
        self.total_bar = tqdm(
            total=total_prompts,
            desc="总进度",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # 每个 benchmark 的进度条
        self.bench_bars = {}
        active_benches = {k: v for k, v in bench_states.items() if v.status != "failed"}
        
        for i, (name, bench) in enumerate(active_benches.items(), 1):
            self.bench_bars[name] = tqdm(
                total=bench.total_prompts,
                desc=f"{name:15s}",
                position=i,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            )
    
    def update(self, bench_name: str, delta: int = 1):
        """更新进度"""
        self.completed_prompts += delta
        self.total_bar.update(delta)
        
        if bench_name in self.bench_bars:
            self.bench_bars[bench_name].update(delta)
    
    def close(self):
        """关闭所有进度条"""
        self.total_bar.close()
        for bar in self.bench_bars.values():
            bar.close()


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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建 evaluator 并运行
    evaluator = CrossBenchmarkEvaluator(
        config_file=args.config,
        resume=args.resume,
        resume_dir=args.resume_dir,
        output_dir_override=args.output_dir
    )
    
    evaluator.run()


if __name__ == '__main__':
    main()
