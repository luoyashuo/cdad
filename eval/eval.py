#!/usr/bin/env python3
"""
EvalScope测评脚本 - 离线vLLM推理版本
使用vLLM离线模式直接推理，支持TP/DP配置，无需启动HTTP服务。
- dp=1: 单进程，无僵尸进程风险
- dp>1: 通过Ray实现数据并行，自动打满GPU
"""

import os
import sys
import gc
import json
import time
import logging
import threading
import concurrent.futures
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union
from tqdm import tqdm
import yaml

# 添加evalscope路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evalscope'))
from evalscope.config import TaskConfig
from evalscope.run import run_task
from evalscope.constants import EvalBackend, JudgeStrategy
from evalscope.utils.logger import get_logger
from evalscope.api.model.generate_config import GenerateConfig
from evalscope.api.model.model import ModelAPI, Model
from evalscope.api.model.model_output import ModelOutput, as_stop_reason
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo

from vllm import LLM, SamplingParams

logger = get_logger()


# ============================================================
# 辅助函数（不变）
# ============================================================

def _extract_report_subsets(dataset_result: dict) -> list:
    """从序列化后的 Report 字典中提取 per-subset 分数列表（用于 difficulty 分类统计）。
    返回 [{'name': 'easy', 'score': 0.5, 'num': 100}, ...] 或空列表。
    """
    try:
        metrics = dataset_result.get('metrics', [])
        if not (metrics and isinstance(metrics, list)):
            return []
        categories = metrics[0].get('categories', [])
        if not (categories and isinstance(categories, list)):
            return []
        subsets = categories[0].get('subsets', [])
        if subsets and isinstance(subsets, list) and len(subsets) > 0:
            return subsets
    except (IndexError, KeyError, TypeError, AttributeError):
        pass
    return []


def run_task_with_progress(task_config: TaskConfig, benchmark_name: str):
    """运行评测任务并显示进度条"""
    logger.info(f"[进度条] {benchmark_name} 开始评测")

    from threading import Thread, Event

    progress_bar = None
    stop_event = Event()

    try:
        progress_bar = tqdm(total=100, desc=f"【评测中】{benchmark_name}",
                           bar_format='{l_bar}{bar} | [{elapsed}<{remaining}]{postfix}')

        def update_spinner():
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            cycle = 0
            while not stop_event.is_set():
                cycle = (cycle + 1) % len(spinner)
                progress_bar.set_description(f"【评测中】{benchmark_name} {spinner[cycle]}")
                time.sleep(0.1)

        spinner_thread = Thread(target=update_spinner)
        spinner_thread.start()

        result = run_task(task_config)

        stop_event.set()
        spinner_thread.join()

        progress_bar.update(100)
        progress_bar.set_description(f"【已完成】{benchmark_name} ✓")
        progress_bar.close()

        logger.info(f"[进度条] {benchmark_name} 评测完成 ✓")
        return result

    except Exception as e:
        logger.error(f"[进度条] {benchmark_name} 评测失败 ✗")
        stop_event.set()
        if progress_bar:
            progress_bar.set_description(f"【失败】{benchmark_name} ✗")
            progress_bar.close()
        raise e


def _extract_model_name(model_path: str) -> str:
    """从模型路径提取简短名称"""
    path_parts = model_path.split('/')
    if 'checkpoint' in path_parts[-1].lower() and len(path_parts) > 1:
        return path_parts[-2]
    return path_parts[-1]


# ============================================================
# 离线 vLLM 引擎（替代 VLLMServiceManager）
# ============================================================

class VLLMWorker:
    """Ray actor：持有一个 vLLM LLM 实例，处理一个 DP 分片。"""

    def __init__(self, model_path: str, tp: int,
                 gpu_memory_utilization: float,
                 max_model_len: Optional[int],
                 enforce_eager: bool):
        from vllm import LLM as _LLM
        kwargs = dict(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
        )
        if max_model_len:
            kwargs['max_model_len'] = max_model_len
        self.llm = _LLM(**kwargs)

    def generate(self, prompts: List[str],
                 sampling_params_dict: dict) -> List[dict]:
        from vllm import SamplingParams as _SP
        outputs = self.llm.generate(prompts, _SP(**sampling_params_dict))
        return [
            {'text': o.outputs[0].text,
             'finish_reason': o.outputs[0].finish_reason}
            for o in outputs
        ]


class VLLMOfflineEngine:
    """离线 vLLM 推理引擎，支持 TP 和 Ray DP。

    - dp=1: 直接在当前进程创建 LLM，kill 进程即释放 GPU，零僵尸。
    - dp>1: 通过 Ray 创建多个 Worker actor，每个占 tp 张 GPU。
    """

    def __init__(self, model_path: str, tp: int = 1, dp: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 enforce_eager: bool = True):
        self.model_path = model_path
        self.tp = tp
        self.dp = dp
        self.model_name = _extract_model_name(model_path)

        logger.info(f"初始化 VLLMOfflineEngine: model={self.model_name}, "
                    f"tp={tp}, dp={dp}, 总GPU={tp * dp}")

        if dp == 1:
            # ---------- 单进程模式 ----------
            kwargs = dict(
                model=model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                trust_remote_code=True,
            )
            if max_model_len:
                kwargs['max_model_len'] = max_model_len
            self.llm = LLM(**kwargs)
            self.tokenizer = self.llm.get_tokenizer()
            self._workers = None
            logger.info("单进程离线模式初始化完成")
        else:
            # ---------- Ray DP 模式 ----------
            import ray
            ray.init(ignore_reinit_error=True)

            # 主进程只加载 tokenizer（几 MB），不占 GPU
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)

            Worker = ray.remote(num_gpus=tp)(VLLMWorker)
            self._workers = [
                Worker.remote(model_path, tp, gpu_memory_utilization,
                              max_model_len, enforce_eager)
                for _ in range(dp)
            ]
            self.llm = None
            logger.info(f"Ray DP 模式初始化完成，{dp} 个 Worker")

        # ---------- 请求批处理（线程安全） ----------
        # evalscope 用 ThreadPoolExecutor 逐条调 generate()，
        # 但 vLLM LLM.generate() 不是线程安全的。
        # 这里用队列 + 后台线程，收集并发请求后统一批量提交。
        self._batch_lock = threading.Lock()
        self._batch_queue: List[tuple] = []   # [(prompt, SamplingParams, Future), ...]
        self._batch_event = threading.Event()
        self._batch_running = True
        self._batch_wait = 0.1  # 等待 100ms 收集更多请求
        self._batch_thread = threading.Thread(
            target=self._batch_loop, daemon=True, name='vllm-batcher')
        self._batch_thread.start()

    # ---- 核心推理 ----

    def generate_single(self, prompt: str,
                        sampling_params: SamplingParams) -> dict:
        """线程安全的单条推理。请求会被自动批处理。

        evalscope 的 ThreadPoolExecutor 会并发调用此方法，
        内部通过队列 + 后台线程收集请求，批量提交给 vLLM。
        """
        future = concurrent.futures.Future()
        with self._batch_lock:
            self._batch_queue.append((prompt, sampling_params, future))
            self._batch_event.set()
        return future.result()

    def _batch_loop(self):
        """后台线程：收集并发请求，批量提交给 vLLM。"""
        while self._batch_running:
            # 等待至少一个请求到来
            self._batch_event.wait(timeout=1.0)
            if not self._batch_running:
                break

            # 短暂等待以收集更多并发请求
            # evalscope 的 ThreadPoolExecutor 会在极短时间内提交 eval_batch_size 个请求
            time.sleep(self._batch_wait)

            with self._batch_lock:
                if not self._batch_queue:
                    self._batch_event.clear()
                    continue
                batch = list(self._batch_queue)
                self._batch_queue.clear()
                self._batch_event.clear()

            prompts = [item[0] for item in batch]
            sp = batch[0][1]   # 同一 benchmark 的所有请求使用相同的采样参数
            futures = [item[2] for item in batch]

            logger.info(f"Batcher: 收集到 {len(prompts)} 个请求，批量推理中...")

            try:
                results = self._do_generate(prompts, sp)
                for fut, result in zip(futures, results):
                    fut.set_result(result)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _do_generate(self, prompts: List[str],
                     sampling_params: SamplingParams) -> List[dict]:
        """内部批量推理（非线程安全，仅由 _batch_loop 调用）。"""
        if not prompts:
            return []

        if self.dp == 1:
            outputs = self.llm.generate(prompts, sampling_params)
            return [
                {'text': o.outputs[0].text,
                 'finish_reason': o.outputs[0].finish_reason}
                for o in outputs
            ]

        # dp>1: 均分到各 Ray Worker 并行推理
        import ray
        sp_dict = self._sampling_params_to_dict(sampling_params)
        n = len(prompts)
        chunk_size = max(1, (n + self.dp - 1) // self.dp)
        chunks = [prompts[i:i + chunk_size]
                  for i in range(0, n, chunk_size)]

        ray_futures = [
            worker.generate.remote(chunk, sp_dict)
            for worker, chunk in zip(self._workers, chunks)
            if chunk
        ]
        chunk_results = ray.get(ray_futures)

        all_results = []
        for cr in chunk_results:
            all_results.extend(cr)
        return all_results

    # ---- 生命周期 ----

    def shutdown(self):
        """释放所有 GPU 资源。"""
        # 停止批处理线程
        self._batch_running = False
        self._batch_event.set()
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=5)

        if self._workers:
            import ray
            for w in self._workers:
                ray.kill(w)
            ray.shutdown()
            self._workers = None
            logger.info("Ray Workers 已关闭")

        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("LLM 实例已释放")

    # ---- 内部工具 ----

    @staticmethod
    def _sampling_params_to_dict(sp: SamplingParams) -> dict:
        """将 SamplingParams 序列化为 dict，用于 Ray 远程传输。"""
        d = {}
        for field in ('temperature', 'top_p', 'top_k',
                      'max_tokens', 'repetition_penalty'):
            val = getattr(sp, field, None)
            if val is not None:
                d[field] = val
        return d


# ============================================================
# evalscope ModelAPI 适配器
# ============================================================

class VLLMOfflineModelAPI(ModelAPI):
    """将 VLLMOfflineEngine 包装为 evalscope 的 ModelAPI 接口。

    evalscope 内部在 get_model() 中检测到 model 是 ModelAPI 实例时，
    会直接使用它，不再走 OpenAI HTTP 客户端。
    """

    def __init__(self, engine: VLLMOfflineEngine, model_name: str,
                 config: GenerateConfig = GenerateConfig()):
        super().__init__(model_name=model_name, config=config)
        self.engine = engine

    # ---- 消息 → prompt 转换 ----

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        msg_dicts = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else msg.text
            msg_dicts.append({"role": msg.role, "content": content})
        return self.engine.tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True)

    # ---- GenerateConfig → SamplingParams ----

    @staticmethod
    def _config_to_sampling_params(config: GenerateConfig) -> SamplingParams:
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

    # ---- ModelAPI 接口实现 ----

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = self._messages_to_prompt(input)
        sp = self._config_to_sampling_params(config)
        # generate_single 是线程安全的：请求进入队列，由后台线程批量提交
        r = self.engine.generate_single(prompt, sp)
        return ModelOutput.from_content(
            model=self.model_name,
            content=r['text'],
            stop_reason=as_stop_reason(r['finish_reason']),
        )


# ============================================================
# BenchmarkEvaluator（核心改动：用 engine 替代 vllm_manager）
# ============================================================

class BenchmarkEvaluator:
    """benchmark测评器"""

    def __init__(self, config_file: str, run_id: int = None,
                 version: str = None,
                 vllm_engine: Optional[VLLMOfflineEngine] = None):
        self.config_file = config_file
        self.config = self._load_config()
        self.vllm_engine = vllm_engine
        self.results = {}
        self.base_dir = Path(self.config['model_path'])
        self.version = version or datetime.now().strftime("version_%Y%m%d_%H%M%S")

        if run_id is not None:
            self.output_dir = self.base_dir / "evalscope" / self.version / f"run_{run_id}"
            self.run_id = run_id
        else:
            self.output_dir = self.base_dir / "evalscope" / self.version
            self.run_id = None

        self.debug_mode = self.config.get('debug', False)
        self.model_name = _extract_model_name(self.config['model_path'])

    def _load_config(self) -> Dict:
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_task_config(self, benchmark_name: str) -> TaskConfig:
        """创建 TaskConfig 配置"""
        benchmark_config = self.config['benchmarks'][benchmark_name]

        logger.info(f"开始创建 benchmark {benchmark_name} 的 TaskConfig")

        benchmark_judge_strategy = benchmark_config.get(
            'judge_strategy', JudgeStrategy.RULE)
        use_llm = 'llm' in str(benchmark_judge_strategy).lower()

        task_config = TaskConfig()

        # ===== 关键改动：传入离线 ModelAPI，不再走 HTTP =====
        model_api = VLLMOfflineModelAPI(
            engine=self.vllm_engine,
            model_name=self.model_name,
        )
        task_config.model = model_api
        task_config.model_id = self.model_name
        task_config.eval_backend = EvalBackend.NATIVE

        # 数据集
        task_config.datasets = [benchmark_config['dataset_name']]
        if 'limit' in benchmark_config:
            task_config.limit = benchmark_config['limit']
        task_config.repeats = benchmark_config.get('repeats', 1)

        dataset_args = {
            'dataset_name': benchmark_config.get('subset', None),
            'few_shot_num': benchmark_config.get('few_shot_num', 0)
        }
        if 'extra_params' in benchmark_config:
            dataset_args['extra_params'] = benchmark_config['extra_params']
        task_config.dataset_args = {
            benchmark_config['dataset_name']: dataset_args}

        # 生成参数
        generation_config_dict = {
            'temperature': benchmark_config.get('temperature', 0.0),
            'top_p': benchmark_config.get('top_p', 1.0),
            'top_k': benchmark_config.get('top_k', 50),
            'max_tokens': benchmark_config.get('max_tokens', 2048),
            'repetition_penalty': benchmark_config.get('repetition_penalty', 1.0)
        }
        gen_config = benchmark_config.get('gen_config', {})
        if gen_config:
            generation_config_dict.update(gen_config)
        task_config.generation_config = GenerateConfig.model_validate(
            generation_config_dict)

        # 批次大小（控制 evalscope ThreadPoolExecutor 的并发线程数）
        batch_size = self.config.get('eval', {}).get('batch_size',
                     self.config.get('eval', {}).get('parallel', 64))
        task_config.eval_batch_size = batch_size
        task_config.generation_config.batch_size = batch_size

        # 工作目录
        task_config.work_dir = str(self.output_dir / benchmark_name)

        # Sandbox 配置
        if 'use_sandbox' in benchmark_config:
            task_config.use_sandbox = benchmark_config['use_sandbox']
            sandbox_workers = benchmark_config.get('sandbox_workers', 5)
            task_config.judge_worker_num = sandbox_workers

            sandbox_config = self.config.get('sandbox', {})
            remote_url = sandbox_config.get('remote_url')
            if remote_url:
                remote_config = {'base_url': remote_url}
                if sandbox_config.get('timeout'):
                    remote_config['timeout'] = sandbox_config['timeout']
                task_config.sandbox_manager_config = remote_config
            elif sandbox_config.get('work_dir'):
                sandbox_work_dir = sandbox_config['work_dir']
                Path(sandbox_work_dir).mkdir(parents=True, exist_ok=True)
                task_config.sandbox_manager_config = {
                    'work_dir': sandbox_work_dir}

        # Judge 配置
        task_config.judge_strategy = benchmark_judge_strategy
        if use_llm:
            api_key = self.config.get('API_KEY', '')
            base_url = self.config.get('BASE_URL', '').replace(
                '/chat/completions', '').rstrip('/')
            judge_model = benchmark_config.get(
                'judge_model', 'Qwen/Qwen3-8B')
            if not api_key or not base_url:
                raise ValueError(
                    f"[{benchmark_name}] LLM judge 需要 API_KEY 和 BASE_URL")
            task_config.judge_model_args = {
                'api_key': api_key,
                'api_url': base_url,
                'model_id': judge_model,
                'generation_config': {
                    'temperature': 0.0, 'max_tokens': 1024,
                    'extra_body': {'enable_thinking': False}},
            }
            judge_parallel = benchmark_config.get('judge_parallel', 1)
            task_config.judge_worker_num = judge_parallel

        logger.info(f"[{benchmark_name}] TaskConfig创建完成: "
                    f"model={task_config.model_id}, "
                    f"datasets={task_config.datasets}, "
                    f"batch_size={batch_size}")
        return task_config

    def _save_config(self):
        config_path = self.output_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"配置已保存至: {config_path}")

    def run_evaluation(self):
        """运行评测"""
        run_info = f" (第 {self.run_id} 次运行)" if self.run_id else ""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始运行评估{run_info} - 模型: {self.model_name}")
        logger.info(f"配置文件: {self.config_file}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"{'='*60}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        manage_engine = False
        if self.vllm_engine is None:
            # 自行创建引擎（单次运行模式）
            vllm_cfg = self.config.get('vllm', {})
            self.vllm_engine = VLLMOfflineEngine(
                model_path=self.config['model_path'],
                tp=vllm_cfg.get('tp', 1),
                dp=vllm_cfg.get('dp', 1),
                gpu_memory_utilization=vllm_cfg.get(
                    'gpu_memory_utilization', 0.9),
                max_model_len=vllm_cfg.get('max_model_len', None),
                enforce_eager=vllm_cfg.get('enforce_eager', True),
            )
            manage_engine = True

        try:
            # 检查有效 benchmarks
            valid_benchmarks = []
            for benchmark_name in self.config['benchmarks'].keys():
                try:
                    self._create_task_config(benchmark_name)
                    valid_benchmarks.append(benchmark_name)
                except ValueError as e:
                    logger.warning(
                        f"跳过 benchmark {benchmark_name}: {str(e)}")

            if not valid_benchmarks:
                logger.error("没有有效的 benchmark 可以评估")
                return

            logger.info(f"有效 benchmarks: {valid_benchmarks}")
            total = len(valid_benchmarks)

            for idx, benchmark_name in enumerate(valid_benchmarks):
                repeats = self.config['benchmarks'][benchmark_name].get(
                    'repeats', 1)
                logger.info(f"\n{'='*50}")
                logger.info(
                    f"正在评估 {benchmark_name} "
                    f"(repeat: {repeats}次) "
                    f"(剩余: {total - idx - 1}个)")
                logger.info(f"模型: {self.model_name}")

                task_config = self._create_task_config(benchmark_name)

                logger.info(f"开始评测 {benchmark_name}...")
                result = run_task_with_progress(task_config, benchmark_name)
                self.results[benchmark_name] = result
                logger.info(
                    f"完成评测 {benchmark_name} ({idx+1}/{total})")

        except Exception as e:
            logger.error(f"评测过程中出错: {str(e)}")
            raise
        finally:
            if manage_engine and self.vllm_engine:
                self.vllm_engine.shutdown()
                self.vllm_engine = None

    def _convert_to_serializable(self, obj):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v)
                    for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
        if hasattr(obj, 'to_dict'):
            return self._convert_to_serializable(obj.to_dict())
        return str(obj)

    def save_results(self):
        results_summary = self._prepare_results_summary()

        excel_path = self.output_dir / "summary.xlsx"
        results_summary.to_excel(excel_path, index=False)
        logger.info(f"Excel汇总结果已保存至: {excel_path}")

        json_path = self.output_dir / "detailed_results.json"
        serializable_results = self._convert_to_serializable(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存至: {json_path}")

    def _prepare_results_summary(self) -> pd.DataFrame:
        summary_data = {'benchmark': [], 'dataset': [], 'score': []}
        serializable_results = self._convert_to_serializable(self.results)

        for benchmark_name, result in serializable_results.items():
            if isinstance(result, dict):
                for dataset_name, dataset_result in result.items():
                    if not isinstance(dataset_result, dict):
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(dataset_name)
                        summary_data['score'].append('N/A')
                        continue

                    overall_score = dataset_result.get(
                        'score', dataset_result.get('accuracy', 'N/A'))
                    subsets = _extract_report_subsets(dataset_result)

                    if subsets:
                        for subset in subsets:
                            summary_data['benchmark'].append(benchmark_name)
                            summary_data['dataset'].append(
                                f"{dataset_name}/{subset['name']}")
                            summary_data['score'].append(
                                round(subset.get('score', 0), 4))
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(
                            f"{dataset_name}/all")
                        summary_data['score'].append(overall_score)
                    else:
                        summary_data['benchmark'].append(benchmark_name)
                        summary_data['dataset'].append(dataset_name)
                        summary_data['score'].append(overall_score)
            else:
                summary_data['benchmark'].append(benchmark_name)
                summary_data['dataset'].append('unknown')
                summary_data['score'].append('N/A')

        return pd.DataFrame(summary_data)


# ============================================================
# MultiRunEvaluator（改用 engine）
# ============================================================

class MultiRunEvaluator:
    """多次运行评测器，用于统计方差"""

    def __init__(self, config_file: str, num_runs: int = 1):
        self.config_file = config_file
        self.num_runs = num_runs
        self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = Path(self.config['model_path'])
        self.output_dir = self.base_dir / "evalscope" / self.version
        self.all_results = []

    def run_all(self):
        logger.info(f"\n{'='*70}")
        logger.info(f"开始多次运行评测 - 共 {self.num_runs} 次")
        logger.info(f"总输出目录: {self.output_dir}")
        logger.info(f"{'='*70}\n")

        # 创建离线引擎（一次加载，多次评测复用）
        vllm_cfg = self.config.get('vllm', {})
        engine = VLLMOfflineEngine(
            model_path=self.config['model_path'],
            tp=vllm_cfg.get('tp', 1),
            dp=vllm_cfg.get('dp', 1),
            gpu_memory_utilization=vllm_cfg.get(
                'gpu_memory_utilization', 0.9),
            max_model_len=vllm_cfg.get('max_model_len', None),
            enforce_eager=vllm_cfg.get('enforce_eager', True),
        )

        try:
            for run_id in range(1, self.num_runs + 1):
                logger.info(f"\n{'#'*70}")
                logger.info(f"# 第 {run_id}/{self.num_runs} 次运行")
                logger.info(f"{'#'*70}\n")

                evaluator = BenchmarkEvaluator(
                    self.config_file, run_id=run_id,
                    version=self.version, vllm_engine=engine)
                evaluator.run_evaluation()
                evaluator.save_results()

                self.all_results.append({
                    'run_id': run_id,
                    'results': evaluator.results,
                    'output_dir': evaluator.output_dir
                })
                logger.info(f"第 {run_id} 次运行完成")

                if run_id < self.num_runs:
                    logger.info("等待 5 秒后开始下一次运行...")
                    time.sleep(5)

            self._summarize_results()

        finally:
            engine.shutdown()

    def _summarize_results(self):
        logger.info(f"\n{'='*70}")
        logger.info("汇总多次运行结果")
        logger.info(f"{'='*70}\n")

        benchmark_scores = {}

        def _serialize(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if hasattr(obj, 'to_dict'):
                return _serialize(obj.to_dict())
            if hasattr(obj, '__dict__'):
                return _serialize(obj.__dict__)
            return str(obj)

        for run_data in self.all_results:
            results = _serialize(run_data['results'])
            for benchmark_name, result in results.items():
                if benchmark_name not in benchmark_scores:
                    benchmark_scores[benchmark_name] = {}
                if not isinstance(result, dict):
                    continue
                for dataset_name, dataset_result in result.items():
                    if not isinstance(dataset_result, dict):
                        continue
                    overall_score = dataset_result.get(
                        'score', dataset_result.get('accuracy'))
                    subsets = _extract_report_subsets(dataset_result)
                    if subsets:
                        score_entries = [
                            (f"{dataset_name}/{s['name']}", s.get('score'))
                            for s in subsets]
                        score_entries.append(
                            (f"{dataset_name}/all", overall_score))
                    else:
                        score_entries = [(dataset_name, overall_score)]
                    for key_suffix, score in score_entries:
                        full_key = f"{benchmark_name}/{key_suffix}"
                        if full_key not in benchmark_scores[benchmark_name]:
                            benchmark_scores[benchmark_name][full_key] = {
                                'scores': [],
                                'dataset_name': key_suffix,
                            }
                        if score is not None:
                            benchmark_scores[benchmark_name][full_key][
                                'scores'].append(score)

        import numpy as np

        summary_rows = []
        for benchmark_name, datasets in benchmark_scores.items():
            for key, data in datasets.items():
                scores = data['scores']
                dataset_name = data['dataset_name']
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = (np.std(scores, ddof=1)
                                 if len(scores) > 1 else 0.0)
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    row = {
                        'benchmark': benchmark_name,
                        'dataset': dataset_name,
                        'run_count': len(scores),
                        'mean': round(mean_score, 4),
                        'std': round(std_score, 4),
                        'std_percent': (round(std_score / mean_score * 100, 2)
                                        if mean_score > 0 else 0),
                        'min': round(min_score, 4),
                        'max': round(max_score, 4),
                        'range': round(max_score - min_score, 4)
                    }
                    for i, score in enumerate(scores, 1):
                        row[f'run_{i}'] = round(score, 4)
                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        excel_path = self.output_dir / "multi_run_summary.xlsx"
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"多次运行汇总结果已保存至: {excel_path}")

        json_path = self.output_dir / "multi_run_summary.json"
        summary_data = {
            'num_runs': self.num_runs,
            'version': self.version,
            'config_file': self.config_file,
            'results': summary_rows
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info(f"详细汇总结果已保存至: {json_path}")

        logger.info("\n多次运行结果汇总:")
        logger.info("-" * 100)
        for _, row in summary_df.iterrows():
            logger.info(
                f"{row['benchmark']}/{row['dataset']}: "
                f"均值={row['mean']:.4f}, "
                f"标准差={row['std']:.4f} ({row['std_percent']:.2f}%), "
                f"范围=[{row['min']:.4f}, {row['max']:.4f}]")
        logger.info("-" * 100)

        return summary_df


# ============================================================
# 主函数
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python eval.py <config_file> [num_runs]")
        print("例如: python eval.py eval_configs/example.yaml")
        print("例如: python eval.py eval_configs/example.yaml 5  # 运行5次")
        sys.exit(1)

    config_file = sys.argv[1]

    num_runs = 1
    if len(sys.argv) >= 3:
        try:
            num_runs = int(sys.argv[2])
            if num_runs < 1:
                print("运行次数必须大于等于1")
                sys.exit(1)
        except ValueError:
            print(f"无效的运行次数: {sys.argv[2]}")
            sys.exit(1)

    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        sys.exit(1)

    logger.info(f"开始运行评测，配置文件: {config_file}, 运行次数: {num_runs}")

    if num_runs > 1:
        multi_evaluator = MultiRunEvaluator(config_file, num_runs=num_runs)
        multi_evaluator.run_all()
    else:
        evaluator = BenchmarkEvaluator(config_file)
        evaluator.run_evaluation()
        evaluator.save_results()

    logger.info("评测完成")


if __name__ == "__main__":
    main()
