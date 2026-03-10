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
        outputs = self.llm.generate(prompts, _SP(**sampling_params_dict), use_tqdm=False)
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
                 enforce_eager: bool = True,
                 worker_chunk_size: int = 512):
        self.model_path = model_path
        self.tp = tp
        self.dp = dp
        self.model_name = _extract_model_name(model_path)
        # dp>1 时每次给每个 Worker 的 prompt 数，控制负载均衡粒度
        self._worker_chunk_size = worker_chunk_size

        logger.info(f"初始化 VLLMOfflineEngine: model={self.model_name}, "
                    f"tp={tp}, dp={dp}, 总GPU={tp * dp}"
                    + (f", chunk_size={worker_chunk_size}" if dp > 1 else ""))

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
        # 这里用队列 + 后台调度线程统一分发。
        self._batch_lock = threading.Lock()
        self._batch_queue: List[tuple] = []   # [(prompt, SamplingParams, Future), ...]
        self._batch_event = threading.Event()
        self._batch_running = True
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
        """后台调度线程入口：根据 dp 选择调度策略。"""
        if self.dp == 1:
            self._batch_loop_single()
        else:
            self._batch_loop_dp()

    def _batch_loop_single(self):
        """dp=1：收集请求后整体批量推理（vLLM 内部做 continuous batching）。"""
        while self._batch_running:
            self._batch_event.wait(timeout=1.0)
            if not self._batch_running:
                break
            # 短暂等待，让 evalscope ThreadPool 把同一波请求都提交进来
            time.sleep(0.05)
            with self._batch_lock:
                if not self._batch_queue:
                    self._batch_event.clear()
                    continue
                batch = list(self._batch_queue)
                self._batch_queue.clear()
                self._batch_event.clear()

            prompts = [x[0] for x in batch]
            sp = batch[0][1]
            futures = [x[2] for x in batch]
            logger.info(f"Batcher(dp=1): {len(prompts)} 个请求，推理中...")
            try:
                outputs = self.llm.generate(prompts, sp, use_tqdm=False)
                results = [{'text': o.outputs[0].text,
                            'finish_reason': o.outputs[0].finish_reason}
                           for o in outputs]
                for fut, res in zip(futures, results):
                    fut.set_result(res)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _batch_loop_dp(self):
        """dp>1：连续调度，Worker 完成后立即接收新任务，保证满负荷运行。

        设计要点：
        - 每次只给每个 Worker _worker_chunk_size 个 prompt（而非平均分割全量）
        - 用 ray.wait(num_returns=1) 检测最先完成的 Worker，立刻分配新任务
        - 绝不等待所有 Worker 同步完成，消除慢 Worker 拖慢快 Worker 的问题
        """
        import ray as _ray

        chunk = self._worker_chunk_size
        pending: List[tuple] = []   # (prompt, sp, client_future)
        inflight: dict = {}         # ray_ref -> (worker_idx, [client_future, ...])

        def _dispatch():
            """将 pending 中的 prompt 均匀分配给所有空闲 Worker（每个最多 chunk 个）。"""
            if not pending:
                return
            busy = {v[0] for v in inflight.values()}
            free = [i for i in range(self.dp) if i not in busy]
            if not free:
                return
            sp_dict = self._sampling_params_to_dict(pending[0][1])
            # 均分给空闲 Worker，单次上限 chunk，避免把所有任务堆给一个 Worker
            per_worker = min(chunk, max(1, len(pending) // len(free)))
            for w_idx in free:
                if not pending:
                    break
                take = pending[:per_worker]
                del pending[:per_worker]
                ref = self._workers[w_idx].generate.remote(
                    [x[0] for x in take], sp_dict)
                inflight[ref] = (w_idx, [x[2] for x in take])
                logger.info(f"Dispatcher: Worker-{w_idx} <- {len(take)} prompts "
                            f"(pending={len(pending)}, inflight={len(inflight)})")

        while True:
            # 1. 等待：优先用 ray.wait 阻塞（检测 Worker 完成），无 inflight 时等新请求
            if inflight:
                done, _ = _ray.wait(
                    list(inflight.keys()), num_returns=1, timeout=0.05)
            else:
                self._batch_event.wait(timeout=0.05)
                done = []

            # 2. 收集新到达的请求
            with self._batch_lock:
                if self._batch_queue:
                    pending.extend(self._batch_queue)
                    self._batch_queue.clear()
                self._batch_event.clear()

            # 3. 处理已完成的 Worker → 解决其 Future，立刻触发 _dispatch
            for ref in done:
                w_idx, cfuts = inflight.pop(ref)
                try:
                    results = _ray.get(ref)
                    for fut, res in zip(cfuts, results):
                        if not fut.done():
                            fut.set_result(res)
                except Exception as e:
                    for fut in cfuts:
                        if not fut.done():
                            fut.set_exception(e)

            # 4. 将 pending 分配给空闲 Worker
            _dispatch()

            # 5. 退出：_batch_running=False 且所有任务处理完毕
            if not self._batch_running and not pending and not inflight:
                break

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
                 vllm_engine: Optional[VLLMOfflineEngine] = None,
                 completed_benchmarks: Optional[set] = None):
        self.config_file = config_file
        self.config = self._load_config()
        self.vllm_engine = vllm_engine
        self.results = {}
        self.base_dir = Path(self.config['model_path'])
        self.version = version or datetime.now().strftime("version_%Y%m%d_%H%M%S")
        # 断点续传：已完成的 benchmark 列表（跳过推理，直接加载已有结果）
        self.completed_benchmarks = completed_benchmarks or set()

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

    def _save_benchmark_result(self, benchmark_name: str, result):
        """每个 benchmark 完成后立即持久化，用于断点续传。"""
        result_path = self.output_dir / f"{benchmark_name}_result.json"
        serializable = self._convert_to_serializable(result)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"[断点续传] {benchmark_name} 结果已保存: {result_path}")

    def _load_benchmark_result(self, benchmark_name: str):
        """加载已有的 benchmark 结果（断点续传时使用）。

        优先读取我们自己保存的 {benchmark}_result.json；
        若不存在（旧版运行或首次引入该机制前的结果），则从 evalscope 原生
        reports/ 目录中读取 JSON 并重建 {dataset_name: report_dict} 格式。
        """
        # 1. 新机制：直接读取
        result_path = self.output_dir / f"{benchmark_name}_result.json"
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 2. 旧机制回退：从 evalscope reports/ 目录读取
        bm_dir = self.output_dir / benchmark_name
        if not bm_dir.is_dir():
            return None
        # benchmark 目录下有一个时间戳子目录（evalscope 自动生成）
        ts_dirs = sorted(bm_dir.iterdir(), key=lambda d: d.name)
        for ts_dir in ts_dirs:
            reports_dir = ts_dir / "reports"
            if not reports_dir.is_dir():
                continue
            result = {}
            for model_dir in reports_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for report_file in model_dir.glob("*.json"):
                    with open(report_file, 'r', encoding='utf-8') as f:
                        result[report_file.stem] = json.load(f)
            if result:
                logger.info(f"[断点续传] {benchmark_name} 从 evalscope reports/ 加载结果")
                return result
        return None

    def _find_partial_prediction_dir(self, benchmark_name: str) -> Optional[str]:
        """查找已有 predictions 但尚无 reports 的时间戳目录。

        这类目录说明上次推理中途中断，可通过 evalscope use_cache 机制续跑。
        只在非 completed_benchmarks 场景下被调用（completed 的直接跳过）。
        """
        bm_dir = self.output_dir / benchmark_name
        if not bm_dir.is_dir():
            return None
        # 取最新的时间戳子目录（evalscope 按时间命名）
        for ts_dir in sorted(bm_dir.iterdir(), key=lambda d: d.name, reverse=True):
            if not ts_dir.is_dir():
                continue
            pred_dir = ts_dir / "predictions"
            rep_dir = ts_dir / "reports"
            has_predictions = pred_dir.is_dir() and any(pred_dir.rglob("*.jsonl"))
            has_reports = rep_dir.is_dir() and any(rep_dir.rglob("*.json"))
            if has_predictions and not has_reports:
                return str(ts_dir)
        return None

    @staticmethod
    def _count_existing_predictions(ts_dir: str) -> int:
        """统计已有 predictions JSONL 中的条目数（用于日志展示）。"""
        total = 0
        pred_root = Path(ts_dir) / "predictions"
        if pred_root.is_dir():
            for jsonl in pred_root.rglob("*.jsonl"):
                try:
                    total += sum(1 for _ in open(jsonl, 'rb'))
                except OSError:
                    pass
        return total

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

        # evalscope ThreadPoolExecutor 并发数：足够大，让 pending 队列始终有余量
        # 供连续调度器（_batch_loop_dp）分配给所有 Worker，无需手动配置
        vllm_cfg = self.config.get('vllm', {})
        _dp = vllm_cfg.get('dp', 1)
        _chunk = vllm_cfg.get('worker_chunk_size', 512)
        # dp=1 用适中并发；dp>1 至少能填满所有 Worker 的 2 轮 chunk
        eval_batch_size = _chunk if _dp == 1 else max(2048, _dp * _chunk * 2)
        task_config.eval_batch_size = eval_batch_size
        task_config.generation_config.batch_size = eval_batch_size

        # 工作目录
        task_config.work_dir = str(self.output_dir / benchmark_name)

        # prediction 级断点续传：检测已有 predictions（无 reports）→ 设置 use_cache
        # evalscope 的 use_cache 机制会读取已有 JSONL、自动过滤已预测 sample、只跑剩余部分
        partial_pred_dir = self._find_partial_prediction_dir(benchmark_name)
        if partial_pred_dir:
            task_config.use_cache = partial_pred_dir
            existing = self._count_existing_predictions(partial_pred_dir)
            logger.info(
                f"[prediction续传] {benchmark_name}: 检测到 {existing} 条已有 predictions，"
                f"续跑剩余部分（目录: {partial_pred_dir}）")

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
                    f"eval_batch_size={eval_batch_size}")
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
                worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
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

                # 断点续传：跳过已完成的 benchmark
                if benchmark_name in self.completed_benchmarks:
                    saved = self._load_benchmark_result(benchmark_name)
                    if saved is not None:
                        self.results[benchmark_name] = saved
                        logger.info(
                            f"[断点续传] 跳过 {benchmark_name}，加载已有结果 "
                            f"({idx+1}/{total})")
                        continue
                    else:
                        logger.warning(
                            f"[断点续传] {benchmark_name} 标记为已完成但结果文件缺失，重新评测")

                logger.info(
                    f"正在评估 {benchmark_name} "
                    f"(repeat: {repeats}次) "
                    f"(剩余: {total - idx - 1}个)")
                logger.info(f"模型: {self.model_name}")

                task_config = self._create_task_config(benchmark_name)

                logger.info(f"开始评测 {benchmark_name}...")
                result = run_task_with_progress(task_config, benchmark_name)
                self.results[benchmark_name] = result
                # 立即保存，确保中断后可续跑
                self._save_benchmark_result(benchmark_name, result)
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
    """多次运行评测器，用于统计方差。支持断点续传（--resume）。"""

    _PROGRESS_FILE = "progress.json"

    def __init__(self, config_file: str, num_runs: int = 1,
                 resume: bool = False, resume_dir: Optional[str] = None):
        self.config_file = config_file
        self.num_runs = num_runs
        self.resume = resume or (resume_dir is not None)

        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = Path(self.config['model_path'])
        self.all_results = []

        if resume_dir is not None:
            self._init_resume(Path(resume_dir))
        elif resume:
            self._init_resume(None)
        else:
            self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
            self.output_dir = self.base_dir / "evalscope" / self.version

    # ---- 断点续传 ----

    def _init_resume(self, specified_dir: Optional[Path]):
        """定位续跑目录。
        - specified_dir 不为 None：直接使用该目录（支持旧版无 progress.json 的目录）
        - specified_dir 为 None：自动选取 evalscope/ 下最新的 version_* 目录
        """
        if specified_dir is not None:
            # 用户指定了目录
            target = specified_dir.resolve()
            if not target.is_dir():
                logger.warning(
                    f"[断点续传] 指定目录不存在: {target}，将新建评测")
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = self.base_dir / "evalscope" / self.version
                return
            self.output_dir = target
            self.version = target.name
            logger.info(f"[断点续传] 使用指定目录: {target}")
        else:
            # 自动检测：取 evalscope/ 下最新的 version_* 目录
            evalscope_dir = self.base_dir / "evalscope"
            candidates = []
            if evalscope_dir.exists():
                candidates = sorted(
                    [d for d in evalscope_dir.iterdir()
                     if d.is_dir() and d.name.startswith("version_")],
                    key=lambda d: d.name, reverse=True
                )
            if not candidates:
                logger.info("[断点续传] 未找到历史版本目录，将新建评测")
                self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
                self.output_dir = self.base_dir / "evalscope" / self.version
                return
            self.output_dir = candidates[0]
            self.version = candidates[0].name
            logger.info(f"[断点续传] 自动选取最新目录: {self.output_dir}")

        # 从 progress.json 读取历史 num_runs（如有）
        progress_path = self.output_dir / self._PROGRESS_FILE
        if progress_path.exists():
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logger.info(
                f"[断点续传] 版本: {self.version}，"
                f"原始运行次数: {progress.get('num_runs', '?')}")

    def _save_progress(self):
        """保存 progress.json（版本标记，供下次 --resume 定位）。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        progress = {
            'version': self.version,
            'num_runs': self.num_runs,
            'config_file': self.config_file,
            'created_at': datetime.now().isoformat(),
        }
        with open(self.output_dir / self._PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _get_completed_benchmarks(self, run_id: int) -> set:
        """扫描 run_N 目录，返回已完成的 benchmark 集合。

        判定规则（按优先级）：
        1. 存在 {benchmark}_result.json  → 完成（新机制）
        2. evalscope 原生 reports/ 目录中存在 *.json → 完成（旧机制回退）
        只有 predictions/ 没有 reports/ → 未完成（推理中断）
        """
        run_dir = self.output_dir / f"run_{run_id}"
        if not run_dir.exists():
            return set()

        completed = set()

        # 新机制：{benchmark}_result.json
        for p in run_dir.glob('*_result.json'):
            completed.add(p.stem[:-len('_result')])

        # 旧机制回退：扫描 evalscope 原生目录结构
        for bm_dir in run_dir.iterdir():
            if not bm_dir.is_dir() or bm_dir.name.endswith('_result.json'):
                continue
            bm_name = bm_dir.name
            if bm_name in completed:
                continue
            # 在时间戳子目录下找 reports/*.json
            for ts_dir in bm_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                reports_dir = ts_dir / "reports"
                if reports_dir.is_dir() and any(reports_dir.rglob("*.json")):
                    completed.add(bm_name)
                    break

        return completed

    # ---- 主流程 ----

    def run_all(self):
        logger.info(f"\n{'='*70}")
        logger.info(f"开始多次运行评测 - 共 {self.num_runs} 次")
        logger.info(f"总输出目录: {self.output_dir}")
        logger.info(f"{'='*70}\n")

        # 保存进度标记（首次或续跑均保存，供下次 --resume 定位）
        self._save_progress()

        all_benchmarks = set(self.config.get('benchmarks', {}).keys())

        # 判断哪些 run 还有待完成的 benchmark，决定是否需要引擎
        run_plan = []  # list of (run_id, done_set, remaining_list)
        for run_id in range(1, self.num_runs + 1):
            done = self._get_completed_benchmarks(run_id)
            remaining = [b for b in self.config['benchmarks'] if b not in done]
            run_plan.append((run_id, done, remaining))

        needs_engine = any(remaining for _, _, remaining in run_plan)

        engine = None
        if needs_engine:
            vllm_cfg = self.config.get('vllm', {})
            engine = VLLMOfflineEngine(
                model_path=self.config['model_path'],
                tp=vllm_cfg.get('tp', 1),
                dp=vllm_cfg.get('dp', 1),
                gpu_memory_utilization=vllm_cfg.get(
                    'gpu_memory_utilization', 0.9),
                max_model_len=vllm_cfg.get('max_model_len', None),
                enforce_eager=vllm_cfg.get('enforce_eager', True),
                worker_chunk_size=vllm_cfg.get('worker_chunk_size', 512),
            )

        try:
            for run_id, done_benchmarks, remaining in run_plan:
                logger.info(f"\n{'#'*70}")
                logger.info(f"# 第 {run_id}/{self.num_runs} 次运行")
                if done_benchmarks:
                    logger.info(f"# [断点续传] 已完成: {sorted(done_benchmarks)}")
                    logger.info(f"# [断点续传] 待运行: {remaining}")
                logger.info(f"{'#'*70}\n")

                if not remaining:
                    # 整个 run 已完成，直接从磁盘加载结果
                    logger.info(f"第 {run_id} 次运行已全部完成，加载已有结果")
                    run_dir = self.output_dir / f"run_{run_id}"
                    results = {}
                    for bm in all_benchmarks:
                        result_path = run_dir / f"{bm}_result.json"
                        if result_path.exists():
                            with open(result_path, 'r', encoding='utf-8') as f:
                                results[bm] = json.load(f)
                    self.all_results.append({
                        'run_id': run_id,
                        'results': results,
                        'output_dir': run_dir
                    })
                    continue

                evaluator = BenchmarkEvaluator(
                    self.config_file, run_id=run_id,
                    version=self.version, vllm_engine=engine,
                    completed_benchmarks=done_benchmarks)
                evaluator.run_evaluation()
                evaluator.save_results()

                self.all_results.append({
                    'run_id': run_id,
                    'results': evaluator.results,
                    'output_dir': evaluator.output_dir
                })
                logger.info(f"第 {run_id} 次运行完成")

                if run_id < self.num_runs and remaining:
                    logger.info("等待 5 秒后开始下一次运行...")
                    time.sleep(5)

            self._summarize_results()

        finally:
            if engine is not None:
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
    import argparse
    parser = argparse.ArgumentParser(
        description="EvalScope 离线评测脚本",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config_file', help="评测配置文件路径")
    parser.add_argument('num_runs', nargs='?', type=int, default=1,
                        help="运行次数（默认 1，>1 时统计均值/方差）")
    parser.add_argument('--resume', action='store_true',
                        help="断点续传：自动选取最新 version_* 目录，跳过已完成的 benchmark 继续评测")
    parser.add_argument('--resume-dir', metavar='DIR',
                        help="断点续传：直接指定上次的版本目录路径\n"
                             "例: --resume-dir /data/.../evalscope/version_20260310_043148")
    args = parser.parse_args()

    if args.num_runs < 1:
        print("运行次数必须大于等于 1")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        print(f"配置文件不存在: {args.config_file}")
        sys.exit(1)

    is_resume = args.resume or bool(args.resume_dir)
    resume_hint = (f"，断点续传模式（指定目录: {args.resume_dir}）" if args.resume_dir
                   else "，断点续传模式（自动检测）" if args.resume else "")
    logger.info(
        f"开始运行评测，配置文件: {args.config_file}, "
        f"运行次数: {args.num_runs}{resume_hint}")

    # num_runs=1 且无 resume 时走单次路径；其他情况均走 MultiRunEvaluator
    if args.num_runs == 1 and not is_resume:
        evaluator = BenchmarkEvaluator(args.config_file)
        evaluator.run_evaluation()
        evaluator.save_results()
    else:
        multi_evaluator = MultiRunEvaluator(
            args.config_file, num_runs=args.num_runs,
            resume=args.resume, resume_dir=args.resume_dir)
        multi_evaluator.run_all()

    logger.info("评测完成")


if __name__ == "__main__":
    main()
