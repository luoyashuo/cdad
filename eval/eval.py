#!/usr/bin/env python3
"""
EvalScope测评脚本
用于自动化运行多个benchmark测评任务
"""

import os
import sys
import json
import time
import random
import logging
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import yaml
import requests

# 添加evalscope路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evalscope'))
from evalscope.config import TaskConfig
from evalscope.run import run_task
from evalscope.constants import EvalBackend, JudgeStrategy
from evalscope.utils.logger import get_logger
from evalscope.api.model.generate_config import GenerateConfig

logger = get_logger()

def run_task_with_progress(task_config: TaskConfig, benchmark_name: str):
    """运行评测任务并显示进度条"""
    logger.info(f"[进度条] {benchmark_name} 开始评测")

    # 使用tqdm包装run_task的执行
    from threading import Thread, Event

    progress_bar = None
    stop_event = Event()

    try:
        # 创建一个带有spinner动画的进度条
        progress_bar = tqdm(total=100, desc=f"【评测中】{benchmark_name}",
                           bar_format='{l_bar}{bar} | [{elapsed}<{remaining}]{postfix}')

        def update_spinner():
            """更新进度条的简单动画"""
            spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            cycle = 0
            while not stop_event.is_set():
                cycle = (cycle + 1) % len(spinner)
                spinner_char = spinner[cycle]
                progress_bar.set_description(f"【评测中】{benchmark_name} {spinner_char}")
                time.sleep(0.1)

        # 启动spinner线程
        spinner_thread = Thread(target=update_spinner)
        spinner_thread.start()

        # 运行实际的任务
        result = run_task(task_config)

        # 停止spinner
        stop_event.set()
        spinner_thread.join()

        # 标记任务完成
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


class VLLMServiceManager:
    """管理vLLM服务的启动和停止"""

    def __init__(self, model_path: str, tp: int = 1, gpu_memory_utilization: float = 0.9,
                 timeout: float = None, max_model_len: int = None):
        self.model_path = model_path
        self.tp = tp
        self.gpu_memory_utilization = gpu_memory_utilization
        self.timeout = timeout or 60  # 默认10分钟超时
        self.max_model_len = max_model_len  # 最大模型长度，与benchmark的max_tokens保持一致
        self.process = None
        self.api_url = "http://localhost:8000"
        self.log_file = None
        # 从路径中提取合适的模型名称
        path_parts = model_path.split('/')
        if 'checkpoint' in path_parts[-1].lower() and len(path_parts) > 1:
            self.model_name = path_parts[-2]
        else:
            self.model_name = path_parts[-1]

        # 设置vLLM日志目录（相对于脚本所在目录）
        self.log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "vllm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self, timeout: int = 300) -> bool:
        """启动vLLM服务"""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.model_name,  # 显式指定服务模型名称
            "--tensor-parallel-size", str(self.tp),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--port", "8000",
            "--host", "0.0.0.0",
            "--generation-config", "vllm",  # 使用vLLM的生成配置而不是模型自带的
            "--enforce-eager"  # 禁用torch.compile和CUDA graph，避免V1 Engine编译问题
        ]

        # 如果设置了max_model_len，添加到启动命令
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"vllm_{timestamp}.log"
        self.log_file = open(log_path, 'w', encoding='utf-8')
        logger.info(f"vLLM日志将保存至: {log_path}")

        logger.info(f"启动vLLM服务: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT  # 将stderr也重定向到同一个日志文件
        )

        # 等待服务启动
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._check_service():
                logger.info("vLLM服务启动成功")
                return True
            time.sleep(5)

        logger.error("vLLM服务启动超时")
        return False

    def stop(self):
        """停止vLLM服务"""
        if self.process:
            self.process.terminate()
            logger.info("vLLM服务已停止")
        if self.log_file:
            self.log_file.close()
            logger.info("vLLM日志文件已关闭")

    def _check_service(self) -> bool:
        """检查vLLM服务是否可用，包括模型是否已加载"""
        try:
            # 检查服务健康状态
            health_response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            if health_response.status_code != 200:
                return False

            # 检查模型是否已注册（确保模型加载完成）
            models_response = requests.get(
                f"{self.api_url}/v1/models",
                timeout=5
            )
            if models_response.status_code != 200:
                return False

            models_data = models_response.json()
            model_ids = [m.get('id') for m in models_data.get('data', [])]
            if self.model_name not in model_ids:
                logger.debug(f"模型 {self.model_name} 尚未加载，当前可用模型: {model_ids}")
                return False

            # 额外检查：发送一个实际的测试请求，确保模型真正可用（CUDA图编译完成）
            # 这是必要的，因为vLLM启动后需要时间来编译CUDA图
            test_response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1
                },
                timeout=30
            )
            if test_response.status_code != 200:
                logger.debug(f"模型测试请求失败，状态码: {test_response.status_code}, 响应: {test_response.text[:200]}")
                return False

            return True
        except Exception as e:
            logger.debug(f"服务检查异常: {str(e)}")
            return False


class BenchmarkEvaluator:
    """benchmark测评器"""

    def __init__(self, config_file: str, run_id: int = None, version: str = None, vllm_manager: Optional['VLLMServiceManager'] = None):
        self.config_file = config_file
        self.config = self._load_config()
        self.vllm_manager = vllm_manager
        self.results = {}
        self.base_dir = Path(self.config['model_path'])
        self.version = version or datetime.now().strftime("version_%Y%m%d_%H%M%S")

        # 支持多次运行，每次运行有独立的输出目录
        if run_id is not None:
            self.output_dir = self.base_dir / "evalscope" / self.version / f"run_{run_id}"
            self.run_id = run_id
        else:
            self.output_dir = self.base_dir / "evalscope" / self.version
            self.run_id = None
            
        self.debug_mode = self.config.get('debug', False)  # 支持debug模式

        # 提取模型名称用于日志显示
        path_parts = self.config['model_path'].split('/')
        if 'checkpoint' in path_parts[-1].lower() and len(path_parts) > 1:
            self.model_name = path_parts[-2]
        else:
            self.model_name = path_parts[-1]

    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_task_config(self, benchmark_name: str) -> TaskConfig:
        """创建TaskConfig配置"""
        benchmark_config = self.config['benchmarks'][benchmark_name]

        logger.info(f"开始创建 benchmark {benchmark_name} 的 TaskConfig")
        logger.info(f"benchmark 配置内容: {benchmark_config}")

        # 读取 benchmark 级别的 judge_strategy（默认 rule）
        benchmark_judge_strategy = benchmark_config.get('judge_strategy', JudgeStrategy.RULE)
        use_llm = 'llm' in str(benchmark_judge_strategy).lower()
        logger.info(f"[{benchmark_name}] judge_strategy: {benchmark_judge_strategy}")

        task_config = TaskConfig()

        # 设置模型信息 - 与VLLMServiceManager使用相同的模型名称提取逻辑
        path_parts = self.config['model_path'].split('/')
        if 'checkpoint' in path_parts[-1].lower() and len(path_parts) > 1:
            model_name = path_parts[-2]
        else:
            model_name = path_parts[-1]

        task_config.model = model_name  # 设置实际模型名称
        task_config.model_id = model_name
        task_config.eval_type = 'openai_api'  # 改为正确的API类型，支持通过HTTP API调用模型
        task_config.eval_backend = EvalBackend.NATIVE  # 保持使用原生的评估后端
        task_config.api_url = 'http://localhost:8000/v1'  # 设置API URL
        task_config.api_key = 'EMPTY'  # 设置API密钥

        # 设置数据集
        task_config.datasets = [benchmark_config['dataset_name']]

        # 设置重复次数
        task_config.repeats = benchmark_config.get('repeats', 1)

        # 设置数据集参数
        dataset_args = {
            'dataset_name': benchmark_config.get('subset', None),
            'few_shot_num': benchmark_config.get('few_shot_num', 0)
        }
        task_config.dataset_args = {benchmark_config['dataset_name']: dataset_args}

        # 设置模型API参数 - 使用 GenerateConfig 对象而不是字典
        generation_config_dict = {
            'temperature': benchmark_config.get('temperature', 0.0),
            'top_p': benchmark_config.get('top_p', 1.0),
            'top_k': benchmark_config.get('top_k', 50),
            'max_tokens': benchmark_config.get('max_tokens', 2048),
            'repetition_penalty': benchmark_config.get('repetition_penalty', 1.0)
        }
        logger.info(f"[{benchmark_name}] 生成配置 (从benchmark读取): {generation_config_dict}")

        # 添加额外的gen_config配置
        gen_config = benchmark_config.get('gen_config', {})
        if gen_config:
            logger.info(f"[{benchmark_name}] 额外的gen_config配置: {gen_config}")
            generation_config_dict.update(gen_config)

        # 创建 GenerateConfig 对象
        task_config.generation_config = GenerateConfig.model_validate(generation_config_dict)
        logger.info(f"[{benchmark_name}] 最终GenerateConfig对象: {task_config.generation_config}")

        # 详细打印GenerateConfig的属性
        if hasattr(task_config.generation_config, '__dict__'):
            logger.info(f"[{benchmark_name}] GenerateConfig详细属性: {task_config.generation_config.__dict__}")
        if hasattr(task_config.generation_config, 'max_tokens'):
            logger.info(f"[{benchmark_name}] GenerateConfig.max_tokens = {task_config.generation_config.max_tokens}")

        # 设置工作目录
        task_config.work_dir = str(self.output_dir / benchmark_name)

        # 设置并发数
        parallel = self.config.get('eval', {}).get('parallel', 1)
        task_config.eval_batch_size = parallel

        # 设置sandbox配置
        if 'use_sandbox' in benchmark_config:
            task_config.use_sandbox = benchmark_config['use_sandbox']
            logger.info(f"[{benchmark_name}] use_sandbox设置为: {task_config.use_sandbox}")

            # 设置sandbox worker数量 (docker容器数量)
            sandbox_workers = benchmark_config.get('sandbox_workers', 5)
            task_config.judge_worker_num = sandbox_workers
            logger.info(f"[{benchmark_name}] judge_worker_num设置为: {sandbox_workers}")

            # 设置沙箱工作目录 (避免临时文件打爆主机存储)
            sandbox_config = self.config.get('sandbox', {})
            if sandbox_config.get('work_dir'):
                sandbox_work_dir = sandbox_config['work_dir']
                # 确保目录存在
                Path(sandbox_work_dir).mkdir(parents=True, exist_ok=True)
                # 配置沙箱管理器使用指定的工作目录
                task_config.sandbox_manager_config = {
                    'work_dir': sandbox_work_dir
                }
                logger.info(f"[{benchmark_name}] sandbox工作目录设置为: {sandbox_work_dir}")

        # 设置 judge_strategy
        task_config.judge_strategy = benchmark_judge_strategy

        # 若使用 LLM judge，从全局配置读取 API_KEY / BASE_URL 并构建 judge_model_args
        if use_llm:
            api_key = self.config.get('API_KEY', '')
            base_url = self.config.get('BASE_URL', '').replace('/chat/completions', '').rstrip('/')
            judge_model = benchmark_config.get('judge_model', 'Qwen/Qwen3-8B')
            if not api_key or not base_url:
                raise ValueError(
                    f"[{benchmark_name}] 使用 LLM judge 需要在配置文件顶层设置 API_KEY 和 BASE_URL"
                )
            task_config.judge_model_args = {
                'api_key': api_key,
                'api_url': base_url,
                'model_id': judge_model,
                'generation_config': {'temperature': 0.0, 'max_tokens': 1024},
            }
            # 限制 LLM judge 并发，避免触发 TPM/RPM 限制，默认串行 (1)
            judge_parallel = benchmark_config.get('judge_parallel', 1)
            task_config.judge_worker_num = judge_parallel
            logger.info(f"[{benchmark_name}] LLM judge 已配置: model={judge_model}, url={base_url}, judge_parallel={judge_parallel}")

        # 打印最终的TaskConfig配置摘要
        logger.info(f"[{benchmark_name}] TaskConfig创建完成:")
        logger.info(f"  - model: {task_config.model}")
        logger.info(f"  - datasets: {task_config.datasets}")
        logger.info(f"  - eval_type: {task_config.eval_type}")
        logger.info(f"  - api_url: {task_config.api_url}")
        logger.info(f"  - generation_config: {task_config.generation_config}")
        logger.info(f"  - work_dir: {task_config.work_dir}")

        return task_config

    def _save_config(self):
        """保存配置文件"""
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
        logger.info(f"调试模式: {self.debug_mode}")
        logger.info(f"{'='*60}\n")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        # 从vllm配置中获取max_model_len
        max_model_len = self.config.get('vllm', {}).get('max_model_len', None)
        if max_model_len:
            logger.info(f"vLLM max_model_len 设置为: {max_model_len}")

        manage_vllm = False
        if self.vllm_manager is None:
            # 启动vLLM服务
            self.vllm_manager = VLLMServiceManager(
                model_path=self.config['model_path'],
                tp=self.config.get('vllm', {}).get('tp', 1),
                gpu_memory_utilization=self.config.get('vllm', {}).get('gpu_memory_utilization', 0.9),
                max_model_len=max_model_len
            )
            manage_vllm = True

            if not self.vllm_manager.start():
                logger.error("vLLM服务启动失败，评测终止")
                return

        try:
            # 运行所有benchmark - 先检查哪些benchmarks可以使用规则验证
            valid_benchmarks = []
            skipped_benchmarks = []

            for benchmark_name in self.config['benchmarks'].keys():
                try:
                    self._create_task_config(benchmark_name)
                    valid_benchmarks.append(benchmark_name)
                except ValueError as e:
                    logger.warning(f"跳过 benchmark {benchmark_name}: {str(e)}")
                    skipped_benchmarks.append(benchmark_name)
                    continue

            if not valid_benchmarks:
                logger.error("没有有效的benchmark可以评估")
                return

            logger.info(f"有效benchmarks: {valid_benchmarks}")
            if skipped_benchmarks:
                logger.info(f"已跳过的benchmarks: {skipped_benchmarks}")

            total_benchmarks = len(valid_benchmarks)

            for idx, benchmark_name in enumerate(valid_benchmarks):
                # 外部进度显示
                repeats = self.config['benchmarks'][benchmark_name].get('repeats', 1)
                remaining_benchmarks = total_benchmarks - idx - 1
                logger.info(f"\n{'='*50}")
                logger.info(f"正在评估 {benchmark_name} (repeat: {repeats}次) (剩余: {remaining_benchmarks}个benchmark)")
                logger.info(f"数据集: {self.config['benchmarks'][benchmark_name].get('dataset_name', 'unknown')}")
                logger.info(f"子集: {self.config['benchmarks'][benchmark_name].get('subset', 'default')}")
                logger.info(f"模型: {self.model_name if hasattr(self, 'model_name') else 'unknown'}")

                try:
                    logger.info(f"开始创建 {benchmark_name} 的 TaskConfig...")
                    task_config = self._create_task_config(benchmark_name)
                    logger.info(f"TaskConfig 创建成功: {task_config.model} {task_config.datasets}")
                except Exception as e:
                    logger.error(f"创建 TaskConfig 失败: {str(e)}")
                    raise

                # 使用tqdm显示内部进度
                logger.info(f"开始评测 {benchmark_name}...")
                result = run_task_with_progress(task_config, benchmark_name)

                # 保存结果
                self.results[benchmark_name] = result

                logger.info(f"完成评测 {benchmark_name} ({idx+1}/{total_benchmarks})")

        except Exception as e:
            logger.error(f"评测过程中出错: {str(e)}")
        finally:
            if manage_vllm and self.vllm_manager:
                # 停止vLLM服务
                self.vllm_manager.stop()

    def _convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        # 处理Report等自定义对象
        if hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
        if hasattr(obj, 'to_dict'):
            return self._convert_to_serializable(obj.to_dict())
        # 其他类型转为字符串
        return str(obj)

    def save_results(self):
        """保存评测结果"""
        # 创建结果汇总文件
        results_summary = self._prepare_results_summary()

        # 保存Excel汇总结果
        excel_path = self.output_dir / "summary.xlsx"
        results_summary.to_excel(excel_path, index=False)
        logger.info(f"Excel汇总结果已保存至: {excel_path}")

        # 保存JSON格式的详细结果 - 先转换为可序列化格式
        json_path = self.output_dir / "detailed_results.json"
        serializable_results = self._convert_to_serializable(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存至: {json_path}")

    def _prepare_results_summary(self) -> pd.DataFrame:
        """准备结果汇总"""
        summary_data = {'benchmark': [], 'dataset': [], 'score': []}

        # 先将结果转换为可序列化的字典格式，以便正确提取分数
        serializable_results = self._convert_to_serializable(self.results)

        for benchmark_name, result in serializable_results.items():
            # run_task 返回的结果格式是 {dataset_name: {score: xxx, ...}}
            # 需要遍历内层字典来提取分数
            if isinstance(result, dict):
                for dataset_name, dataset_result in result.items():
                    if isinstance(dataset_result, dict):
                        # 从嵌套字典中提取分数
                        if 'score' in dataset_result:
                            score = dataset_result['score']
                        elif 'accuracy' in dataset_result:
                            score = dataset_result['accuracy']
                        else:
                            score = 'N/A'
                    else:
                        score = 'N/A'

                    summary_data['benchmark'].append(benchmark_name)
                    summary_data['dataset'].append(dataset_name)
                    summary_data['score'].append(score)
            else:
                # 兜底处理
                summary_data['benchmark'].append(benchmark_name)
                summary_data['dataset'].append('unknown')
                summary_data['score'].append('N/A')

        return pd.DataFrame(summary_data)


class MultiRunEvaluator:
    """多次运行评测器，用于统计方差"""
    
    def __init__(self, config_file: str, num_runs: int = 1):
        self.config_file = config_file
        self.num_runs = num_runs
        self.version = datetime.now().strftime("version_%Y%m%d_%H%M%S")
        
        # 加载配置获取模型路径
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_dir = Path(self.config['model_path'])
        self.output_dir = self.base_dir / "evalscope" / self.version
        self.all_results = []  # 存储所有运行的结果
        
    def run_all(self):
        """运行多次评测"""
        logger.info(f"\n{'='*70}")
        logger.info(f"开始多次运行评测 - 共 {self.num_runs} 次")
        logger.info(f"总输出目录: {self.output_dir}")
        logger.info(f"{'='*70}\n")

        # 初始化并启动vLLM服务
        max_model_len = self.config.get('vllm', {}).get('max_model_len', None)
        vllm_manager = VLLMServiceManager(
            model_path=self.config['model_path'],
            tp=self.config.get('vllm', {}).get('tp', 1),
            gpu_memory_utilization=self.config.get('vllm', {}).get('gpu_memory_utilization', 0.9),
            max_model_len=max_model_len
        )

        try:
            if not vllm_manager.start():
                logger.error("vLLM服务启动失败，评测终止")
                return
            for run_id in range(1, self.num_runs + 1):
                logger.info(f"\n{'#'*70}")
                logger.info(f"# 第 {run_id}/{self.num_runs} 次运行")
                logger.info(f"{'#'*70}\n")

                evaluator = BenchmarkEvaluator(self.config_file, run_id=run_id, version=self.version, vllm_manager=vllm_manager)
                evaluator.run_evaluation()
                evaluator.save_results()

                # 保存结果用于后续统计
                self.all_results.append({
                    'run_id': run_id,
                    'results': evaluator.results,
                    'output_dir': evaluator.output_dir
                })

                logger.info(f"第 {run_id} 次运行完成")

                # 如果不是最后一次运行，等待一段时间再开始下一次
                if run_id < self.num_runs:
                    wait_time = 5
                    logger.info(f"等待 {wait_time} 秒后开始下一次运行...")
                    time.sleep(wait_time)

            # 汇总所有结果
            self._summarize_results()

        finally:
            # 停止vLLM服务
            vllm_manager.stop()
        
    def _summarize_results(self):
        """汇总多次运行的结果并计算统计信息"""
        logger.info(f"\n{'='*70}")
        logger.info("汇总多次运行结果")
        logger.info(f"{'='*70}\n")
        
        # 收集每个benchmark在所有运行中的分数
        benchmark_scores = {}
        
        for run_data in self.all_results:
            run_id = run_data['run_id']
            results = run_data['results']
            
            for benchmark_name, result in results.items():
                if benchmark_name not in benchmark_scores:
                    benchmark_scores[benchmark_name] = {}
                    
                # 解析结果获取分数
                if isinstance(result, dict):
                    for dataset_name, dataset_result in result.items():
                        key = f"{benchmark_name}/{dataset_name}"
                        if key not in benchmark_scores[benchmark_name]:
                            benchmark_scores[benchmark_name][key] = {
                                'scores': [],
                                'dataset_name': dataset_name
                            }
                        
                        if isinstance(dataset_result, dict):
                            if 'score' in dataset_result:
                                score = dataset_result['score']
                            elif 'accuracy' in dataset_result:
                                score = dataset_result['accuracy']
                            else:
                                score = None
                        else:
                            score = None
                            
                        if score is not None:
                            benchmark_scores[benchmark_name][key]['scores'].append(score)
        
        # 计算统计信息
        import numpy as np
        
        summary_rows = []
        for benchmark_name, datasets in benchmark_scores.items():
            for key, data in datasets.items():
                scores = data['scores']
                dataset_name = data['dataset_name']
                
                if len(scores) > 0:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                    min_score = np.min(scores)
                    max_score = np.max(scores)
                    
                    row = {
                        'benchmark': benchmark_name,
                        'dataset': dataset_name,
                        'run_count': len(scores),
                        'mean': round(mean_score, 4),
                        'std': round(std_score, 4),
                        'std_percent': round(std_score / mean_score * 100, 2) if mean_score > 0 else 0,
                        'min': round(min_score, 4),
                        'max': round(max_score, 4),
                        'range': round(max_score - min_score, 4)
                    }
                    
                    # 添加每次运行的详细分数
                    for i, score in enumerate(scores, 1):
                        row[f'run_{i}'] = round(score, 4)
                        
                    summary_rows.append(row)
        
        # 创建汇总DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        # 保存Excel汇总
        excel_path = self.output_dir / "multi_run_summary.xlsx"
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_excel(excel_path, index=False)
        logger.info(f"多次运行汇总结果已保存至: {excel_path}")
        
        # 保存JSON格式的详细结果
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
        
        # 打印汇总表格
        logger.info("\n多次运行结果汇总:")
        logger.info("-" * 100)
        for _, row in summary_df.iterrows():
            logger.info(f"{row['benchmark']}/{row['dataset']}: "
                       f"均值={row['mean']:.4f}, 标准差={row['std']:.4f} ({row['std_percent']:.2f}%), "
                       f"范围=[{row['min']:.4f}, {row['max']:.4f}]")
        logger.info("-" * 100)
        
        return summary_df


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python eval.py <config_file> [num_runs]")
        print("例如: python eval.py eval_configs/example.yaml")
        print("例如: python eval.py eval_configs/example.yaml 5  # 运行5次")
        sys.exit(1)

    config_file = sys.argv[1]
    
    # 获取运行次数，默认为1
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
        # 多次运行模式
        multi_evaluator = MultiRunEvaluator(config_file, num_runs=num_runs)
        multi_evaluator.run_all()
    else:
        # 单次运行模式
        evaluator = BenchmarkEvaluator(config_file)
        evaluator.run_evaluation()
        evaluator.save_results()

    logger.info("评测完成")


if __name__ == "__main__":
    main()