#!/usr/bin/env python3
"""
简化版数据集改写脚本

功能：
1. 读取配置，转换改写模型格式
2. 读取原始数据集，构建prompt
3. 调用改写模型生成n_rewrites次结果（支持断点续传）
4. 自动打乱结果
5. 生成训练集格式

不包含：
- similarity 计算
- student_model 计算
- fsf_model 计算
"""

import argparse
import json
import os
import re
import sys
import yaml
import gc
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# 添加项目路径
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo')

import torch
import ray
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# =============================================================================
# 配置相关
# =============================================================================

@dataclass
class SimpleRewriteConfig:
    """简化改写任务配置"""
    # 模型路径
    rewrite_model_path: str

    # 数据路径
    input_data_path: str
    output_dir: str

    # 改写参数
    n_rewrites: int = 4  # 每个样本调用n次改写模型

    # 打乱参数
    shuffle_rewrites: bool = True  # 是否打乱改写结果
    random_seed: int = 42  # 随机种子

    # 失败处理参数
    use_original_on_fail: bool = True  # 改写失败时使用原始CoT还是放弃该样本
    use_rewrite_output_as_fallback: bool = False  # 当rewrite_reasoning为null时，尝试从rewrite_output提取标签内容（仅在use_original_on_fail=False时生效）

    # 模型生成参数
    max_new_tokens: int = 15000
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # 批处理参数
    process_batch_size: int = 32  # 每处理这么多样本后保存一次 ckpt

    # vLLM 参数
    vllm_tensor_parallel_size: int = 2
    vllm_gpu_memory_utilization: float = 0.85
    vllm_micro_batch_size: int = 1

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SimpleRewriteConfig':
        return cls(**config_dict)


def load_config(config_path: str) -> SimpleRewriteConfig:
    """从yaml文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = SimpleRewriteConfig.from_dict(config_dict)

    # 参数互斥验证
    if config.use_rewrite_output_as_fallback and config.use_original_on_fail:
        raise ValueError(
            "use_rewrite_output_as_fallback=True cannot be used together with use_original_on_fail=True. "
            "use_rewrite_output_as_fallback only works when use_original_on_fail=False."
        )

    return config


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


def get_checkpoint_path(output_dir: str) -> str:
    """获取检查点文件路径"""
    return os.path.join(output_dir, 'ckpt', f'rewrites.jsonl')


def get_progress_path(output_dir: str) -> str:
    """获取进度文件路径"""
    return os.path.join(output_dir, 'ckpt', f'progress.json')


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



# =============================================================================
# Step 0: Model Format Conversion
# =============================================================================

def _is_hf_format(path: str) -> bool:
    """
    检查给定路径是否是 HuggingFace 格式（包含必要的模型文件）
    
    检查项：
    - config.json
    - tokenizer 文件 (tokenizer.json 或 tokenizer_config.json)
    - 模型权重文件 (*.bin, *.safetensors)
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    
    files = os.listdir(path)
    has_config = 'config.json' in files
    has_tokenizer = 'tokenizer.json' in files or 'tokenizer_config.json' in files
    has_model = any(f.endswith('.bin') or f.endswith('.safetensors') for f in files)
    
    return has_config and has_tokenizer and has_model


def convert_model_to_hf(model_path: str) -> str:
    """
    将FSDP checkpoint转换为Huggingface格式 (使用VERL的model_merger库)
    
    检测逻辑（按优先级）：
    1. 如果 model_path 本身就是 HF 格式，直接返回
    2. 检查 model_path/hf/ 是否存在且为 HF 格式
    3. 如果是 FSDP 格式，则进行转换
    """
    # 1. 首先检查传入的路径本身是否是 HF 格式
    if _is_hf_format(model_path):
        print(f"  [Model Convert] Input path is already HF format: {model_path}")
        return model_path
    
    # 2. 检查是否已存在转换后的模型（hf/ 子目录）
    hf_path = os.path.join(model_path, 'hf')
    if _is_hf_format(hf_path):
        print(f"  [Model Convert] HF format found at {hf_path}")
        return hf_path

    print(f"  [Model Convert] Converting FSDP checkpoint to HF format...")
    print(f"    Source: {model_path}")
    print(f"    Target: {hf_path}")

    try:
        os.makedirs(hf_path, exist_ok=True)

        # 使用 VERL 的 FSDPModelMerger
        actor_dir = os.path.join(model_path, 'actor')
        hf_config_path = os.path.join(actor_dir, 'huggingface')

        fsdp_config_path = os.path.join(actor_dir, 'fsdp_config.json')
        if not os.path.exists(fsdp_config_path):
            raise FileNotFoundError(f"FSDP config not found: {fsdp_config_path}")

        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=actor_dir,
            hf_model_config_path=hf_config_path,
            target_dir=hf_path,
            trust_remote_code=False,
        )

        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        merger.cleanup()

        print(f"  [Model Convert] Conversion completed: {hf_path}")
        return hf_path

    except Exception as e:
        import traceback
        print(f"  [Model Convert] Error during conversion: {e}")
        print(traceback.format_exc())
        raise


# =============================================================================
# Ray 数据并行改写客户端
# =============================================================================

class vLLMWorkerActor:
    """Ray Actor: 每个 actor 运行一个 vLLM 实例"""

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

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.llm_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )

        self.system_prompt = """\
You are an expert in math and code. The 'Original Reasoning Process' is a verified and valid derivation. Using it as a reference, rewrite the reasoning with a better logical structure and a more clear delivery.

Follow these guidelines:
1. **Logical Optimization**: Keep logic direct and coherent; eliminate redundancy without losing necessary complexity.
2. **Step-by-Step Delivery**: Present the reasoning in a clear, sequential, step-by-step manner.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags. Output nothing else."""
#         self.system_prompt = """\
# You are an expert in math and code. The 'Original Reasoning Process' is a verified and valid derivation. Using it as a reference, rewrite the reasoning with a better logical structure and a more clear delivery.
# Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags. Output nothing else."""

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
        """
        if not tasks:
            return []

        from vllm import SamplingParams

        sp_params = SamplingParams(**sampling_params_dict)

        results = {}
        retry_counts = {task['task_id']: 0 for task in tasks}

        prompts = [self._build_prompt(task) for task in tasks]

        try:
            outputs = self.llm_engine.generate(prompts, sp_params)

            for task, output in zip(tasks, outputs):
                generated_text = output.outputs[0].text if output.outputs else ""
                rewrite_reasoning = self._extract_rewrite_reasoning(generated_text)

                results[task['task_id']] = {
                    'task_id': task['task_id'],
                    'sample_id': task['sample_id'],
                    'question': task['question'],
                    'original_thought': task['thought'],
                    'answer': task.get('answer', ''),
                    'rewrite_output': generated_text,
                    'rewrite_reasoning': rewrite_reasoning,
                    'success': rewrite_reasoning is not None
                }

        except Exception as e:
            for task in tasks:
                results[task['task_id']] = {
                    'task_id': task['task_id'],
                    'sample_id': task['sample_id'],
                    'question': task['question'],
                    'original_thought': task['thought'],
                    'answer': task.get('answer', ''),
                    'rewrite_output': str(e),
                    'rewrite_reasoning': None,
                    'success': False,
                    'error': str(e)
                }
            return list(results.values())

        # 重试逻辑
        max_retries = 3

        for retry_round in range(max_retries):
            tasks_to_retry = [
                task for task in tasks
                if not results[task['task_id']]['success']
                and retry_counts[task['task_id']] < max_retries
            ]

            if not tasks_to_retry:
                break

            for task in tasks_to_retry:
                retry_counts[task['task_id']] += 1

            print(f"[vLLMWorkerActor {self.worker_id}] Retry round {retry_round + 1}/{max_retries} for {len(tasks_to_retry)} failed tasks")

            retry_prompts = [self._build_prompt(task) for task in tasks_to_retry]

            try:
                retry_outputs = self.llm_engine.generate(retry_prompts, sp_params)

                for task, output in zip(tasks_to_retry, retry_outputs):
                    generated_text = output.outputs[0].text if output.outputs else ""
                    rewrite_reasoning = self._extract_rewrite_reasoning(generated_text)

                    if rewrite_reasoning is not None:
                        results[task['task_id']] = {
                            'task_id': task['task_id'],
                            'sample_id': task['sample_id'],
                            'question': task['question'],
                            'original_thought': task['thought'],
                            'answer': task.get('answer', ''),
                            'rewrite_output': generated_text,
                            'rewrite_reasoning': rewrite_reasoning,
                            'success': True,
                            'retried': retry_counts[task['task_id']]
                        }

            except Exception as e:
                for task in tasks_to_retry:
                    results[task['task_id']]['error'] = str(e)
                    results[task['task_id']]['retried'] = retry_counts[task['task_id']]

        return list(results.values())

    def ping(self) -> str:
        return f"Worker {self.worker_id} is alive"

    def shutdown(self):
        if hasattr(self, 'llm_engine'):
            del self.llm_engine
        print(f"[vLLMWorkerActor {self.worker_id}] Shutdown complete")


class RayRewriteClient:
    """Ray 改写客户端 - 使用 Ray Actor 实现数据并行"""

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
        self.model_path = model_path
        self.num_instances = num_instances
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.total_gpus = total_gpus
        self.micro_batch_size = micro_batch_size
        self.max_model_len = max_model_len

        if num_instances * tensor_parallel_size > total_gpus:
            raise ValueError(
                f"num_instances ({num_instances}) * tensor_parallel_size ({tensor_parallel_size}) "
                f"cannot exceed total_gpus ({total_gpus})"
            )

        self.actors = []
        self.is_initialized = False

        self.actor_class = ray.remote(num_gpus=tensor_parallel_size)(vLLMWorkerActor)

    def initialize(self):
        if self.is_initialized:
            return

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        print(f"\n  [RayRewriteClient] Initializing...")
        print(f"    Total GPUs: {self.total_gpus}")
        print(f"    Number of actors: {self.num_instances}")
        print(f"    GPUs per actor (TP): {self.tensor_parallel_size}")
        print(f"    Micro batch size: {self.micro_batch_size}")

        for i in range(self.num_instances):
            actor = self.actor_class.remote(
                worker_id=i,
                model_path=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            self.actors.append(actor)

        print(f"  [RayRewriteClient] Waiting for actors to initialize...")
        for i, actor in enumerate(self.actors):
            ray.get(actor.ping.remote())

        self.is_initialized = True
        print(f"  [RayRewriteClient] All {self.num_instances} actors ready")

    def process_samples(
        self,
        samples: List[Dict],
        id_to_sample: Dict[str, Dict],
        n_rewrites: int,
        sampling_params_dict: Dict[str, Any],
        ckpt_path: Optional[str] = None,
        progress_path: Optional[str] = None,
        process_batch_size: int = 32,
        is_resuming: bool = False,
    ) -> List[Dict]:
        """处理多个样本，每个样本生成 n_rewrites 次改写

        使用动态负载均衡：快的 actor 会自动获取更多任务，而不是静态分配
        """
        self.initialize()

        # 准备所有任务 - 打乱顺序以更好地实现负载均衡
        import random
        all_tasks = []
        task_id = 0
        for sample in samples:
            sample_id = sample['id']
            for _ in range(n_rewrites):
                all_tasks.append({
                    'task_id': task_id,
                    'sample_id': sample_id,
                    'question': sample['question'],
                    'thought': sample['thought'],
                    'answer': sample.get('answer', ''),
                })
                task_id += 1

        # 打乱任务顺序，使不同样本的任务均匀分布
        random.shuffle(all_tasks)

        total_tasks = len(all_tasks)
        num_actors = len(self.actors)

        # 按 micro_batch_size 分批 - 这是共享的任务队列
        task_batches = []
        for i in range(0, len(all_tasks), self.micro_batch_size):
            batch = all_tasks[i:i + self.micro_batch_size]
            task_batches.append(batch)

        print(f"  [RayRewriteClient] Dynamic load balancing: {total_tasks} tasks in {len(task_batches)} batches across {num_actors} actors")

        # 收集结果
        results = {s['id']: [] for s in samples}
        sample_task_counts = {s['id']: 0 for s in samples}
        completed_ids = []
        write_batch = []

        # 共享任务队列的索引 - 使用列表使其在闭包中可变
        task_queue_index = [0]

        # 使用异步方式提交任务并收集结果 - 动态负载均衡版本
        pending_refs = []  # (actor_idx, batch_idx, ref)

        # 初始提交：每个 actor 获取一个批次
        for actor_idx in range(num_actors):
            batch_idx = task_queue_index[0]
            if batch_idx < len(task_batches):
                task_queue_index[0] += 1
                ref = self.actors[actor_idx].process_tasks.remote(
                    task_batches[batch_idx], sampling_params_dict
                )
                pending_refs.append((actor_idx, batch_idx, ref))

        with tqdm(total=total_tasks, desc="Rewrite", unit="task", dynamic_ncols=True) as pbar:
            while pending_refs:
                pending_object_refs = [r for _, _, r in pending_refs]
                ready_refs, _ = ray.wait(pending_object_refs, num_returns=1)

                for ready_ref in ready_refs:
                    found_idx = None
                    for i, (actor_idx, batch_idx, ref) in enumerate(pending_refs):
                        if ref == ready_ref:
                            found_idx = i
                            break

                    if found_idx is not None:
                        actor_idx, batch_idx, _ = pending_refs.pop(found_idx)
                    else:
                        continue

                    try:
                        batch_results = ray.get(ready_ref)

                        for result in batch_results:
                            sample_id = result['sample_id']
                            results[sample_id].append(result)
                            sample_task_counts[sample_id] += 1
                            pbar.update(1)

                            if sample_task_counts[sample_id] == n_rewrites:
                                write_batch.append(sample_id)
                                completed_ids.append(sample_id)

                        # 动态获取下一批 - 从共享队列中取任务
                        next_batch_idx = task_queue_index[0]
                        if next_batch_idx < len(task_batches):
                            task_queue_index[0] += 1
                            next_ref = self.actors[actor_idx].process_tasks.remote(
                                task_batches[next_batch_idx], sampling_params_dict
                            )
                            pending_refs.append((actor_idx, next_batch_idx, next_ref))

                        completed_tasks = sum(sample_task_counts.values())
                        active_actors = len(pending_refs)
                        pbar.set_postfix({
                            'completed': f'{completed_tasks}/{total_tasks}',
                            'samples': f'{len(completed_ids)}/{len(samples)}',
                            'active': f'{active_actors}/{num_actors}'
                        })

                        if len(write_batch) >= process_batch_size or completed_tasks == total_tasks:
                            batch_write_results = []
                            for sample_id in sorted(write_batch):
                                sample = id_to_sample[sample_id]
                                rewrite_results = []
                                for r in results.get(sample_id, []):
                                    rewrite_results.append({
                                        'rewrite_output': r['rewrite_output'],
                                        'rewrite_reasoning': r['rewrite_reasoning'],
                                        'success': r['success']
                                    })

                                result_item = {
                                    'id': sample_id,
                                    'line_id': sample.get('line_id', -1),
                                    'question': sample['question'],
                                    'original_thought': sample['thought'],
                                    'answer': sample.get('answer', ''),
                                    'rewrite_results': rewrite_results
                                }
                                batch_write_results.append(result_item)

                            if ckpt_path:
                                save_jsonl(batch_write_results, ckpt_path, append=is_resuming)
                                is_resuming = True

                            if progress_path:
                                with open(progress_path, 'w') as f:
                                    json.dump({'completed_ids': completed_ids}, f)

                            pbar.write(f"  Saved batch of {len(write_batch)} samples ({len(completed_ids)}/{len(samples)} completed)")
                            write_batch.clear()

                    except Exception as e:
                        print(f"  Error processing batch on actor {actor_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        # 尝试获取下一批继续处理
                        next_batch_idx = task_queue_index[0]
                        if next_batch_idx < len(task_batches):
                            task_queue_index[0] += 1
                            next_ref = self.actors[actor_idx].process_tasks.remote(
                                task_batches[next_batch_idx], sampling_params_dict
                            )
                            pending_refs.append((actor_idx, next_batch_idx, next_ref))

        # 整理最终结果
        final_results = []
        for sample in samples:
            sample_id = sample['id']
            rewrite_results = []
            for r in results.get(sample_id, []):
                rewrite_results.append({
                    'rewrite_output': r['rewrite_output'],
                    'rewrite_reasoning': r['rewrite_reasoning'],
                    'success': r['success']
                })

            result_item = {
                'id': sample_id,
                'line_id': sample.get('line_id', -1),
                'question': sample['question'],
                'original_thought': sample['thought'],
                'answer': sample.get('answer', ''),
                'rewrite_results': rewrite_results
            }
            final_results.append(result_item)

        return final_results

    def shutdown(self):
        if self.actors:
            shutdown_refs = [actor.shutdown.remote() for actor in self.actors]
            ray.get(shutdown_refs)
            self.actors = []
        self.is_initialized = False


# =============================================================================
# Step 1: Generate Rewrites
# =============================================================================

def step_generate_rewrites(
    config: SimpleRewriteConfig,
    samples: List[Dict],
    rewrite_client: RayRewriteClient,
) -> List[Dict]:
    """步骤1: 调用改写模型生成n次结果"""
    print("\n" + "=" * 60)
    print("Step 1: Rewrite Generation")
    print("=" * 60)

    ckpt_path = get_checkpoint_path(config.output_dir)
    progress_path = get_progress_path(config.output_dir)

    id_to_sample = {s['id']: s for s in samples}

    # 检查断点
    processed_ids = set()
    if os.path.exists(ckpt_path):
        existing_results = load_jsonl(ckpt_path)
        for item in existing_results:
            if 'id' in item:
                processed_ids.add(item['id'])
        print(f"  [Resume] Loaded {len(processed_ids)} existing results")

    samples_to_process = [s for s in samples if s['id'] not in processed_ids]

    total_samples = len(samples)
    remaining_samples = len(samples_to_process)

    print(f"  Total samples: {total_samples}")
    print(f"  Already processed: {len(processed_ids)}")
    print(f"  Remaining: {remaining_samples}")
    print(f"  N rewrites per sample: {config.n_rewrites}")

    if not samples_to_process:
        print("  No samples to process.")
        if processed_ids and os.path.exists(ckpt_path):
            return load_jsonl(ckpt_path)
        return []

    # 构建采样参数
    sampling_params = {
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.top_k > 0:
        sampling_params["top_k"] = config.top_k
    if config.presence_penalty != 0.0:
        sampling_params["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty != 0.0:
        sampling_params["frequency_penalty"] = config.frequency_penalty

    new_results = rewrite_client.process_samples(
        samples=samples_to_process,
        id_to_sample=id_to_sample,
        n_rewrites=config.n_rewrites,
        sampling_params_dict=sampling_params,
        ckpt_path=ckpt_path,
        progress_path=progress_path,
        process_batch_size=config.process_batch_size,
        is_resuming=len(processed_ids) > 0,
    )

    print(f"  Completed. Results saved to: {ckpt_path}")

    if os.path.exists(ckpt_path):
        return load_jsonl(ckpt_path)
    return new_results


# =============================================================================
# Step 2: Shuffle and Generate Dataset
# =============================================================================

def step_shuffle_and_generate(
    config: SimpleRewriteConfig,
    samples: List[Dict],
) -> Tuple[List[Dict], Dict]:
    """步骤2: 打乱改写结果并生成训练集格式"""
    print("\n" + "=" * 60)
    print("Step 2: Shuffle and Generate Dataset")
    print("=" * 60)

    import random

    if config.shuffle_rewrites:
        random.seed(config.random_seed)
        print(f"  Using random seed: {config.random_seed}")

    new_dataset = []
    stats = {
        'total_samples': len(samples),
        'total_rewrites': 0,
        'successful_rewrites': 0,
        'failed_rewrites': 0,
        'using_rewrite': 0,
        'using_original': 0,
        'dropped_samples': 0,
    }

    pbar = tqdm(samples, desc="Generate", unit="sample", dynamic_ncols=True)

    for sample in pbar:
        question = sample['question']
        original_thought = sample.get('original_thought') or sample.get('thought', '')
        answer = sample.get('answer', '')
        rewrite_results = sample.get('rewrite_results', [])

        stats['total_rewrites'] += len(rewrite_results)
        stats['successful_rewrites'] += sum(1 for r in rewrite_results if r.get('success'))
        stats['failed_rewrites'] += sum(1 for r in rewrite_results if not r.get('success'))

        # 找出所有成功的改写
        successful_rewrites = [r for r in rewrite_results if r.get('rewrite_reasoning')]

        if successful_rewrites:
            # 打乱并随机选择一个
            if config.shuffle_rewrites:
                random.shuffle(successful_rewrites)
            selected_rewrite = successful_rewrites[0]['rewrite_reasoning']
            stats['using_rewrite'] += 1
            using_original = False
        elif config.use_original_on_fail:
            # 没有成功的改写，使用原文
            selected_rewrite = original_thought
            stats['using_original'] += 1
            using_original = True
        elif config.use_rewrite_output_as_fallback:
            # 尝试从 rewrite_output 中提取标签内容作为 fallback
            selected_rewrite = None
            for r in rewrite_results:
                rewrite_output = r.get('rewrite_output', '')
                if '<rewrite_reasoning>' in rewrite_output and '</rewrite_reasoning>' in rewrite_output:
                    extracted = rewrite_output.split('<rewrite_reasoning>')[1].split('</rewrite_reasoning>')[0].strip()
                    if extracted:
                        selected_rewrite = extracted
                        break

            if selected_rewrite:
                stats['using_rewrite'] += 1
                using_original = False
            else:
                # 无法提取有效内容，放弃该样本
                stats['dropped_samples'] += 1
                continue
        else:
            # 没有成功的改写，放弃该样本
            stats['dropped_samples'] += 1
            continue

        # 构建 assistant content
        assistant_content = f"<think>\n{selected_rewrite.strip()}\n</think>\n\n{answer.strip()}"

        # 构建新的conversations格式
        new_item = {
            'conversations': [
                {
                    'role': 'user',
                    'content': question
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ],
            'metadata': {
                'sample_id': sample.get('id'),
                'line_id': sample.get('line_id'),
                'using_original': using_original,
                'original_thought': original_thought,
                'answer': answer,
                'n_successful_rewrites': len(successful_rewrites),
                'n_total_rewrites': len(rewrite_results),
            }
        }

        new_dataset.append(new_item)

        pbar.set_postfix({
            'rewrite': stats['using_rewrite'],
            'original': stats['using_original']
        })

    # 保存新数据集
    output_path = os.path.join(config.output_dir, 'final_dataset.jsonl')
    save_jsonl(new_dataset, output_path)

    # 保存统计信息
    stats_path = os.path.join(config.output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # 计算最终数据集大小
    final_dataset_size = len(new_dataset)
    
    print(f"\n  Dataset Statistics:")
    print(f"    Total samples: {stats['total_samples']}")
    print(f"    Total rewrites generated: {stats['total_rewrites']}")
    print(f"    Successful rewrites: {stats['successful_rewrites']}")
    print(f"    Failed rewrites: {stats['failed_rewrites']}")
    print(f"    Using rewrite: {stats['using_rewrite']} ({stats['using_rewrite']/stats['total_samples']*100:.1f}%)")
    print(f"    Using original: {stats['using_original']} ({stats['using_original']/stats['total_samples']*100:.1f}%)")
    if stats['dropped_samples'] > 0:
        print(f"    Dropped samples: {stats['dropped_samples']} ({stats['dropped_samples']/stats['total_samples']*100:.1f}%)")
    print(f"    Final dataset size: {final_dataset_size}")
    print(f"\n  Output saved to: {output_path}")
    print(f"  Stats saved to: {stats_path}")

    return new_dataset, stats


# =============================================================================
# Main
# =============================================================================

def load_and_parse_input_data(input_path: str) -> List[Dict]:
    """加载并解析输入数据"""
    print(f"\n  Loading input data from: {input_path}")

    data = load_jsonl(input_path)
    samples = []

    for line_id, item in enumerate(data):
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

        # 生成唯一 ID
        question_hash = hashlib.sha256(question.encode('utf-8')).hexdigest()[:8]
        unique_id = f"{line_id}-{question_hash}"

        samples.append({
            'id': unique_id,
            'line_id': line_id,
            'question': question,
            'thought': thought,
            'answer': answer,
        })

    print(f"  Loaded {len(samples)} valid samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Simple rewrite dataset generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable shuffling')
    parser.add_argument('--drop-on-fail', action='store_true', help='Drop sample when all rewrites fail (default: use original CoT)')
    args = parser.parse_args()

    # 加载配置
    print("=" * 60)
    print("Loading Configuration")
    print("=" * 60)

    config = load_config(args.config)
    print(f"  Config loaded from: {args.config}")

    # 命令行参数覆盖
    if args.no_shuffle:
        config.shuffle_rewrites = False
    if args.drop_on_fail:
        config.use_original_on_fail = False

    print(f"  Output directory: {config.output_dir}")
    print(f"  Shuffle rewrites: {config.shuffle_rewrites}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  N rewrites per sample: {config.n_rewrites}")
    print(f"  Use original on fail: {config.use_original_on_fail} (if False, drop sample)")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'ckpt'), exist_ok=True)

    # 保存配置副本
    config_copy_path = os.path.join(config.output_dir, 'config_used.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(vars(config), f)

    # 步骤0: 转换模型格式
    print("\n" + "=" * 60)
    print("Step 0: Model Format Conversion")
    print("=" * 60)
    rewrite_model_hf_path = convert_model_to_hf(config.rewrite_model_path)
    print(f"  Rewrite model (HF): {rewrite_model_hf_path}")

    # 初始化改写模型客户端
    total_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').split(','))
    vllm_num_instances = total_gpus // config.vllm_tensor_parallel_size

    print(f"\n  [GPU Allocation]")
    print(f"    Total GPUs: {total_gpus}")
    print(f"    vLLM TP size: {config.vllm_tensor_parallel_size}")
    print(f"    Ray actors: {vllm_num_instances}")
    print(f"    Micro batch size: {config.vllm_micro_batch_size}")

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

    # 步骤1: 生成改写
    samples = step_generate_rewrites(config, samples, rewrite_client)
    rewrite_client.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    # 步骤2: 打乱并生成最终数据集
    new_dataset, stats = step_shuffle_and_generate(config, samples)

    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
