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
import hashlib
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np

# 添加项目路径
sys.path.insert(0, '/data/luoys/projects/cdad/cot_rewriter_grpo')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试导入tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x  # fallback
    print("Warning: tqdm not installed, progress bars disabled")
    print("Install with: pip install tqdm")


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
    similarity_penalty_threshold: float = 0.9
    ppl_relative_improvement_threshold: float = 0.0
    fsf_threshold: float = 0.0
    
    # 模型生成参数
    max_new_tokens: int = 15000
    temperature: float = 0.6
    top_p: float = 0.95
    presence_penalty: float = 1.2
    frequency_penalty: float = 1.2
    
    # 多进程参数
    num_workers: int = 8
    batch_size: int = 32
    
    # SGLang参数
    sglang_tensor_parallel_size: int = 1
    sgland_micro_batch_size: int = 4
    
    # 可选：跳过某些步骤
    skip_rewrite: bool = False
    skip_similarity: bool = False
    skip_ppl: bool = False
    skip_fsf: bool = False


def load_config(config_path: str) -> RewriteConfig:
    """从yaml文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return RewriteConfig(**config_dict)


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
    将FSDP checkpoint转换为Huggingface格式
    
    输入: /path/to/model/global_step_100
    输出: /path/to/model/global_step_100/hf
    
    如果hf目录已存在且包含必要文件，则跳过转换
    """
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor.placement_types import Shard
    
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
    
    print(f"  [Model Convert] Converting FSDP checkpoint to HF format...")
    print(f"    Source: {model_path}")
    print(f"    Target: {hf_path}")
    
    try:
        os.makedirs(hf_path, exist_ok=True)
        
        # 1. 首先复制已有的huggingface目录下的文件（tokenizer配置等）
        hf_source = os.path.join(model_path, 'actor', 'huggingface')
        if os.path.exists(hf_source):
            print(f"    Copying tokenizer files from {hf_source}")
            for fname in os.listdir(hf_source):
                src = os.path.join(hf_source, fname)
                dst = os.path.join(hf_path, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        # 2. 加载FSDP模型权重并合并DTensor
        actor_dir = os.path.join(model_path, 'actor')
        
        # 找到所有model文件
        model_files = sorted([
            f for f in os.listdir(actor_dir) 
            if f.startswith('model_world_size_') and f.endswith('.pt')
        ])
        
        if not model_files:
            raise ValueError(f"No model files found in {actor_dir}")
        
        num_shards = len(model_files)
        print(f"    Found {num_shards} FSDP shard(s)")
        
        # 加载所有shard
        print(f"    Loading all FSDP shards...")
        all_shards = []
        for i, mf in enumerate(model_files):
            checkpoint_path = os.path.join(actor_dir, mf)
            shard = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            all_shards.append(shard)
            if (i + 1) % 4 == 0 or i == num_shards - 1:
                print(f"      Loaded {i + 1}/{num_shards} shards")
        
        # 合并DTensor
        print(f"    Merging FSDP DTensor weights...")
        state_dict = {}
        first_shard = all_shards[0]
        
        for key in first_shard.keys():
            value = first_shard[key]
            
            if isinstance(value, DTensor):
                # 获取DTensor的sharding信息
                placements = value.placements
                
                # 收集所有shard的本地tensor
                local_tensors = [all_shards[i][key].to_local() for i in range(num_shards)]
                
                # 根据placement类型进行合并
                if len(placements) == 1 and isinstance(placements[0], Shard):
                    shard_dim = placements[0].dim
                    # 沿着sharding维度拼接
                    full_tensor = torch.cat(local_tensors, dim=shard_dim)
                    state_dict[key] = full_tensor
                else:
                    # 其他placement类型，默认使用第0维拼接
                    full_tensor = torch.cat(local_tensors, dim=0)
                    state_dict[key] = full_tensor
            else:
                # 非DTensor，直接使用第一个shard的值
                state_dict[key] = value
        
        print(f"    Total parameters merged: {len(state_dict)}")
        
        # 3. 保存模型权重为safetensors格式
        print(f"    Saving model weights...")
        try:
            from safetensors.torch import save_file
            
            # 计算总大小
            total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
            print(f"    Total model size: {total_size / 1024**3:.2f} GB")
            
            # 分割成多个shard保存（每个约5GB）
            max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
            
            current_size = 0
            current_shard = {}
            shard_idx = 0
            all_shards_list = []
            
            for key, tensor in state_dict.items():
                tensor_size = tensor.numel() * tensor.element_size()
                
                if current_size + tensor_size > max_shard_size and current_shard:
                    # 保存当前shard
                    shard_filename = f"model-{shard_idx+1:05d}-of-{0:05d}.safetensors"
                    all_shards_list.append((shard_filename, current_shard))
                    shard_idx += 1
                    current_shard = {}
                    current_size = 0
                
                current_shard[key] = tensor.contiguous()
                current_size += tensor_size
            
            # 保存最后一个shard
            if current_shard:
                shard_filename = f"model-{shard_idx+1:05d}-of-{shard_idx+1:05d}.safetensors"
                all_shards_list.append((shard_filename, current_shard))
                total_shards = shard_idx + 1
            
            # 写入文件并创建索引
            index = {"metadata": {"total_size": total_size}, "weight_map": {}}
            for shard_filename, shard_dict in all_shards_list:
                # 更新文件名中的total_shards
                updated_filename = shard_filename.replace(f"-of-{0:05d}", f"-of-{total_shards:05d}")
                save_file(shard_dict, os.path.join(hf_path, updated_filename))
                print(f"      Saved {updated_filename}")
                
                # 记录weight_map
                for key in shard_dict.keys():
                    index["weight_map"][key] = updated_filename
            
            # 创建safetensors索引文件
            if total_shards > 1:
                with open(os.path.join(hf_path, 'model.safetensors.index.json'), 'w') as f:
                    json.dump(index, f, indent=2)
                    
        except ImportError:
            # 如果没有safetensors，保存为pytorch_model.bin
            print(f"    Saving as pytorch_model.bin (safetensors not available)...")
            torch.save(state_dict, os.path.join(hf_path, 'pytorch_model.bin'))
        
        # 4. 确保有config.json
        if not os.path.exists(os.path.join(hf_path, 'config.json')):
            config_source = os.path.join(hf_source, 'config.json') if os.path.exists(hf_source) else None
            if config_source and os.path.exists(config_source):
                shutil.copy2(config_source, os.path.join(hf_path, 'config.json'))
            else:
                print(f"    Warning: config.json not found, please manually add it")
        
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
# 步骤3: 改写模型推理
# =============================================================================

class RewriteModelClient:
    """改写模型客户端"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.tokenizer = None
        self.engine = None
        self._loop = None
        
    def initialize(self):
        """延迟初始化"""
        if self.engine is not None:
            return
        
        import sglang as sgl
        import asyncio
        
        print(f"  [RewriteModel] Initializing SGLang engine...")
        print(f"    Model: {self.model_path}")
        print(f"    TP size: {self.tensor_parallel_size}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        
        self.engine = sgl.Engine(
            model_path=self.model_path,
            tp_size=self.tensor_parallel_size,
            trust_remote_code=True,
            mem_fraction_static=0.85,
        )
        
        # CRITICAL: Re-create event loop after Engine creation
        # because sglang forces uvloop policy which clears the existing event loop
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        print(f"  [RewriteModel] Engine initialized")
    
    def _ensure_event_loop(self):
        """确保event loop存在"""
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 15000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: float = 1.2,
        frequency_penalty: float = 1.2,
    ) -> str:
        """生成改写结果"""
        self.initialize()
        self._ensure_event_loop()
        
        # 使用chat template构建prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.engine.generate(
            prompt=prompt,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
        )
        
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
        else:
            output = outputs
        
        if isinstance(output, dict):
            return output.get("text", "")
        return str(output)
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 15000,
        temperature: float = 0.6,
        top_p: float = 0.95,
        presence_penalty: float = 1.2,
        frequency_penalty: float = 1.2,
    ) -> List[str]:
        """
        批量生成改写结果
        
        这比逐个调用generate更高效，且避免频繁的event loop问题
        """
        self.initialize()
        self._ensure_event_loop()
        
        outputs = self.engine.generate(
            prompt=prompts,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
        )
        
        results = []
        for output in outputs:
            if isinstance(output, dict):
                results.append(output.get("text", ""))
            else:
                results.append(str(output))
        
        return results
    
    def shutdown(self):
        """关闭引擎"""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


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
    rewrite_client: RewriteModelClient
) -> List[Dict]:
    """
    步骤3: 调用改写模型生成n次结果
    
    支持断点续传。使用批量生成以避免event loop问题。
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
    
    # 初始化改写客户端
    rewrite_client.initialize()
    
    total_samples = len(samples)
    remaining_samples = total_samples - start_idx
    
    print(f"  Total samples: {total_samples}")
    print(f"  Already processed: {start_idx}")
    print(f"  Remaining: {remaining_samples}")
    print(f"  N rewrites per sample: {config.n_rewrites}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  TP size: {config.sglang_tensor_parallel_size}")
    
    # 创建进度条
    pbar = tqdm(
        range(start_idx, total_samples),
        initial=start_idx,
        total=total_samples,
        desc="Rewrite",
        unit="sample",
        dynamic_ncols=True
    )
    
    # 批量处理样本
    for idx in pbar:
        sample = samples[idx]
        question = sample['question']
        thought = sample['thought']
        
        # 更新进度条描述
        pbar.set_postfix({
            'sample': f'{idx}/{total_samples}',
            'progress': f'{(idx/total_samples)*100:.1f}%'
        })
        
        # 构建prompt
        system_prompt, user_prompt = build_prompt(question, thought)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # 使用chat template构建prompt字符串（用于批量生成）
        prompt = rewrite_client.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成n次（批量生成）
        rewrite_results = []
        try:
            # 批量生成n个结果
            prompts = [prompt] * config.n_rewrites
            outputs = rewrite_client.generate_batch(
                prompts,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
            )
            
            success_count = 0
            for n, output in enumerate(outputs):
                try:
                    rewrite_reasoning = extract_rewrite_reasoning(output)
                    rewrite_results.append({
                        'rewrite_output': output,
                        'rewrite_reasoning': rewrite_reasoning,
                        'success': rewrite_reasoning is not None
                    })
                    if rewrite_reasoning is not None:
                        success_count += 1
                except Exception as inner_e:
                    rewrite_results.append({
                        'rewrite_output': output,
                        'rewrite_reasoning': None,
                        'success': False
                    })
            
            # 更新进度条显示成功数
            pbar.set_postfix({
                'sample': f'{idx}/{total_samples}',
                'success': f'{success_count}/{config.n_rewrites}'
            })
                    
        except Exception as e:
            # 如果批量生成失败，使用空结果填充
            for n in range(config.n_rewrites):
                rewrite_results.append({
                    'rewrite_output': str(e),
                    'rewrite_reasoning': None,
                    'success': False
                })
        
        # 保存结果
        result_item = {
            'sample_idx': idx,
            'question': question,
            'original_thought': thought,
            'answer': sample.get('answer', ''),
            'rewrite_results': rewrite_results
        }
        results.append(result_item)
        
        # 实时保存
        save_jsonl([result_item], ckpt_path, append=(idx > 0))
        
        # 更新进度
        with open(progress_path, 'w') as f:
            json.dump({'last_completed_idx': idx + 1}, f)
        
        if (idx + 1) % 10 == 0 or idx == total_samples - 1:
            print(f"  Progress: {idx + 1}/{total_samples} ({(idx + 1) / total_samples * 100:.1f}%)")
    
    print(f"  Completed. Results saved to: {ckpt_path}")
    return results


# =============================================================================
# 步骤4: Similarity Penalty过滤
# =============================================================================

def compute_similarity_worker(args: Tuple[int, Dict, float]) -> Tuple[int, List[Dict]]:
    """多进程worker：计算单个样本的similarity"""
    idx, sample, threshold = args
    
    original_thought = sample['original_thought']
    rewrite_results = sample.get('rewrite_results', [])
    
    filtered_results = []
    
    for r_idx, result in enumerate(rewrite_results):
        rewrite_reasoning = result.get('rewrite_reasoning')
        
        if not rewrite_reasoning:
            continue
        
        similarity = compute_similarity_penalty(original_thought, rewrite_reasoning)
        
        result['similarity_penalty'] = similarity
        result['similarity_passed'] = similarity < threshold
        
        filtered_results.append(result)
    
    return idx, filtered_results


def step_similarity_filter(
    config: RewriteConfig,
    samples: List[Dict]
) -> List[Dict]:
    """
    步骤4: 计算similarity_penalty并过滤
    
    过滤掉similarity_penalty > threshold的结果
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
            idx, filtered = future.result()
            results[idx] = filtered
            pbar.update(1)
        pbar.close()
    
    # 更新样本
    filtered_samples = []
    for i, sample in enumerate(samples):
        sample_copy = sample.copy()
        sample_copy['rewrite_results'] = results[i] if results[i] else []
        filtered_samples.append(sample_copy)
    
    # 统计
    total_before = sum(len(s['rewrite_results']) for s in samples)
    total_after = sum(len(s['rewrite_results']) for s in filtered_samples)
    
    print(f"  Before filter: {total_before} rewrites")
    print(f"  After filter: {total_after} rewrites")
    print(f"  Filtered out: {total_before - total_after} rewrites ({(total_before - total_after) / total_before * 100:.1f}%)")
    
    return filtered_samples


# =============================================================================
# 步骤5: PPL过滤
# =============================================================================

from verl.experimental.reward.cot_rewriter.sglang_student_reward import SGLangStudentRewardClient


def step_ppl_filter(
    config: RewriteConfig,
    samples: List[Dict]
) -> List[Dict]:
    """
    步骤5: 计算ppl_relative_improvement并过滤
    
    支持断点续传
    """
    if config.skip_ppl:
        print("\n  [Skip] PPL filter skipped")
        return samples
    
    print("\n" + "=" * 60)
    print("Step 5: PPL Relative Improvement Filter")
    print("=" * 60)
    print(f"  Student Model: {config.student_model_path}")
    print(f"  Threshold: {config.ppl_relative_improvement_threshold}")
    
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
    
    # 初始化SGLang客户端
    print("  Initializing SGLang student reward client...")
    student_client = SGLangStudentRewardClient(
        model_path=config.student_model_path,
        tensor_parallel_size=config.sglang_tensor_parallel_size,
        reward_mode="ppl",
        use_reward_delta=True,
        devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        micro_batch_size=config.sgland_micro_batch_size,
    )
    student_client.initialize()
    
    total_samples = len(samples)
    all_ppl_improvements = []
    
    # 创建进度条
    pbar = tqdm(
        range(start_idx, total_samples),
        initial=start_idx,
        total=total_samples,
        desc="PPL Filter",
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
            # 计算cognitive reward
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
                
                relative_improvements = reward_result.get('relative_improvement_rates', [])
                
                # 更新结果
                for i, r_idx in enumerate(result_indices):
                    if i < len(relative_improvements):
                        ppl_imp = relative_improvements[i]
                        sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = ppl_imp
                        sample['rewrite_results'][r_idx]['ppl_passed'] = ppl_imp > config.ppl_relative_improvement_threshold
                        all_ppl_improvements.append(ppl_imp)
                    else:
                        sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = -float('inf')
                        sample['rewrite_results'][r_idx]['ppl_passed'] = False
                        
            except Exception as e:
                pbar.set_postfix({'sample': f'{idx}/{total_samples}', 'error': str(e)[:30]})
                for r_idx in result_indices:
                    sample['rewrite_results'][r_idx]['ppl_relative_improvement'] = -float('inf')
                    sample['rewrite_results'][r_idx]['ppl_passed'] = False
        
        processed_samples.append(sample)
        
        # 保存进度
        save_jsonl([sample], ckpt_path, append=(idx > 0))
        with open(progress_path, 'w') as f:
            json.dump({'last_completed_idx': idx + 1}, f)
        
        if (idx + 1) % 10 == 0 or idx == total_samples - 1:
            print(f"  Progress: {idx + 1}/{total_samples}")
    
    # 输出PPL分布统计
    if all_ppl_improvements:
        print("\n  PPL Relative Improvement Distribution:")
        print(f"    Mean: {np.mean(all_ppl_improvements):.4f}")
        print(f"    Std: {np.std(all_ppl_improvements):.4f}")
        print(f"    Min: {np.min(all_ppl_improvements):.4f}")
        print(f"    Max: {np.max(all_ppl_improvements):.4f}")
        print(f"    Median: {np.median(all_ppl_improvements):.4f}")
        print(f"    Passed: {sum(1 for x in all_ppl_improvements if x > config.ppl_relative_improvement_threshold)}/{len(all_ppl_improvements)}")
    
    print(f"  Completed. Results saved to: {ckpt_path}")
    
    # 清理
    student_client.shutdown()
    
    return processed_samples


# =============================================================================
# 步骤6: FSF过滤
# =============================================================================

from verl.experimental.reward.cot_rewriter.sglang_fsf_reward import SGLangFSFClient


def step_fsf_filter(
    config: RewriteConfig,
    samples: List[Dict]
) -> List[Dict]:
    """
    步骤6: 计算FSF并过滤
    
    过滤掉fsf > threshold的结果
    支持断点续传
    """
    if config.skip_fsf:
        print("\n  [Skip] FSF filter skipped")
        return samples
    
    print("\n" + "=" * 60)
    print("Step 6: FSF Filter")
    print("=" * 60)
    print(f"  FSF Judge Model: {config.fsf_judge_model_path}")
    print(f"  Threshold: {config.fsf_threshold}")
    
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
        micro_batch_size=config.sgland_micro_batch_size,
    )
    fsf_client.initialize()
    
    total_samples = len(samples)
    
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
            try:
                fsf_result = fsf_client.compute_rewrite_fsf_only(
                    rewrite_cots=rewrite_cots,
                )
                
                fsf_values = fsf_result.get('rewrite_fsfs', [])
                
                # 更新结果
                for i, r_idx in enumerate(result_indices):
                    if i < len(fsf_values) and fsf_values[i] is not None:
                        fsf_val = fsf_values[i]
                        sample['rewrite_results'][r_idx]['fsf'] = fsf_val
                        sample['rewrite_results'][r_idx]['fsf_passed'] = fsf_val <= config.fsf_threshold
                    else:
                        sample['rewrite_results'][r_idx]['fsf'] = float('inf')
                        sample['rewrite_results'][r_idx]['fsf_passed'] = False
                        
            except Exception as e:
                pbar.set_postfix({'sample': f'{idx}/{total_samples}', 'error': str(e)[:30]})
                for r_idx in result_indices:
                    sample['rewrite_results'][r_idx]['fsf'] = float('inf')
                    sample['rewrite_results'][r_idx]['fsf_passed'] = False
        
        processed_samples.append(sample)
        
        # 保存进度
        save_jsonl([sample], ckpt_path, append=(idx > 0))
        with open(progress_path, 'w') as f:
            json.dump({'last_completed_idx': idx + 1}, f)
    
    print(f"  Completed. Results saved to: {ckpt_path}")
    
    # 清理
    fsf_client.shutdown()
    
    return processed_samples


# =============================================================================
# 步骤7: 选择最优结果并生成数据集
# =============================================================================

def select_best_rewrite(results: List[Dict]) -> Optional[Dict]:
    """
    选择最优的改写结果
    
    优先级:
    1. ppl_relative_improvement 最高
    2. fsf 最低
    3. similarity_penalty 最低
    """
    if not results:
        return None
    
    # 首先过滤掉未通过检查的结果
    valid_results = []
    for r in results:
        # 必须成功提取reasoning
        if not r.get('rewrite_reasoning'):
            continue
        # 检查是否通过各项检查（如果进行了该检查）
        if 'similarity_passed' in r and not r['similarity_passed']:
            continue
        if 'ppl_passed' in r and not r['ppl_passed']:
            continue
        if 'fsf_passed' in r and not r['fsf_passed']:
            continue
        valid_results.append(r)
    
    if not valid_results:
        return None
    
    # 按优先级排序
    # 优先级: ppl_relative_improvement (高->低), fsf (低->高), similarity_penalty (低->高)
    sorted_results = sorted(
        valid_results,
        key=lambda x: (
            -x.get('ppl_relative_improvement', -float('inf')),  # 越高越好
            x.get('fsf', float('inf')),  # 越低越好
            x.get('similarity_penalty', float('inf'))  # 越低越好
        )
    )
    
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
                'ppl_relative_improvement': best_rewrite.get('ppl_relative_improvement'),
                'fsf': best_rewrite.get('fsf'),
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
    
    # 步骤0: 转换模型格式
    print("\n" + "=" * 60)
    print("Step 0: Model Format Conversion")
    print("=" * 60)
    rewrite_model_hf_path = convert_model_to_hf(config.rewrite_model_path)
    print(f"  Rewrite model (HF): {rewrite_model_hf_path}")
    
    # 初始化改写模型客户端（如果需要执行rewrite步骤）
    rewrite_client = None
    if args.step in ['all', 'rewrite']:
        rewrite_client = RewriteModelClient(
            model_path=rewrite_model_hf_path,
            tensor_parallel_size=config.sglang_tensor_parallel_size
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
        samples = step_similarity_filter(config, samples)
        # 保存中间结果
        save_jsonl(samples, get_checkpoint_path(config.output_dir, '1_5'))
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, '1_5')
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)
    
    if args.step == 'all' or args.step == 'ppl':
        # 步骤5: PPL过滤
        samples = step_ppl_filter(config, samples)
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, 2)
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)
    
    if args.step == 'all' or args.step == 'fsf':
        # 步骤6: FSF过滤
        samples = step_fsf_filter(config, samples)
    else:
        ckpt_path = get_checkpoint_path(config.output_dir, 3)
        if os.path.exists(ckpt_path):
            samples = load_jsonl(ckpt_path)
    
    if args.step == 'all' or args.step == 'select':
        # 步骤7: 选择和生成
        new_dataset, stats = step_select_and_generate(config, samples)
    
    print("\n" + "=" * 60)
    print("All steps completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
