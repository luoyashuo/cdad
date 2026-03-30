#!/usr/bin/env python3
"""
FSF (Failure Step Fraction) Calculator - vLLM Version with Ray

Computes FSF = failed_nodes / total_nodes from DOT graph format using vLLM.
Failed nodes are those with fillcolor=lightpink.

Requirements:
- 8 nodes, 8 GPUs, data parallelism
- Maximize GPU memory usage
- Use vLLM for inference
- Continuous batching with tqdm progress

Usage:
    python caculate_fsf.py --num_gpus 8 --gpu_memory_utilization 0.95
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Set environment variables before importing torch/vllm/ray
os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

import ray
import torch
from transformers import AutoTokenizer

# GRAPH_GENERATION_PROMPT from build_fsf_judge_graph.py
GRAPH_GENERATION_PROMPT = """Parse the reasoning trace into a Graphviz diagram. Focus on these essentials:

Node Rules:
- One node per distinct reasoning step
- 'fillcolor=lightblue': Successful reasoning steps
- 'fillcolor=lightpink': Failed attempts

Edge Rules:
- Connect node A → node B if the information or insight from A is actually used to construct the reasoning in B; branch new attempts from their starting ancestor, not from dead ends.

Requirements:
- Use 'rankdir=TB'
- Include ALL attempts (including failures), do not miss any steps in the reasoning.
- ALWAYS start with a "problem statement" node
- ALWAYS end with a "final answer" node
- Do NOT reorder or reorganize the reasoning flow

Generate complete Graphviz DOT code in dot blocks."""


def build_fsf_prompt(reasoning: str, tokenizer) -> str:
    """
    Build FSF prompt with proper format.
    
    Prompt Template (from build_fsf_judge_graph.py):
    - System: "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."
    - User: GRAPH_GENERATION_PROMPT + Reasoning trace
    """
    # 处理 <rewrite_reasoning> 标签
    if '<rewrite_reasoning>' in reasoning:
        reasoning = reasoning.split('<rewrite_reasoning>')[-1]
    if '</rewrite_reasoning>' in reasoning:
        reasoning = reasoning.split('</rewrite_reasoning>')[0]
    
    reasoning = reasoning.strip()
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams.",
        },
        {
            "role": "user",
            "content": f"{GRAPH_GENERATION_PROMPT}\n\nReasoning trace:\n{reasoning}",
        },
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt


def parse_fsf_from_dot(dot_output: str) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[Dict[str, int]]]:
    """
    Parse FSF value from DOT graph format output.
    
    FSF = failed_nodes / total_nodes
    Failed nodes have fillcolor=lightpink
    
    Returns:
        Tuple of (fsf, failed_nodes, total_nodes, node_count_dict)
    """
    try:
        import pydot
        
        # Strip leading newlines and markdown code blocks
        cleaned_output = dot_output.lstrip('\n\n').strip('`')
        
        # Directly parse with pydot
        graphs = pydot.graph_from_dot_data(cleaned_output)
        
        if not graphs or len(graphs) == 0:
            return None, None, None, None
        
        graph = graphs[0]
        nodes = graph.get_nodes() or []
        
        node_count = len(nodes)
        if node_count == 0:
            return 0.0, 0, 0, {"total": 0, "failed": 0, "successful": 0, "other": 0}
        
        # Count nodes by type
        failed_nodes = 0
        successful_nodes = 0
        other_nodes = 0
        
        for node in nodes:
            if node is None:
                continue
            try:
                fillcolor = node.get('fillcolor')
                if fillcolor is not None:
                    fillcolor_str = str(fillcolor).strip('"').lower()
                    if fillcolor_str == 'lightpink':
                        failed_nodes += 1
                    elif fillcolor_str == 'lightblue':
                        successful_nodes += 1
                    else:
                        other_nodes += 1
                else:
                    other_nodes += 1
            except:
                other_nodes += 1
                continue
        
        fsf = failed_nodes / node_count if node_count > 0 else 0.0
        
        node_count_dict = {
            "total": node_count,
            "failed": failed_nodes,
            "successful": successful_nodes,
            "other": other_nodes
        }
        
        return fsf, failed_nodes, node_count, node_count_dict
        
    except ImportError:
        print("Warning: pydot not installed")
        return None, None, None, None
    except Exception as e:
        print(f"DOT parsing error: {e}")
        return None, None, None, None


@ray.remote(num_gpus=1, num_cpus=1)
class VLLMFSFActor:
    """
    Ray actor that runs FSF computation using vLLM on one GPU.
    Uses continuous batching for maximum throughput.
    """
    
    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        max_model_len: int = 8192,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        max_new_tokens: int = 8000,
        gpu_memory_utilization: float = 0.95,
        max_num_seqs: int = 256,  # 增加并发序列数，充分利用 continuous batching
    ):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Import vLLM here to avoid serialization issues
        from vllm import LLM, SamplingParams
        
        # Initialize vLLM engine with continuous batching optimizations
        print(f"[Actor-{gpu_id}] Initializing vLLM engine on GPU {gpu_id}...")
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,  # Each actor uses 1 GPU
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            max_num_seqs=self.max_num_seqs,  # 最大并发序列数
            max_num_batched_tokens=max_model_len,  # 最大 batch token 数
            enable_chunked_prefill=True,  # 启用 chunked prefill 提高吞吐量
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            presence_penalty=1.0,
        )
        
        print(f"[Actor-{gpu_id}] vLLM engine initialized (max_num_seqs={max_num_seqs})")
    
    def compute_fsf_batch(self, items: List[Dict[str, str]], worker_id: int = 0, total_workers: int = 1, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Compute FSF for a batch of items with progress reporting and retry logic.
        
        Args:
            items: List of dict with 'thought' and 'solution' keys
            worker_id: Worker ID for progress reporting
            total_workers: Total number of workers for progress reporting
            max_retries: Maximum number of retries for failed parsing
            
        Returns:
            List of result dicts
        """
        if not items:
            return []
        
        results = [None] * len(items)
        pending_indices = list(range(len(items)))  # 记录还需要处理的索引
        
        for attempt in range(max_retries):
            if not pending_indices:
                break  # 全部成功，提前退出
            
            # 只为失败的样本构建 prompts
            prompts = []
            current_batch_indices = []
            for idx in pending_indices:
                item = items[idx]
                thought = item['thought']
                # Truncate long reasoning to avoid OOM
                thought_truncated = thought[:3000] if len(thought) > 3000 else thought
                prompt = build_fsf_prompt(thought_truncated, self.tokenizer)
                prompts.append(prompt)
                current_batch_indices.append(idx)
            
            # Generate with vLLM - continuous batching happens automatically here
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Parse results and检查哪些成功了
            new_pending_indices = []
            for batch_idx, (output_idx, output) in enumerate(zip(current_batch_indices, outputs)):
                generated_text = output.outputs[0].text if output.outputs else ""
                
                # Parse FSF from DOT output
                fsf, failed_nodes, node_count, node_count_dict = parse_fsf_from_dot(generated_text)
                
                if fsf is not None:
                    # 解析成功，保存结果
                    results[output_idx] = {
                        'thought': items[output_idx]['thought'],
                        'solution': items[output_idx]['solution'],
                        'dot_graph': generated_text,
                        'fsf': fsf,
                        'failed_nodes': failed_nodes,
                        'node_count': node_count,
                        'node_count_dict': node_count_dict,
                        'retry_count': attempt,  # 记录重试次数
                    }
                else:
                    # 解析失败，需要重试
                    if attempt < max_retries - 1:
                        new_pending_indices.append(output_idx)
                    else:
                        # 最后一次尝试也失败了，标记为失败
                        results[output_idx] = {
                            'thought': items[output_idx]['thought'],
                            'solution': items[output_idx]['solution'],
                            'dot_graph': generated_text,
                            'fsf': None,
                            'failed_nodes': None,
                            'node_count': None,
                            'node_count_dict': None,
                            'retry_count': attempt,
                            'parse_failed': True,  # 标记解析失败
                        }
            
            pending_indices = new_pending_indices
            
            if pending_indices and attempt < max_retries - 1:
                print(f"[Actor-{self.gpu_id}] Retry {attempt + 1}: {len(pending_indices)} items failed to parse, retrying...")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vLLM engine stats."""
        return {
            'gpu_id': self.gpu_id,
            'max_num_seqs': self.max_num_seqs,
        }


class VLLMFSFClient:
    """
    Multi-GPU vLLM-based FSF Client using Ray actors.
    Uses continuous batching for maximum throughput.
    """
    
    def __init__(
        self,
        model_path: str,
        num_gpus: int = 8,
        max_model_len: int = 8192,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        max_new_tokens: int = 8000,
        gpu_memory_utilization: float = 0.95,
        max_num_seqs: int = 256,
    ):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        
        self.workers: List[ray.actor.ActorHandle] = []
        self._initialized = False
    
    def initialize(self):
        """Initialize Ray actors."""
        if self._initialized:
            return
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Get available GPUs
        num_available_gpus = torch.cuda.device_count()
        num_workers = min(self.num_gpus, num_available_gpus)
        
        print(f"Creating {num_workers} vLLM actors with max_num_seqs={self.max_num_seqs}...")
        
        # Create Ray actors
        for gpu_id in range(num_workers):
            worker = VLLMFSFActor.remote(
                model_path=self.model_path,
                gpu_id=gpu_id,
                max_model_len=self.max_model_len,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_seqs=self.max_num_seqs,
            )
            self.workers.append(worker)
            print(f"Created worker {gpu_id}")
        
        # Initialize workers with a dummy call
        print("Initializing workers (warm-up)...")
        init_futures = [w.compute_fsf_batch.remote([{'thought': 'test', 'solution': ''}]) for w in self.workers]
        try:
            ray.get(init_futures, timeout=300)
            print(f"All {len(self.workers)} workers initialized and warmed up")
        except Exception as e:
            print(f"Worker initialization failed: {e}")
            raise
        
        self._initialized = True
    
    def shutdown(self):
        """Shutdown all workers."""
        if not self._initialized:
            return
        
        for w in self.workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception:
                pass
        
        self.workers = []
        self._initialized = False
        
        # Clean up
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        print("All workers shutdown")
    
    def compute_fsf_batch(self, items: List[Dict[str, str]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Compute FSF for a batch of items in parallel across GPUs.
        Uses continuous batching within each vLLM instance.
        """
        self.initialize()
        
        if not items:
            return []
        
        # Split items across workers evenly
        num_workers = len(self.workers)
        chunk_size = max(1, (len(items) + num_workers - 1) // num_workers)  # 向上取整
        chunks = []
        
        for i in range(num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, len(items))
            if start < len(items):
                chunks.append(items[start:end])
            else:
                chunks.append([])
        
        # Submit tasks to workers
        print(f"Distributing {len(items)} items across {num_workers} workers (chunk_size={chunk_size})...")
        futures = []
        for worker_id, (worker, chunk) in enumerate(zip(self.workers, chunks)):
            if chunk:
                future = worker.compute_fsf_batch.remote(chunk, worker_id, num_workers)
                futures.append((worker_id, future, len(chunk)))
        
        # Collect results with progress bar
        all_results = [None] * len(items)
        stats = {
            'success': 0,      # 解析成功（包括重试后成功）
            'failed': 0,       # 解析失败（3次都失败）
            'retried': 0,      # 至少重试过1次的数量
        }
        
        if show_progress:
            pbar = tqdm(total=len(items), desc="Processing", unit="item")
        
        completed = 0
        for worker_id, future, chunk_size in futures:
            try:
                results = ray.get(future, timeout=3600)
                # Map results back to original indices
                start_idx = sum(len(chunks[i]) for i in range(worker_id))
                for i, result in enumerate(results):
                    all_results[start_idx + i] = result
                    
                    # 统计结果
                    if result is not None:
                        if result.get('fsf') is not None:
                            stats['success'] += 1
                            if result.get('retry_count', 0) > 0:
                                stats['retried'] += 1
                        elif result.get('parse_failed'):
                            stats['failed'] += 1
                
                completed += len(results)
                if show_progress:
                    pbar.update(len(results))
                    # 更新进度条后缀显示统计信息
                    pbar.set_postfix({
                        'success': stats['success'],
                        'failed': stats['failed'],
                        'retried': stats['retried']
                    })
                
                print(f"[Worker-{worker_id}] Completed {len(results)}/{chunk_size} items")
            except Exception as e:
                print(f"[Worker-{worker_id}] Error: {e}")
                # Add None results for failed chunk
                start_idx = sum(len(chunks[i]) for i in range(worker_id))
                for i in range(chunk_size):
                    all_results[start_idx + i] = None
                    stats['failed'] += 1
                completed += chunk_size
                if show_progress:
                    pbar.update(chunk_size)
        
        if show_progress:
            pbar.close()
        
        # 打印统计信息
        print(f"\n[Summary] Success: {stats['success']}, Failed: {stats['failed']}, Retried: {stats['retried']}")
        
        return all_results
        
        if show_progress:
            pbar.close()
        
        return all_results


def process_file(
    input_file: str,
    output_file: str,
    model_path: str,
    num_gpus: int = 8,
    max_model_len: int = 8192,
    temperature: float = 0.6,
    top_k: int = 20,
    top_p: float = 0.95,
    max_new_tokens: int = 8000,
    gpu_memory_utilization: float = 0.95,
    max_num_seqs: int = 256,
):
    """
    Process the input file and compute FSF for all thoughts.
    """
    # Read input file with progress
    print(f"Reading input file: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} questions")
    
    # Extract all (question, thought, solution) triples
    all_items = []
    for idx, item in enumerate(data):
        question = item.get('question', '')
        reasoning = item.get('reasoning', [])
        for r_idx, r in enumerate(reasoning):
            if isinstance(r, dict):
                thought = r.get('thought', '')
                solution = r.get('solution', '')
            else:
                thought = str(r)
                solution = ''
            all_items.append({
                'question_idx': idx,
                'question': question,
                'reasoning_idx': r_idx,
                'thought': thought,
                'solution': solution,
            })
    
    print(f"Total thoughts to process: {len(all_items)}")
    
    # Create client and process
    client = VLLMFSFClient(
        model_path=model_path,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    
    try:
        # Compute FSF for all items with progress bar
        print("\nStarting FSF computation with continuous batching...")
        results = client.compute_fsf_batch([
            {'thought': item['thought'], 'solution': item['solution']}
            for item in all_items
        ], show_progress=True)
        
        # Organize results by question
        print("\nOrganizing results...")
        output_data = []
        for idx, item in enumerate(tqdm(data, desc="Organizing", unit="question")):
            question = item.get('question', '')
            reasoning_results = []
            
            # Find all results for this question
            for item_info, result in zip(all_items, results):
                if item_info['question_idx'] == idx and result is not None:
                    reasoning_results.append(result)
            
            output_data.append({
                'question': question,
                'reasoning': reasoning_results
            })
        
        # Write output
        print(f"\nWriting output to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(output_data, desc="Writing", unit="line"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Print statistics
        total_thoughts = len(all_items)
        successful_thoughts = sum(1 for r in results if r is not None and r.get('fsf') is not None)
        failed_thoughts = sum(1 for r in results if r is not None and r.get('parse_failed'))
        retried_thoughts = sum(1 for r in results if r is not None and r.get('retry_count', 0) > 0)
        
        # 统计重试分布
        retry_distribution = {}
        for r in results:
            if r is not None:
                retry_count = r.get('retry_count', 0)
                retry_distribution[retry_count] = retry_distribution.get(retry_count, 0) + 1
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total thoughts: {total_thoughts}")
        print(f"Successful: {successful_thoughts} (including {retried_thoughts} after retry)")
        print(f"Failed: {failed_thoughts} (all 3 retries exhausted)")
        print(f"Success rate: {100*successful_thoughts/total_thoughts:.2f}%")
        print(f"\nRetry distribution:")
        for retry_count in sorted(retry_distribution.keys()):
            count = retry_distribution[retry_count]
            print(f"  - {retry_count} attempt(s): {count} items")
        print(f"{'='*60}")
        
    finally:
        client.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Compute FSF for reasoning traces using vLLM with Ray")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train.jsonl",
        help="Input JSONL file with question and reasoning data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl",
        help="Output JSONL file with FSF results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/hf_home/models/Yaseal-qwen3_4b_fsf_grm",
        help="Path to the model"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8000,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="Maximum number of concurrent sequences per GPU (continuous batching)"
    )
    
    args = parser.parse_args()
    
    process_file(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
