"""
Student Reward (Cognitive Reward) Module - SGLang Version

Computes R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
using SGLang for fast log prob computation.

其中 answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
compared to the original CoT.

Uses Ray actors for multi-GPU parallelism. Supports two modes:
1. TP=1: Each GPU runs a separate SGLang instance (Data Parallelism)
2. TP>1: Multiple GPUs share one SGLang instance (Tensor Parallelism)

SGLang 特有优化：
- enable_memory_saver: 启用内存保护，支持 release/resume_memory_occupation
- logprob_start_len: 只计算最后 N 个 token 的 logprobs，节省显存
"""

import asyncio
from typing import Any, Dict, List, Optional
import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger
from transformers import AutoTokenizer
import os
import torch


@ray.remote
class SGLangWorkerActor:
    """
    Ray actor that runs SGLang on one or more GPUs.

    Supports Tensor Parallelism (TP):
    - TP=1: Single GPU, model weights fully on one GPU
    - TP>1: Multiple GPUs, model weights sharded across GPUs
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],           # Physical GPU IDs (e.g., [0, 1] for TP=2)
        logical_gpu_ids: List[int],   # Logical GPU IDs within CUDA_VISIBLE_DEVICES
        tensor_parallel_size: int = 1,  # TP size (must equal len(gpu_ids))
        mem_fraction_static: float = 0.85,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,
        logprob_last_n_tokens: Optional[int] = None,  # 只计算最后 N 个 token 的 logprobs
    ):
        import os
        import subprocess

        # Validate TP configuration
        assert len(gpu_ids) == tensor_parallel_size, \
            f"gpu_ids length {len(gpu_ids)} must equal tensor_parallel_size {tensor_parallel_size}"
        assert len(logical_gpu_ids) == tensor_parallel_size, \
            f"logical_gpu_ids length {len(logical_gpu_ids)} must equal tensor_parallel_size {tensor_parallel_size}"

        self.gpu_ids = gpu_ids
        self.tensor_parallel_size = tensor_parallel_size
        self.logprob_last_n_tokens = logprob_last_n_tokens
        worker_name = f"SGLangWorker-{gpu_ids}"

        # Ray actor inherits CUDA_VISIBLE_DEVICES from parent process
        parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        # Set CUDA_VISIBLE_DEVICES to all logical GPUs for this actor
        # e.g., logical_gpu_ids=[0, 1] → "0,1" for TP=2
        cuda_devices_str = ",".join(map(str, logical_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Log GPU memory status BEFORE loading model
        def get_nvidia_smi_memory_for_gpu(gpu_idx):
            """Get GPU memory from nvidia-smi for a specific GPU"""
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3 and int(parts[0].strip()) == gpu_idx:
                        used_mb = int(parts[1].strip())
                        total_mb = int(parts[2].strip())
                        used_gb = used_mb / 1024
                        total_gb = total_mb / 1024
                        return f"GPU {gpu_idx}: {used_gb:.1f}GB / {total_gb:.1f}GB ({used_mb}MB used)"
                return f"GPU {gpu_idx}: not found"
            except Exception as e:
                return f"nvidia-smi error: {e}"

        def get_all_gpus_memory():
            """Get memory for all GPUs in this actor"""
            return ", ".join(get_nvidia_smi_memory_for_gpu(gpu) for gpu in gpu_ids)

        # GPU memory check (for debugging if needed)
        pass

        self._worker_name = worker_name
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu
        self._get_all_gpus_memory = get_all_gpus_memory

        # Create event loop for SGLang (required when running in Ray actor)
        # Reference: verl-0.5.0/verl/workers/reward_manager/prime.py
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info(f"[SGLangWorkerActor-{gpu_ids}] Created new event loop")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        import time
        import sglang as sgl
        
        llm_start = time.time()
        
        # 初始化 SGLang Engine
        # 注意：不启用 enable_memory_saver，因为 Student Model 每次都是新建并很快销毁
        # 不需要与 Actor/Rollout 共享 GPU 内存
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            mem_fraction_static=mem_fraction_static,
            dtype=dtype,
            trust_remote_code=True,
            enable_memory_saver=False,  # 禁用内存保护，避免与 Actor/Rollout 的 SGLang 冲突
        )
        
        # CRITICAL: Re-create event loop after Engine creation
        # because sglang forces uvloop policy which clears the existing event loop
        # This must be done before any generate() calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.info(f"[SGLangWorkerActor-{gpu_ids}] Event loop re-created after Engine init")
        
        llm_elapsed = time.time() - llm_start
        self.micro_batch_size = micro_batch_size
        
        logger.info(f"[SGLangWorkerActor-{gpu_ids}] Engine created in {llm_elapsed:.2f}s")

    def compute_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a batch of prompts.

        Uses micro-batching to avoid OOM with long sequences.
        
        SGLang 特有优化：
        - 使用 logprob_start_len 参数只计算 answer 部分的 logprobs
        - 这样可以大幅减少返回的数据量，节省显存

        Returns:
            List of dicts, each containing:
                - mean_log_prob: float
                - min_log_prob: float
                - token_log_probs: List[float]
        """
        import time

        # Ensure event loop exists (required when running in Ray actor)
        self._ensure_event_loop()

        # Micro-batch processing to avoid OOM
        all_outputs = []
        total_time = 0
        micro_batch_size = self.micro_batch_size

        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]

            start_time = time.time()
            
            # SGLang 生成参数
            # 注意：我们使用 max_tokens=1 因为只需要 logprobs，不需要生成新内容
            # 但实际上 SGLang 需要至少生成 1 个 token 来获取 prompt_logprobs
            batch_outputs = []
            for prompt, ans_start in zip(batch_prompts, batch_ans_starts):
                # IMPORTANT: Set logprob_start_len=ans_start to ONLY get answer portion logprobs
                # SGLang's input_token_logprobs only includes tokens from logprob_start_len onwards
                # We only need answer tokens, not question and CoT tokens
                logprob_start_len = ans_start
                
                # 调用 SGLang engine
                # 注意：SGLang 的 generate 返回的是一个 dict (single prompt) 或 list
                outputs = self.engine.generate(
                    prompt=prompt,
                    sampling_params={
                        "max_new_tokens": 1,  # 只需要 1 个 token 来触发 prompt_logprobs
                        "temperature": 0,
                    },
                    return_logprob=True,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=1,
                )
                batch_outputs.append(outputs)
            
            elapsed = time.time() - start_time
            total_time += elapsed

            all_outputs.extend(batch_outputs)

        speed = len(prompts) / total_time if total_time > 0 else 0
        logger.debug(f"[SGLangWorkerActor] Logprob computation speed: {speed:.2f} samples/s")

        # Extract log probs with mean and min
        results = []
        for outputs, ans_start in zip(all_outputs, answer_starts):
            logprobs = self._extract_answer_logprobs(outputs, ans_start)
            if logprobs:
                results.append({
                    "mean_log_prob": sum(logprobs) / len(logprobs),
                    "min_log_prob": min(logprobs),
                    "token_log_probs": logprobs,
                })
            else:
                results.append({
                    "mean_log_prob": -100.0,
                    "min_log_prob": -100.0,
                    "token_log_probs": [],
                })

        return results

    def _extract_answer_logprobs(self, outputs, answer_start_idx: int) -> List[float]:
        """Extract log probabilities for the answer portion from SGLang output.
        
        SGLang output format:
        - outputs is a dict (or list containing a dict)
        - meta_info contains 'input_token_logprobs' which is a list of tuples:
          [(logprob, token_id, token_text), ...]
        - First token's logprob may be None
        """
        try:
            # Handle both dict and list outputs
            if isinstance(outputs, dict):
                output = outputs
            elif isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[0]
            else:
                logger.info(f"[SGLangWorkerActor] Empty or invalid outputs: {type(outputs)}")
                return []

            if not isinstance(output, dict):
                logger.info(f"[SGLangWorkerActor] output is not a dict: {type(output)}")
                return []

            meta_info = output.get("meta_info", {})
            
            # Get input_token_logprobs (this is the correct field name in SGLang)
            input_token_logprobs = meta_info.get("input_token_logprobs", [])
            
            if not input_token_logprobs:
                logger.info(f"[SGLangWorkerActor] No input_token_logprobs found. meta_info keys: {list(meta_info.keys())}")
                return []

            # Extract logprobs from tuples (logprob, token_id, token_text)
            # Note: Since we set logprob_start_len=ans_start in generate(),
            # input_token_logprobs already contains ONLY answer portion logprobs
            log_probs = []
            for token_tuple in input_token_logprobs:
                if isinstance(token_tuple, tuple) and len(token_tuple) >= 2:
                    logprob = token_tuple[0]
                    if logprob is not None:
                        log_probs.append(float(logprob))
                elif isinstance(token_tuple, (int, float)):
                    # Direct value (shouldn't happen with SGLang, but handle anyway)
                    log_probs.append(float(token_tuple))

            if log_probs:
                logger.info(f"[SGLangWorkerActor] Extracted {len(log_probs)} log_probs, mean={sum(log_probs)/len(log_probs):.4f}")
                return log_probs
            else:
                logger.info(f"[SGLangWorkerActor] No valid logprobs extracted")
                return [-100.0]

        except Exception as e:
            logger.warning(f"[SGLangWorkerActor] Failed to extract logprobs: {e}")
            import traceback
            logger.warning(f"[SGLangWorkerActor] Traceback: {traceback.format_exc()}")
            return [-100.0]

    def _ensure_event_loop(self):
        """Ensure an event loop exists for the current thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            if not hasattr(self, '_loop') or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def release_memory(self):
        """
        释放 SGLang 占用的 GPU 显存（保留权重）。
        使用 release_memory_occupation() 释放 KV Cache
        
        注意：当 enable_memory_saver=False 时，此方法不执行任何操作
        """
        # 当 enable_memory_saver=False 时，不需要释放内存
        # 这是因为 Student Model 的 SGLang 实例每次都是新建并很快销毁
        pass

    def resume_memory(self):
        """
        恢复 SGLang 的 GPU 显存占用（恢复 KV Cache）。
        使用 resume_memory_occupation() 恢复 KV Cache
        
        注意：当 enable_memory_saver=False 时，此方法不执行任何操作
        """
        # 当 enable_memory_saver=False 时，不需要恢复内存
        # 这是因为 Student Model 的 SGLang 实例每次都是新建并很快销毁
        pass

    def shutdown(self):
        """Shutdown the SGLang engine and release all resources."""
        try:
            if hasattr(self.engine, "shutdown"):
                self.engine.shutdown()
            del self.engine
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Engine shutdown complete")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Error during shutdown: {e}")


class SGLangStudentRewardClient:
    """
    Multi-GPU SGLang-based Student Reward Client using Ray actors.

    Supports two modes:
    1. TP=1 (Data Parallelism): Each GPU runs a separate SGLang instance
       - num_actors = num_gpus
       - Each actor processes a portion of the batch independently

    2. TP>1 (Tensor Parallelism): Multiple GPUs share one SGLang instance
       - num_actors = num_gpus / tensor_parallel_size
       - Model weights are sharded across GPUs, reducing per-GPU memory
       - Good for large models that don't fit on single GPU
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,  # TP size per SGLang instance
        mem_fraction_static: float = 0.85,
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,
        logprob_last_n_tokens: Optional[int] = None,  # 只计算最后 N 个 token 的 logprobs
    ):
        """
        Initialize Multi-GPU SGLang Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            tensor_parallel_size: Tensor parallelism size per SGLang instance.
                - TP=1: Each GPU runs independent SGLang (Data Parallelism)
                - TP=2: Every 2 GPUs share one SGLang (model sharded)
                - TP=4: Every 4 GPUs share one SGLang
            mem_fraction_static: Fraction of GPU memory for SGLang static allocation
            dtype: Data type for the model
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
            logprob_last_n_tokens: 只计算 answer 部分最后 N 个 token 的 logprobs，节省显存
        """

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.logprob_last_n_tokens = logprob_last_n_tokens

        # Auto-detect devices if not specified
        if devices is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
        self.devices = devices
        self.num_gpus = len(devices)
        self.gpu_ids = [int(d.split(":")[-1]) for d in devices]

        # Calculate number of actors based on TP size
        # num_actors = num_gpus / tensor_parallel_size
        assert self.num_gpus >= tensor_parallel_size, \
            f"num_gpus ({self.num_gpus}) must be >= tensor_parallel_size ({tensor_parallel_size})"
        assert self.num_gpus % tensor_parallel_size == 0, \
            f"num_gpus ({self.num_gpus}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
        self.num_actors = self.num_gpus // tensor_parallel_size

        self.workers: List[ray.actor.ActorHandle] = []
        self.tokenizer = None
        self._initialized = False

    def cleanup(self):
        """Cleanup Ray SGLang actors to release GPU memory."""
        import subprocess
        import time

        if not self._initialized:
            return

        def get_nvidia_smi():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                lines = result.stdout.strip().split('\n')
                output = []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = parts[0].strip()
                        used_mb = int(parts[1].strip())
                        used_gb = used_mb / 1024
                        output.append(f"GPU {gpu_idx}: {used_gb:.1f}GB")
                return ", ".join(output)
            except:
                return "nvidia-smi unavailable"

        # Shutdown all workers gracefully
        for i, worker in enumerate(self.workers):
            try:
                ray.get(worker.shutdown.remote())
            except Exception as e:
                logger.warning(f"[SGLangStudentReward] Error shutting down worker {i}: {e}")

        # Kill all Ray actors to release GPU memory
        for i, worker in enumerate(self.workers):
            try:
                ray.kill(worker)
            except Exception as e:
                logger.warning(f"[SGLangStudentReward] Error killing worker {i}: {e}")

        self.workers = []
        self._initialized = False

        torch.cuda.empty_cache()
        logger.info(f"[SGLangStudentReward] Cleanup complete. GPU memory: {get_nvidia_smi()}")

    def initialize(self):
        """Lazy initialization of Ray SGLang actors."""
        if self._initialized:
            return

        # Initialize tokenizer (shared, for prompt building)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Compute logical GPU IDs from CUDA_VISIBLE_DEVICES
        
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            # Parse physical GPU IDs from CUDA_VISIBLE_DEVICES
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]

            # Map physical GPU IDs to logical indices
            physical_to_logical = {gpu: idx for idx, gpu in enumerate(available_gpus)}
        else:
            # No CUDA_VISIBLE_DEVICES set, assume physical == logical
            physical_to_logical = {gpu: gpu for gpu in self.gpu_ids}
        
        # Group GPUs by tensor_parallel_size
        # e.g., gpu_ids=[0,1,2,3], TP=2 → [[0,1], [2,3]]
        gpu_groups = []
        for i in range(0, self.num_gpus, self.tensor_parallel_size):
            group = self.gpu_ids[i : i + self.tensor_parallel_size]
            gpu_groups.append(group)

        # Create Ray actors, one per GPU group
        for group_idx, gpu_group in enumerate(gpu_groups):
            # Compute logical GPU IDs for this group
            logical_gpu_ids = [physical_to_logical.get(gpu, 0) for gpu in gpu_group]
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = SGLangWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_ids=gpu_group,
                logical_gpu_ids=logical_gpu_ids,
                tensor_parallel_size=self.tensor_parallel_size,
                mem_fraction_static=self.mem_fraction_static,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
                logprob_last_n_tokens=self.logprob_last_n_tokens,
            )
            self.workers.append(worker)

        # Initialize workers with a test call
        init_futures = [w.compute_logprobs.remote(["test"], [0]) for w in self.workers]
        for i, future in enumerate(init_futures):
            try:
                result = ray.get(future, timeout=120)  # 120 second timeout for SGLang
                logger.info(f"[SGLangStudentReward] Worker {i} initialized successfully")
            except Exception as e:
                logger.error(f"[SGLangStudentReward] Worker {i} init FAILED: {e}")
                raise

        self._initialized = True
        logger.info(f"[SGLangStudentReward] All {len(self.workers)} workers initialized")

    def _build_prompt(self, question: str, cot: str, answer: str) -> tuple[str, int]:
        """
        Build the full prompt and compute the answer start position.

        Returns:
            tuple: (full_prompt, answer_start_token_idx)
        """
        # Build full prompt using chat template
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer, 'reasoning_content': cot}
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Build prefix to find where answer starts
        prefix_messages = [{'role': 'user', 'content': question}]
        prefix_base = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True
        )
        prefix_with_think = prefix_base + f"<think>\n{cot}\n</think>\n\n"

        # Tokenize to get positions
        prefix_token_ids = self.tokenizer.encode(prefix_with_think, add_special_tokens=False)
        answer_start_idx = len(prefix_token_ids)

        return full_text, answer_start_idx

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
        alpha_cog: float = 0.01,
        **kwargs,  # Accept extra args for compatibility
    ) -> Dict[str, Any]:
        """
        Compute cognitive rewards for a batch using multi-GPU SGLang Ray actors.

        Each Ray actor runs SGLang on a separate GPU. We split the batch across
        all actors for true data parallelism.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers
            original_cots: Optional list of original CoTs
            alpha_cog: Weight for min log prob term

        Returns:
            Dict containing rewards and log probs
        """
        import time

        self.initialize()

        batch_size = len(questions)
        start_time = time.time()

        # Phase 1: Build all prompts (CPU)

        rewrite_prompts = []
        rewrite_ans_starts = []
        original_prompts = []
        original_ans_starts = []
        original_indices = []  # Track which indices have original_cot

        for i in range(batch_size):
            # Rewrite prompt
            rewrite_prompt, rewrite_ans_start = self._build_prompt(questions[i], rewrite_cots[i], answers[i])
            rewrite_prompts.append(rewrite_prompt)
            rewrite_ans_starts.append(rewrite_ans_start)

            # Original prompt (if provided)
            if original_cots and i < len(original_cots) and original_cots[i]:
                orig_prompt, orig_ans_start = self._build_prompt(questions[i], original_cots[i], answers[i])
                original_prompts.append(orig_prompt)
                original_ans_starts.append(orig_ans_start)
                original_indices.append(i)

        prep_time = time.time() - start_time

        # Phase 2: Distribute work to Ray actors
        inference_start = time.time()

        # Split rewrite prompts across actors (not GPUs, since TP>1 means multiple GPUs per actor)
        rewrite_chunks = self._split_into_chunks(rewrite_prompts, self.num_actors)
        rewrite_ans_start_chunks = self._split_into_chunks(rewrite_ans_starts, self.num_actors)

        # Submit rewrite tasks to all workers in parallel
        rewrite_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(rewrite_chunks, rewrite_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                rewrite_futures.append((worker_idx, len(prompts_chunk), future))

        # Split original prompts across actors (if any)
        original_futures = []
        if original_prompts:
            original_chunks = self._split_into_chunks(original_prompts, self.num_actors)
            original_ans_start_chunks = self._split_into_chunks(original_ans_starts, self.num_actors)

            for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(original_chunks, original_ans_start_chunks)):
                if prompts_chunk:
                    future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                    original_futures.append((worker_idx, len(prompts_chunk), future))

        # Gather rewrite results
        rewrite_results = [None] * batch_size
        for worker_idx, chunk_size, future in rewrite_futures:
            get_start = time.time()
            chunk_results = ray.get(future)
            get_elapsed = time.time() - get_start
            # Place results back in original order
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_results[start_idx + i] = result

        # Gather original results
        original_results_map = {}
        if original_futures:
            for worker_idx, chunk_size, future in original_futures:
                get_start = time.time()
                chunk_results = ray.get(future)
                get_elapsed = time.time() - get_start
                start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
                for i, result in enumerate(chunk_results):
                    original_idx = original_indices[start_idx + i]
                    original_results_map[original_idx] = result

        # Phase 3: Compute rewards
        rewards = []
        rewrite_mean_log_probs = []
        original_mean_log_probs = []
        all_token_log_probs = []
        all_token_ids = []

        for i in range(batch_size):
            # Get rewrite result
            rewrite_result = rewrite_results[i]
            rewrite_mean = rewrite_result["mean_log_prob"]
            rewrite_min = rewrite_result["min_log_prob"]
            rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min

            # Get original result (if available)
            if i in original_results_map:
                original_result = original_results_map[i]
                original_mean = original_result["mean_log_prob"]
                original_min = original_result["min_log_prob"]
                original_answer_log_prob = original_mean + alpha_cog * original_min
            else:
                original_mean = -100.0
                original_answer_log_prob = -100.0

            # ===== R_cog 计算公式 =====
            # R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
            # 其中 answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
            #
            # 这个差值公式鼓励 rewrite 后的 CoT 比 original CoT 更能帮助 Student Model 预测答案
            # 如果 rewrite 后答案的 log prob 更高，说明 rewrite 质量更好，reward 为正
            # 如果 rewrite 后答案的 log prob 更低，说明 rewrite 质量更差，reward 为负
            reward = rewrite_answer_log_prob - original_answer_log_prob

            rewards.append(reward)
            rewrite_mean_log_probs.append(rewrite_mean)
            original_mean_log_probs.append(original_mean)
            all_token_log_probs.append(rewrite_result.get("token_log_probs", []))
            all_token_ids.append([])  # Token IDs not needed for reward

        return {
            "rewards": rewards,
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "rewrite_token_log_probs": all_token_log_probs,
            "rewrite_token_ids": all_token_ids,
            "log_probs": rewrite_mean_log_probs,
            "token_log_probs": all_token_log_probs,
            "token_ids": all_token_ids,
        }

    def release_memory(self):
        """
        释放所有 worker 的 KV Cache，保留权重。
        用于在 reward 计算前释放显存给其他模型使用。
        """
        if not self._initialized:
            return
        
        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangStudentReward] Released memory for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to release memory: {e}")

    def resume_memory(self):
        """
        恢复所有 worker 的 KV Cache。
        用于在 reward 计算后恢复 SGLang 的完整功能。
        """
        if not self._initialized:
            return
        
        try:
            futures = [w.resume_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info("[SGLangStudentReward] Resumed memory for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to resume memory: {e}")

    def _split_into_chunks(self, items: List, num_chunks: int) -> List[List]:
        """Split items into roughly equal chunks."""
        if num_chunks <= 0:
            return [items]
        chunk_size = max(1, len(items) // num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            if i == num_chunks - 1:
                end = len(items)
            else:
                end = min(start + chunk_size, len(items))
            chunks.append(items[start:end])
        return chunks
