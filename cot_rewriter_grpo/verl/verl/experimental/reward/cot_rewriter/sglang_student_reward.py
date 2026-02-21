"""
Student Reward (Cognitive Reward) Module - SGLang Version

Computes R_cog = -PPL(Q, Rewrite_CoT, A) - (-PPL(Q, Original_CoT, A))
            = PPL(original) - PPL(rewrite)
using SGLang for fast log prob computation.

其中 PPL = exp(-mean(token_log_probs))

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
(lower perplexity) compared to the original CoT.

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
        base_port: int = 30000,  # Base port for this worker (different per worker to avoid conflicts)
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
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
        self.ppl_on_full_response = ppl_on_full_response
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_norm_sigmoid_temperature
        self._base_port = base_port  # Store base port for SGLang Engine
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

        # Save initialization parameters for lazy engine creation
        self._model_path = model_path
        self._tensor_parallel_size = tensor_parallel_size
        self._mem_fraction_static = mem_fraction_static
        self._dtype = dtype
        self._micro_batch_size = micro_batch_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Engine will be created lazily in _init_engine() to avoid port conflicts
        self.engine = None
        self._engine_initialized = False

        # Track memory state for release/resume
        self._memory_released = False

        logger.info(f"[SGLangWorkerActor-{gpu_ids}] Actor created, engine will be initialized on first use")

    def _init_engine(self):
        """
        Initialize SGLang Engine on demand.

        This is called separately from __init__ to avoid port conflicts when
        multiple workers are created simultaneously. The engine should be
        initialized serially (one worker at a time) to prevent race conditions.
        """
        if self._engine_initialized:
            return

        import time
        import sglang as sgl

        llm_start = time.time()

        # Initialize SGLang Engine
        # Using enable_memory_saver=True to support memory release/resume mechanism
        # This avoids port conflicts from cleanup/recreate cycles
        #
        # IMPORTANT: Each worker uses a different base_port to avoid port conflicts
        # during parallel initialization. SGLang uses port + random.randint(100, 1000)
        # for inter-process communication, so we use non-overlapping port ranges:
        # - Worker 0: 30000 + random[100, 1000]  -> [30100, 31000)
        # - Worker 1: 31000 + random[100, 1000]  -> [31100, 32000)
        # - Worker N: (30000 + N*1000) + random[100, 1000) -> [30000+N*1000+100, 30000+N*1000+1000)
        self.engine = sgl.Engine(
            model_path=self._model_path,
            tp_size=self._tensor_parallel_size,
            mem_fraction_static=self._mem_fraction_static,
            dtype=self._dtype,
            trust_remote_code=True,
            enable_memory_saver=True,
            port=self._base_port,  # Use unique base port per worker
        )

        # CRITICAL: Re-create event loop after Engine creation
        # because sglang forces uvloop policy which clears the existing event loop
        # This must be done before any generate() calls
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Event loop re-created after Engine init")

        llm_elapsed = time.time() - llm_start
        self.micro_batch_size = self._micro_batch_size
        self._engine_initialized = True

        logger.info(f"[SGLangWorkerActor-{self.gpu_ids}] Engine created in {llm_elapsed:.2f}s")

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

        # Initialize engine on first use (lazy initialization to avoid port conflicts)
        self._init_engine()

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
                # IMPORTANT: Set logprob_start_len based on ppl_on_full_response
                # - If ppl_on_full_response=False (default): logprob_start_len=ans_start
                #   SGLang's input_token_logprobs only includes tokens from logprob_start_len onwards
                #   We only get answer tokens, not question and CoT tokens
                # - If ppl_on_full_response=True: logprob_start_len=0
                #   We get logprobs for the entire response (Q+CoT+A)
                logprob_start_len = 0 if self.ppl_on_full_response else ans_start

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
        完全销毁 SGLang Engine，释放所有 GPU 显存。

        不使用 release_memory_occupation()（有 CUDA 错误 bug），
        而是直接 shutdown 引擎，释放所有资源。
        下次使用时通过 _init_engine() 重新创建引擎。
        """
        # 完全 shutdown 引擎释放所有显存，而不是尝试释放 KV Cache
        # 因为 release_memory_occupation() 有 bug：RuntimeError: CUDA error: invalid argument
        # 下次使用时通过 _init_engine() 重新创建引擎
        try:
            if self.engine is not None:
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
                del self.engine
                self.engine = None
                self._engine_initialized = False
                self._memory_released = False
                logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Engine destroyed, all GPU memory released")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Error during engine shutdown: {e}")
            self.engine = None
            self._engine_initialized = False
            self._memory_released = False


    def resume_memory(self):
        """
        恢复 SGLang 的 GPU 显存占用（恢复 KV Cache）。
        使用 resume_memory_occupation() 恢复 KV Cache

        在下次 reward 计算前调用，恢复 SGLang 的完整功能。

        注意：只在已 release 过的情况下才调用 resume，
        否则 SGLang 会报错 "Cannot resume allocation that is not paused"
        """
        if not self._memory_released:
            # 首次调用或未 release 过，不需要 resume
            logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Memory not released, skipping resume")
            return

        try:
            if hasattr(self.engine, "resume_memory_occupation"):
                self.engine.resume_memory_occupation()
                self._memory_released = False
                logger.debug(f"[SGLangWorkerActor-{self.gpu_ids}] Resumed KV Cache memory")
            else:
                logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Engine does not support resume_memory_occupation")
        except Exception as e:
            logger.warning(f"[SGLangWorkerActor-{self.gpu_ids}] Failed to resume memory: {e}")

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
        reward_mode: str = "ppl",  # "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
        ppl_ratio_ceiling: float = 0.95,  # Ceiling for ppl_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_ratio_ceiling: float = 0.95,  # Ceiling for cot_effect_ratio mode: R_cog = min(ratio, ceiling)
        cot_effect_epsilon: float = 0.01,  # Epsilon for cot_effect_delta_sigmoid mode: avoids division by zero
        cot_effect_temperature: float = 2.0,  # Temperature for cot_effect_delta_sigmoid mode: controls sigmoid steepness
        ppl_delta_temperature: float = 1.0,  # Temperature for ppl_delta_norm_sigmoid mode sigmoid (after group normalization)
        use_reward_delta: bool = True,  # If True (default), R_cog = rewrite - original (delta).
                                      # If False, R_cog = rewrite only (absolute value).
        reward_only_use_logprob: bool = None,  # Deprecated: use use_reward_delta instead
        ppl_on_full_response: bool = False,  # If True, compute PPL on full response (Q+CoT+A)
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
            reward_mode: Reward calculation mode - "ppl", "ppl_delta", "ppl_delta_norm_sigmoid", "ppl_ratio", "cot_effect_ratio", "cot_effect_delta_sigmoid", or "logprob"
                - "ppl": R_cog = avg_prob(rewrite) = 1 / PPL(rewrite) (absolute quality)
                - "ppl_delta": R_cog = max(0, avg_prob(rewrite) - avg_prob(original)) (delta with negative clamp only)
                - "ppl_delta_norm_sigmoid": R_cog = sigmoid(normalized_delta / temperature) with group normalization
                - "ppl_ratio": R_cog = min(ppl_original / ppl_rewrite, ppl_ratio_ceiling)
                - "cot_effect_ratio": R_cog = min(cot_effect(rewrite) / cot_effect(original), cot_effect_ratio_ceiling)
                - "cot_effect_delta_sigmoid": R_cog = sigmoid((avg_prob(rewrite) - avg_prob(original)) / (|cot_effect(original)| + epsilon) / temperature)
                - "logprob": R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
            ppl_ratio_ceiling: Ceiling for ppl_ratio mode (default: 0.95).
                When ppl_ratio mode: R_cog = min(ppl_original / ppl_rewrite, ppl_ratio_ceiling).
            cot_effect_ratio_ceiling: Ceiling for cot_effect_ratio mode (default: 0.95).
                When cot_effect_ratio mode: R_cog = min(cot_effect(rewrite) / cot_effect(original), cot_effect_ratio_ceiling).
                where cot_effect(cot) = 1/ppl(Q,CoT,A) - 1/ppl(Q,A) = avg_prob(Q,CoT,A) - avg_prob(Q,A)
            use_reward_delta: If True (default), R_cog = value(rewrite) - value(original) (delta).
                If False, R_cog = value(rewrite) only (absolute value).
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead
            ppl_on_full_response: If True, compute PPL on full response (Q+CoT+A).
                If False (default), compute PPL on answer portion only.
        """

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.logprob_last_n_tokens = logprob_last_n_tokens
        self.reward_mode = reward_mode
        self.ppl_ratio_ceiling = ppl_ratio_ceiling
        self.cot_effect_ratio_ceiling = cot_effect_ratio_ceiling
        self.cot_effect_epsilon = cot_effect_epsilon
        self.cot_effect_temperature = cot_effect_temperature
        self.ppl_delta_temperature = ppl_delta_norm_sigmoid_temperature
        self.ppl_on_full_response = ppl_on_full_response

        # Handle reward_only_use_logprob deprecation and map to use_reward_delta
        # reward_only_use_logprob has INVERTED logic compared to use_reward_delta
        # - reward_only_use_logprob=True means absolute value (use_reward_delta=False)
        # - reward_only_use_logprob=False means delta (use_reward_delta=True)
        if reward_only_use_logprob is not None:
            import warnings
            warnings.warn(
                "[DEPRECATION] reward_only_use_logprob is deprecated. Use use_reward_delta instead. "
                "Note: reward_only_use_logprob has inverted logic (True=absolute, False=delta). "
                "Setting use_reward_delta accordingly.",
                DeprecationWarning,
                stacklevel=2
            )
            self.use_reward_delta = not reward_only_use_logprob  # Invert the logic!
        else:
            self.use_reward_delta = use_reward_delta
        self.reward_only_use_logprob = reward_only_use_logprob  # Keep for logging

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
        """
        释放所有 workers 的 KV Cache，保留权重和 Engine 实例。

        使用 release_memory_occupation() 释放 KV Cache，
        避免了 shutdown/recreate 导致的端口冲突问题。
        """
        import subprocess
        import gc

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

        # 调用每个 worker 的 release_memory() 方法
        # 这会调用 engine.release_memory_occupation() 释放 KV Cache
        try:
            futures = [w.release_memory.remote() for w in self.workers]
            ray.get(futures, timeout=60)
            logger.info(f"[SGLangStudentReward] Released KV Cache for all workers")
        except Exception as e:
            logger.warning(f"[SGLangStudentReward] Failed to release memory: {e}")

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        gc.collect()

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
        # Each worker gets a unique base_port to avoid port conflicts during parallel initialization
        # SGLang uses port + random.randint(100, 1000) for inter-process communication,
        # so we use non-overlapping port ranges with 1000-port spacing:
        # - Worker 0: 30000 + random[100, 1000)  -> [30100, 31000)
        # - Worker 1: 31000 + random[100, 1000)  -> [31100, 32000)
        # - Worker N: (30000 + N*1000) + random[100, 1000)
        base_port_start = 30000
        port_spacing = 1000  # Large enough to avoid overlap with random[100, 1000)
        max_port_attempts = 100  # Max attempts to find an available port

        def is_port_available(port: int) -> bool:
            """Check if a port is available for binding."""
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                    return True
            except OSError:
                return False

        # Find available ports for all workers
        assigned_ports = []
        for group_idx in range(len(gpu_groups)):
            preferred_port = base_port_start + group_idx * port_spacing
            port = preferred_port
            attempts = 0

            # Try to find an available port, starting from preferred_port
            while not is_port_available(port) and attempts < max_port_attempts:
                attempts += 1
                # Try next port with port_spacing
                port = base_port_start + (group_idx + attempts * len(gpu_groups)) * port_spacing

            if not is_port_available(port):
                # Fallback: try any port in a wider range
                for fallback_attempt in range(max_port_attempts):
                    port = base_port_start + group_idx * port_spacing + fallback_attempt * 100
                    if is_port_available(port):
                        break
                else:
                    raise RuntimeError(f"Cannot find available port for worker {group_idx} after {max_port_attempts} attempts")

            assigned_ports.append(port)
            logger.info(f"[SGLangStudentReward] Worker {group_idx}: using port {port} (preferred: {preferred_port})")

        for group_idx, gpu_group in enumerate(gpu_groups):
            # Compute logical GPU IDs for this group
            logical_gpu_ids = [physical_to_logical.get(gpu, 0) for gpu in gpu_group]
            # Assign unique base_port for each worker
            base_port = assigned_ports[group_idx]
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
                base_port=base_port,  # Unique base port per worker
                reward_mode=self.reward_mode,
                ppl_ratio_ceiling=self.ppl_ratio_ceiling,
                cot_effect_ratio_ceiling=self.cot_effect_ratio_ceiling,
                cot_effect_epsilon=self.cot_effect_epsilon,
                cot_effect_temperature=self.cot_effect_temperature,
                ppl_delta_norm_sigmoid_temperature=self.ppl_delta_temperature,
                ppl_on_full_response=self.ppl_on_full_response,
            )
            self.workers.append(worker)
            logger.info(f"[SGLangStudentReward] Created worker {group_idx} on GPUs {gpu_group} with base_port={base_port}")

        # IMPORTANT: Initialize workers IN PARALLEL using unique ports
        # Each worker has a different base_port, so their random port ranges don't overlap:
        # - Worker 0 uses ports in [30100, 31000)
        # - Worker 1 uses ports in [31100, 32000)
        # - etc.
        logger.info(f"[SGLangStudentReward] Initializing {len(self.workers)} workers IN PARALLEL with unique ports...")
        init_futures = []
        for i, worker in enumerate(self.workers):
            logger.info(f"[SGLangStudentReward] Submitting init for worker {i}/{len(self.workers)} on GPUs {gpu_groups[i]} (base_port={assigned_ports[i]})...")
            init_futures.append(worker.compute_logprobs.remote(["test"], [0]))

        # Wait for all workers to complete initialization in parallel
        try:
            results = ray.get(init_futures, timeout=180)  # 3 min timeout for all workers
            logger.info(f"[SGLangStudentReward] All {len(self.workers)} workers initialized successfully in parallel")
        except Exception as e:
            logger.error(f"[SGLangStudentReward] Parallel init FAILED: {e}")
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

        # Compute answer start position using two apply_chat_template calls
        # 方法：分别用 question+cot+dummy_answer 和 question+cot+answer 进行两次 tokenize
        # answer_start_idx 就是第一次（用 dummy_answer）tokenize 后的长度
        #
        # 原理：无论 answer 是什么，chat template 在  后面的格式是固定的
        # 所以 dummy_answer 和 answer 的 token 序列只在 answer 内容处不同
        # 第一次 tokenize 的长度就是 answer 开始的位置

        # 第一次：用 dummy_answer (一个字符，如 "X")
        dummy_answer = "X"
        messages_with_dummy = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': dummy_answer, 'reasoning_content': cot}
        ]
        text_with_dummy = self.tokenizer.apply_chat_template(
            messages_with_dummy, tokenize=False, add_generation_prompt=False
        )
        tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

        # 找到 dummy_answer ("X") 在 tokens 中的位置
        # 这个位置就是 answer 开始的地方
        dummy_token_id = self.tokenizer.encode(dummy_answer, add_special_tokens=False)[0]
        # 从后向前找 dummy_token_id（因为可能有重复）
        for i in range(len(tokens_with_dummy) - 1, -1, -1):
            if tokens_with_dummy[i] == dummy_token_id:
                answer_start_idx = i
                break

        return full_text, answer_start_idx

    def _build_prompt_without_cot(self, question: str, answer: str) -> tuple[str, int]:
        """
        Build prompt without CoT for computing R_useful baseline.

        This builds a prompt of the form:
        Question + Answer (without any CoT/reasoning)

        Returns:
            tuple: (full_prompt, answer_start_token_idx)
        """
        # Build prompt WITHOUT reasoning_content
        # This represents the baseline: how well does the model predict Answer given only Question
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}  # No reasoning_content
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Find where answer starts (after the question and assistant header)
        # Use the same dummy_answer technique
        dummy_answer = "X"
        messages_with_dummy = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': dummy_answer}  # No reasoning_content
        ]
        text_with_dummy = self.tokenizer.apply_chat_template(
            messages_with_dummy, tokenize=False, add_generation_prompt=False
        )
        tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

        dummy_token_id = self.tokenizer.encode(dummy_answer, add_special_tokens=False)[0]
        for i in range(len(tokens_with_dummy) - 1, -1, -1):
            if tokens_with_dummy[i] == dummy_token_id:
                answer_start_idx = i
                break

        return full_text, answer_start_idx

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
        alpha_cog: float = 0.01,
        use_reward_delta: Optional[bool] = None,
        reward_only_use_logprob: Optional[bool] = None,
        save_prompts: bool = False,  # Whether to save prompts for debugging
        **kwargs,  # Accept extra args for compatibility
    ) -> Dict[str, Any]:
        """
        Compute cognitive rewards for a batch using multi-GPU SGLang Ray actors.

        Each Ray actor runs SGLang on a separate GPU. We split the batch across
        all actors for true data parallelism.

        Computes two reward components:
        1. R_cog = value(rewrite) - value(original) (if use_reward_delta=True)
                  OR R_cog = value(rewrite) only (if use_reward_delta=False)
        2. R_useful = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, A)

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers
            original_cots: Optional list of original CoTs
            alpha_cog: Weight for min log prob term
            use_reward_delta: If True, compute delta (rewrite - original).
                If False, use absolute value. If None, use self.use_reward_delta.
            reward_only_use_logprob: (Deprecated) Use use_reward_delta instead

        Returns:
            Dict containing rewards and log probs including R_cog and R_useful
        """
        # Use instance value if not provided in call
        # Handle both new and old parameter names
        if use_reward_delta is None and reward_only_use_logprob is None:
            use_reward_delta = self.use_reward_delta
        elif reward_only_use_logprob is not None:
            # Old parameter provided: invert logic (True=absolute, False=delta)
            import warnings
            warnings.warn(
                "[DEPRECATION] reward_only_use_logprob is deprecated. Use use_reward_delta instead. "
                "Note: reward_only_use_logprob has inverted logic (True=absolute, False=delta).",
                DeprecationWarning,
                stacklevel=2
            )
            use_reward_delta = not reward_only_use_logprob
        else:
            use_reward_delta = use_reward_delta if use_reward_delta is not None else self.use_reward_delta
        import time

        self.initialize()

        # 恢复 KV Cache（如果之前被 release_memory 释放了）
        self.resume_memory()

        batch_size = len(questions)
        start_time = time.time()

        # Phase 1: Build all prompts (CPU)

        rewrite_prompts = []
        rewrite_ans_starts = []
        original_prompts = []
        original_ans_starts = []
        original_indices = []  # Track which indices have original_cot
        # New: Build prompts WITHOUT CoT for R_useful baseline
        no_cot_prompts = []
        no_cot_ans_starts = []

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

            # No-CoT prompt for R_useful baseline (Question + Answer only, no CoT)
            no_cot_prompt, no_cot_ans_start = self._build_prompt_without_cot(questions[i], answers[i])
            no_cot_prompts.append(no_cot_prompt)
            no_cot_ans_starts.append(no_cot_ans_start)

        # Save prompts for debugging if requested
        # Store prompts in dict for later return
        saved_prompts = {}
        if save_prompts:
            saved_prompts["rewrite_prompts"] = rewrite_prompts
            saved_prompts["no_cot_prompts"] = no_cot_prompts
            # For original prompts, map them back to original indices
            original_prompts_full = [None] * batch_size
            for idx, orig_prompt in zip(original_indices, original_prompts):
                original_prompts_full[idx] = orig_prompt
            saved_prompts["original_prompts"] = original_prompts_full

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

        # Split no_cot prompts across actors for R_useful baseline
        no_cot_chunks = self._split_into_chunks(no_cot_prompts, self.num_actors)
        no_cot_ans_start_chunks = self._split_into_chunks(no_cot_ans_starts, self.num_actors)

        no_cot_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(no_cot_chunks, no_cot_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                no_cot_futures.append((worker_idx, len(prompts_chunk), future))

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

        # Gather no_cot results (R_useful baseline)
        no_cot_results = [None] * batch_size
        for worker_idx, chunk_size, future in no_cot_futures:
            get_start = time.time()
            chunk_results = ray.get(future)
            get_elapsed = time.time() - get_start
            start_idx = sum(len(no_cot_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                no_cot_results[start_idx + i] = result

        # Phase 3: Compute rewards
        rewards = []
        useful_rewards = []  # R_useful: how much does CoT help vs baseline
        rewrite_mean_log_probs = []
        original_mean_log_probs = []
        no_cot_mean_log_probs = []  # Baseline: Answer|Question only (no CoT)
        all_token_log_probs = []
        all_token_ids = []

        # Store PPL values for ppl mode (for logging/debugging)
        rewrite_ppls = []
        original_ppls = []
        # Store relative improvement rate (formula 1) as observation value
        relative_improvement_rates = []  # R_cog_formula1 = avg_prob(rewrite)/avg_prob(original) - 1

        for i in range(batch_size):
            # Get rewrite result
            rewrite_result = rewrite_results[i]
            rewrite_mean = rewrite_result["mean_log_prob"]
            rewrite_min = rewrite_result["min_log_prob"]

            # Get no_cot result (baseline without CoT)
            no_cot_result = no_cot_results[i]
            no_cot_mean = no_cot_result["mean_log_prob"]
            no_cot_min = no_cot_result["min_log_prob"]

            # Get original result (if available)
            if i in original_results_map:
                original_result = original_results_map[i]
                original_mean = original_result["mean_log_prob"]
                original_min = original_result["min_log_prob"]
            else:
                original_mean = -100.0
                original_min = -100.0

            # ===== R_cog 计算公式 =====
            # R_cog = value(rewrite) - value(original) (if use_reward_delta=True)
            # OR R_cog = value(rewrite) only (if use_reward_delta=False)
            # 根据 reward_mode 选择计算方式
            if self.reward_mode == "ppl_ratio":
                # PPL Ratio 模式: R_cog = min(ppl_original / ppl_rewrite, ceiling)
                # 其中 ppl = 1 / avg_prob = exp(-mean_log_prob)
                import math
                rewrite_ppl = math.exp(-rewrite_mean)  # PPL(rewrite) = exp(-mean_log_prob)

                # 计算公式1作为观测值（用于日志）
                if original_mean > -100:
                    original_ppl = math.exp(-original_mean)  # PPL(original)
                    if original_ppl < 1e-10:
                        ppl_ratio = 0.0
                    else:
                        ppl_ratio = original_ppl / rewrite_ppl
                    # 应用天花板
                    reward = min(ppl_ratio, self.ppl_ratio_ceiling)
                    relative_improvement_rate = ppl_ratio - 1.0  # 用于日志
                    original_ppls.append(original_ppl)
                else:
                    # 如果没有 original，reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(rewrite_ppl)
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "ppl":
                # PPL 模式: 使用公式2 (绝对质量)
                # R_cog = avg_prob(rewrite) = 1 / PPL(rewrite)
                # 同时计算公式1 (相对改进率) 作为观测值
                # R_cog_formula1 = avg_prob(rewrite) / avg_prob(original) - 1
                #               = PPL(original) / PPL(rewrite) - 1
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob for rewrite

                # 公式2: 绝对质量度量 (实际用于GRPO的reward)
                reward = rewrite_avg_prob

                # 计算公式1作为观测值
                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)
                    if original_avg_prob < 1e-10:
                        relative_improvement_rate = 0.0
                    else:
                        relative_improvement_rate = (rewrite_avg_prob / original_avg_prob) - 1.0
                else:
                    relative_improvement_rate = 0.0
                relative_improvement_rates.append(relative_improvement_rate)

                # Store PPL values for logging (PPL = 1/avg_prob)
                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    original_ppls.append(float('nan'))
            elif self.reward_mode == "cot_effect_ratio":
                # CoT Effect Ratio 模式
                # cot_effect(cot) = 1/ppl(Q,CoT,A) - 1/ppl(Q,A) = avg_prob(Q,CoT,A) - avg_prob(Q,A)
                # R_cog = min(cot_effect(rewrite) / cot_effect(original), ceiling)
                import math

                # avg_prob = exp(mean_log_prob)
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(Q, rewrite_CoT, A)
                no_cot_avg_prob = math.exp(no_cot_mean)     # avg_prob(Q, A) - baseline without CoT

                # CoT effect for rewrite: how much does CoT help vs baseline
                cot_effect_rewrite = rewrite_avg_prob - no_cot_avg_prob

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(Q, original_CoT, A)

                    # CoT effect for original: how much did original CoT help vs baseline
                    cot_effect_original = original_avg_prob - no_cot_avg_prob

                    # Handle edge case: if cot_effect_original is too small or non-positive
                    if cot_effect_original < 1e-10:
                        # Original CoT had little to no effect, return ceiling value (max reward)
                        cot_effect_ratio = self.cot_effect_ratio_ceiling
                        relative_improvement_rate = self.cot_effect_ratio_ceiling - 1.0
                    else:
                        # Normal case: compute the ratio
                        # cot_effect_ratio > 1 means rewrite is more helpful than original
                        cot_effect_ratio = cot_effect_rewrite / cot_effect_original
                        # Apply ceiling
                        cot_effect_ratio = min(cot_effect_ratio, self.cot_effect_ratio_ceiling)
                        relative_improvement_rate = cot_effect_ratio - 1.0

                    reward = cot_effect_ratio
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # No original CoT available, reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "cot_effect_delta_sigmoid":
                # CoT Effect Delta Sigmoid 模式
                #
                # 数学公式:
                #   coe_orig = avg_prob(Q, original, A) - avg_prob(Q, A)
                #   coe_rew  = avg_prob(Q, rewrite, A)  - avg_prob(Q, A)
                #   delta    = avg_prob(rewrite) - avg_prob(original)
                #   delta_clamped = max(0, delta)  # delta <= 0 时视为 0
                #   Δ_norm = delta_clamped / (|coe_orig| + ε)
                #   R_cog = sigmoid(Δ_norm / τ)
                #
                # 其中 ε = cot_effect_epsilon (默认 0.01), τ = cot_effect_temperature (默认 1.0)
                # 奖励值域：[0.5, 1)，与其他 penalty 同量级
                import math

                # avg_prob = exp(mean_log_prob)
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(Q, rewrite_CoT, A)
                no_cot_avg_prob = math.exp(no_cot_mean)     # avg_prob(Q, A) - baseline

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(Q, original_CoT, A)

                    # CoT effect for original
                    cot_effect_original = original_avg_prob - no_cot_avg_prob

                    # 改进量（delta <= 0 时视为 0）
                    delta = rewrite_avg_prob - original_avg_prob
                    delta_clamped = max(0.0, delta)

                    epsilon = getattr(self, 'cot_effect_epsilon', 0.01)
                    temperature = getattr(self, 'cot_effect_temperature', 2.0)

                    normalized_delta = delta_clamped / (abs(cot_effect_original) + epsilon)

                    # Sigmoid 映射到 [0.5, 1)
                    reward = 1.0 / (1.0 + math.exp(-normalized_delta / temperature))
                    relative_improvement_rate = delta  # 用于日志（原始 delta）

                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # 没有 original，reward = 0.5
                    reward = 0.5
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "ppl_delta":
                # PPL Delta 模式 (仅负值截断)
                # delta = avg_prob(rewrite) - avg_prob(original)
                # R_cog = max(0, delta)
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(rewrite) = 1/ppl(rewrite)

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(original) = 1/ppl(original)
                    delta = rewrite_avg_prob - original_avg_prob
                    reward = max(0.0, delta)  # 负值截断
                    relative_improvement_rate = delta  # 用于日志
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # 没有 original，reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            elif self.reward_mode == "ppl_delta_norm_sigmoid":
                # PPL Delta 模式 (带组内归一化 + sigmoid)
                # 原始 delta = avg_prob(rewrite) - avg_prob(original)
                # 归一化: delta_norm = (delta - mean(delta)) / std(delta)
                # Sigmoid: reward = sigmoid(delta_norm / temperature)
                import math
                rewrite_avg_prob = math.exp(rewrite_mean)  # avg_prob(rewrite) = 1/ppl(rewrite)

                if original_mean > -100:
                    original_avg_prob = math.exp(original_mean)  # avg_prob(original) = 1/ppl(original)
                    delta = rewrite_avg_prob - original_avg_prob
                    # 先保存原始 delta，稍后进行组内归一化
                    reward = delta  # 临时值，稍后会被替换
                    relative_improvement_rate = delta  # 用于日志
                    original_ppls.append(1.0 / original_avg_prob if original_avg_prob > 0 else float('inf'))
                else:
                    # 没有 original，reward = 0
                    reward = 0.0
                    relative_improvement_rate = 0.0
                    original_ppls.append(float('nan'))

                rewrite_ppls.append(1.0 / rewrite_avg_prob if rewrite_avg_prob > 0 else float('inf'))
                relative_improvement_rates.append(relative_improvement_rate)
            else:  # "logprob" 模式
                # Logprob 模式: R_cog = answer_log_prob(rewrite) - answer_log_prob(original)
                # 其中 answer_log_prob = mean(token_log_probs) + alpha_cog * min(token_log_probs)
                rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min
                if not use_reward_delta:
                    # When use_reward_delta=False, use only rewrite logprob (no delta)
                    reward = rewrite_answer_log_prob
                else:
                    # Default: compute delta from original
                    original_answer_log_prob = original_mean + alpha_cog * original_min
                    reward = rewrite_answer_log_prob - original_answer_log_prob

            # ===== R_useful 计算公式 =====
            # R_useful = answer_log_prob(Q, Rewrite_CoT, A) - answer_log_prob(Q, A)
            # 这代表改写后CoT的有用程度：相对于没有CoT的情况，有了CoT之后预测答案的能力提升
            # 注意：这里使用 logprob 模式计算，不使用 PPL 模式
            # 因为我们要比较的是 logprob 的差异，而不是 PPL 的差异
            rewrite_answer_log_prob = rewrite_mean + alpha_cog * rewrite_min
            no_cot_answer_log_prob = no_cot_mean + alpha_cog * no_cot_min
            useful_reward = rewrite_answer_log_prob - no_cot_answer_log_prob

            rewards.append(reward)
            useful_rewards.append(useful_reward)
            rewrite_mean_log_probs.append(rewrite_mean)
            original_mean_log_probs.append(original_mean)
            no_cot_mean_log_probs.append(no_cot_mean)
            all_token_log_probs.append(rewrite_result.get("token_log_probs", []))
            all_token_ids.append([])  # Token IDs not needed for reward

        result_dict = {
            "rewards": rewards,  # R_cog (formula 2: avg_prob(rewrite))
            "useful_rewards": useful_rewards,  # R_useful
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "no_cot_mean_log_probs": no_cot_mean_log_probs,  # Baseline (no CoT)
            "rewrite_ppls": rewrite_ppls,  # PPL values for rewrite (ppl mode only)
            "original_ppls": original_ppls,  # PPL values for original (ppl mode only)
            "relative_improvement_rates": relative_improvement_rates,  # Formula 1: avg_prob(rewrite)/avg_prob(original) - 1
            "rewrite_token_log_probs": all_token_log_probs,
            "rewrite_token_ids": all_token_ids,
            "log_probs": rewrite_mean_log_probs,
            "token_log_probs": all_token_log_probs,
            "token_ids": all_token_ids,
        }

        # ===== Post-processing for ppl_delta_norm_sigmoid mode: Group normalization + Sigmoid =====
        if self.reward_mode == "ppl_delta_norm_sigmoid":
            import math
            # Extract raw deltas from rewards (which currently hold delta values)
            raw_deltas = rewards[:]  # Copy

            # Find valid deltas (from samples with original_cot)
            # First apply max(0, delta) to clamp negative values
            valid_deltas_raw = [raw_deltas[i] for i in range(batch_size) if i in original_results_map]
            valid_deltas = [max(0.0, d) for d in valid_deltas_raw]
            valid_indices = [i for i in range(batch_size) if i in original_results_map]

            if valid_deltas:
                # Compute mean and std of clamped deltas
                mean_delta = sum(valid_deltas) / len(valid_deltas)
                variance = sum((d - mean_delta) ** 2 for d in valid_deltas) / len(valid_deltas)
                std_delta = math.sqrt(variance)

                # Avoid division by zero (when all deltas are 0 or very similar)
                if std_delta < 1e-8:
                    std_delta = 1.0

                temperature = self.ppl_delta_temperature

                # Apply group normalization + sigmoid to valid samples
                for i, idx in enumerate(valid_indices):
                    clamped_delta = valid_deltas[i]
                    normalized_delta = (clamped_delta - mean_delta) / std_delta
                    # Sigmoid: reward = 1 / (1 + exp(-x))
                    rewards[idx] = 1.0 / (1.0 + math.exp(-normalized_delta / temperature))

                # For samples without original, set reward to 0.5 (sigmoid of 0)
                for i in range(batch_size):
                    if i not in valid_indices:
                        rewards[i] = 0.5

        # Add saved prompts if requested
        if save_prompts and saved_prompts:
            result_dict["saved_prompts"] = saved_prompts

        return result_dict

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
