"""
Student Reward (Cognitive Reward) Module - vLLM Version

Computes R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
using vLLM for fast log prob computation.

其中 answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
compared to the original CoT.

Uses Ray actors for multi-GPU parallelism. Supports two modes:
1. TP=1: Each GPU runs a separate vLLM instance (Data Parallelism)
2. TP>1: Multiple GPUs share one vLLM instance (Tensor Parallelism)
"""

from typing import Any, Dict, List, Optional
import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger


@ray.remote
class VLLMWorkerActor:
    """
    Ray actor that runs vLLM on one or more GPUs.

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
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,
        max_num_seqs: int = 1,
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
        worker_name = f"VLLMWorker-{gpu_ids}"

        # Ray actor inherits CUDA_VISIBLE_DEVICES from parent process
        parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[{worker_name}] Parent CUDA_VISIBLE_DEVICES={parent_cuda_devices}", flush=True)

        # Set CUDA_VISIBLE_DEVICES to all logical GPUs for this actor
        # e.g., logical_gpu_ids=[0, 1] → "0,1" for TP=2
        cuda_devices_str = ",".join(map(str, logical_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str
        print(f"[{worker_name}] Set CUDA_VISIBLE_DEVICES={cuda_devices_str} (logical GPUs for physical GPUs {gpu_ids})", flush=True)

        # Force vLLM V0 engine to avoid V1 serialization bugs
        os.environ["VLLM_USE_V1"] = "0"

        from vllm import LLM
        from transformers import AutoTokenizer
        import torch

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

        print(f"[{worker_name}] ========== GPU MEMORY CHECK ==========", flush=True)
        print(f"[{worker_name}] BEFORE loading Student vLLM: {get_all_gpus_memory()}", flush=True)
        print(f"[{worker_name}] tensor_parallel_size={tensor_parallel_size}, gpu_memory_utilization={gpu_memory_utilization}", flush=True)
        print(f"[{worker_name}] max_model_len={max_model_len}, micro_batch_size={micro_batch_size}", flush=True)
        print(f"[{worker_name}] Initializing vLLM on GPUs {gpu_ids} with TP={tensor_parallel_size}...", flush=True)

        self._worker_name = worker_name
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu
        self._get_all_gpus_memory = get_all_gpus_memory

        # Log CUDA environment
        print(f"[{worker_name}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[{worker_name}] PyTorch sees {torch.cuda.device_count()} GPU(s)", flush=True)

        print(f"[{worker_name}] Loading tokenizer from {model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[{worker_name}] Tokenizer loaded", flush=True)
        
        print(f"[{worker_name}] Initializing vLLM LLM with: TP={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}, max_len={max_model_len}...", flush=True)
        import time
        llm_start = time.time()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,  # Use actual TP size
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            max_num_seqs=max_num_seqs,
            max_logprobs=1,
        )
        llm_elapsed = time.time() - llm_start
        self.micro_batch_size = micro_batch_size
        print(f"[{worker_name}] vLLM LLM initialized successfully in {llm_elapsed:.2f}s", flush=True)
        print(f"[{worker_name}] nvidia-smi AFTER loading Student vLLM:", flush=True)
        print(f"[{worker_name}] {get_all_gpus_memory()}", flush=True)
        print(f"[{worker_name}] ========================================", flush=True)

    def compute_logprobs(
        self,
        prompts: List[str],
        answer_starts: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Compute log probs for a batch of prompts.

        Uses micro-batching to avoid OOM with long sequences.

        Returns:
            List of dicts, each containing:
                - mean_log_prob: float
                - min_log_prob: float
                - token_log_probs: List[float]
        """
        from vllm import SamplingParams
        import time

        worker_name = self._worker_name
        print(f"[{worker_name}] compute_logprobs START: {len(prompts)} prompts", flush=True)
        
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0,
        )

        # Log GPU memory before generation
        print(f"[{worker_name}] Processing {len(prompts)} prompts (micro_batch_size={self.micro_batch_size})...", flush=True)
        if hasattr(self, '_get_all_gpus_memory'):
            print(f"[{worker_name}] GPU memory BEFORE generate: {self._get_all_gpus_memory()}", flush=True)

        # Micro-batch processing to avoid OOM
        all_outputs = []
        total_time = 0
        micro_batch_size = self.micro_batch_size

        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]

            print(f"[{worker_name}] Processing micro-batch {i//micro_batch_size + 1}/{(len(prompts)-1)//micro_batch_size + 1} ({len(batch_prompts)} prompts)...", flush=True)

            start_time = time.time()
            batch_outputs = self.llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - start_time
            total_time += elapsed

            all_outputs.extend(batch_outputs)

            # Log memory after each micro-batch
            if hasattr(self, '_get_all_gpus_memory'):
                print(f"[{worker_name}] Micro-batch done in {elapsed:.2f}s, GPU: {self._get_all_gpus_memory()}", flush=True)

        speed = len(prompts) / total_time if total_time > 0 else 0
        print(f"[{worker_name}] Done: {len(prompts)} prompts in {total_time:.2f}s ({speed:.1f} samples/s)")

        # Extract log probs with mean and min
        results = []
        for output, ans_start in zip(all_outputs, answer_starts):
            logprobs = self._extract_answer_logprobs(output.prompt_logprobs, ans_start)
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

        print(f"[{worker_name}] compute_logprobs END: returned {len(results)} results", flush=True)
        return results

    def _extract_answer_logprobs(self, prompt_logprobs, answer_start_idx: int) -> List[float]:
        """Extract log probabilities for the answer portion."""
        if not prompt_logprobs:
            return []

        log_probs = []
        for i in range(answer_start_idx, len(prompt_logprobs)):
            token_logprob = prompt_logprobs[i]
            if token_logprob is not None:
                # Get the log prob of the actual token
                if hasattr(token_logprob, 'values'):
                    # vLLM 0.8.x format - values() returns Logprob objects
                    value = list(token_logprob.values())[0]
                    # Extract float from Logprob object if needed
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))
                elif isinstance(token_logprob, dict):
                    value = list(token_logprob.values())[0]
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))
                elif isinstance(token_logprob, (list, tuple)) and len(token_logprob) > 0:
                    value = token_logprob[0]
                    if hasattr(value, 'logprob'):
                        log_probs.append(float(value.logprob))
                    else:
                        log_probs.append(float(value))

        return log_probs if log_probs else [-100.0]


class VLLMStudentRewardClient:
    """
    Multi-GPU vLLM-based Student Reward Client using Ray actors.

    Supports two modes:
    1. TP=1 (Data Parallelism): Each GPU runs a separate vLLM instance
       - num_actors = num_gpus
       - Each actor processes a portion of the batch independently

    2. TP>1 (Tensor Parallelism): Multiple GPUs share one vLLM instance
       - num_actors = num_gpus / tensor_parallel_size
       - Model weights are sharded across GPUs, reducing per-GPU memory
       - Good for large models that don't fit on single GPU
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,  # TP size per vLLM instance
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,
        max_num_seqs: int = 1,
    ):
        """
        Initialize Multi-GPU vLLM Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            tensor_parallel_size: Tensor parallelism size per vLLM instance.
                - TP=1: Each GPU runs independent vLLM (Data Parallelism)
                - TP=2: Every 2 GPUs share one vLLM (model sharded)
                - TP=4: Every 4 GPUs share one vLLM
            gpu_memory_utilization: Fraction of GPU memory to use per GPU
            dtype: Data type for the model
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
            max_num_seqs: Maximum concurrent sequences in vLLM scheduler
        """
        import torch

        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size
        self.max_num_seqs = max_num_seqs

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

        logger.info(f"[VLLMStudentReward] Configuration:")
        logger.info(f"[VLLMStudentReward]   - Total GPUs: {self.num_gpus} ({self.gpu_ids})")
        logger.info(f"[VLLMStudentReward]   - Tensor Parallel Size: {tensor_parallel_size}")
        logger.info(f"[VLLMStudentReward]   - Number of vLLM actors: {self.num_actors}")
        logger.info(f"[VLLMStudentReward]   - Micro batch size per actor: {self.micro_batch_size}")
        logger.info(f"[VLLMStudentReward]   - Max concurrent sequences per vLLM: {self.max_num_seqs}")

    def cleanup(self):
        """Cleanup Ray vLLM actors to release GPU memory."""
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

        logger.info(f"[VLLMStudentReward] Cleaning up {len(self.workers)} vLLM Ray actors...")
        logger.info(f"[VLLMStudentReward] nvidia-smi BEFORE cleanup: {get_nvidia_smi()}")

        # Kill all Ray actors to release GPU memory
        for i, worker in enumerate(self.workers):
            try:
                ray.kill(worker)
                logger.info(f"[VLLMStudentReward] Killed worker {i}")
            except Exception as e:
                logger.warning(f"[VLLMStudentReward] Error killing worker {i}: {e}")

        self.workers = []
        self._initialized = False

        # Force garbage collection and CUDA cache cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for GPU memory to be released
        logger.info(f"[VLLMStudentReward] Waiting for GPU memory release...")
        time.sleep(2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[VLLMStudentReward] nvidia-smi AFTER cleanup: {get_nvidia_smi()}")
        logger.info(f"[VLLMStudentReward] Cleanup completed")

    def initialize(self):
        """Lazy initialization of Ray vLLM actors."""
        if self._initialized:
            return

        from transformers import AutoTokenizer

        logger.info(f"[VLLMStudentReward] Initializing {self.num_actors} vLLM Ray actors (TP={self.tensor_parallel_size})...")

        # Initialize tokenizer (shared, for prompt building)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Compute logical GPU IDs from CUDA_VISIBLE_DEVICES
        import os
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            # Parse physical GPU IDs from CUDA_VISIBLE_DEVICES
            available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            logger.info(f"[VLLMStudentReward] CUDA_VISIBLE_DEVICES={cuda_visible_devices}, available_gpus={available_gpus}")

            # Map physical GPU IDs to logical indices
            physical_to_logical = {gpu: idx for idx, gpu in enumerate(available_gpus)}
        else:
            # No CUDA_VISIBLE_DEVICES set, assume physical == logical
            physical_to_logical = {gpu: gpu for gpu in self.gpu_ids}
            logger.info(f"[VLLMStudentReward] No CUDA_VISIBLE_DEVICES set, using physical GPU IDs directly")

        # Group GPUs by tensor_parallel_size
        # e.g., gpu_ids=[0,1,2,3], TP=2 → [[0,1], [2,3]]
        gpu_groups = []
        for i in range(0, self.num_gpus, self.tensor_parallel_size):
            group = self.gpu_ids[i : i + self.tensor_parallel_size]
            gpu_groups.append(group)

        logger.info(f"[VLLMStudentReward] GPU groups: {gpu_groups}")

        # Create Ray actors, one per GPU group
        for group_idx, gpu_group in enumerate(gpu_groups):
            # Compute logical GPU IDs for this group
            logical_gpu_ids = [physical_to_logical.get(gpu, 0) for gpu in gpu_group]

            logger.info(f"[VLLMStudentReward] Creating actor {group_idx} for physical GPUs {gpu_group} (logical GPUs {logical_gpu_ids})...")

            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = VLLMWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_ids=gpu_group,
                logical_gpu_ids=logical_gpu_ids,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
                max_num_seqs=self.max_num_seqs,
            )
            self.workers.append(worker)

        # Wait for all actors to initialize
        logger.info(f"[VLLMStudentReward] Waiting for {self.num_actors} actors to initialize...")
        init_futures = [w.compute_logprobs.remote(["test"], [0]) for w in self.workers]
        logger.info(f"[VLLMStudentReward] Submitted {len(init_futures)} init tasks, waiting for results...")
        for i, future in enumerate(init_futures):
            logger.info(f"[VLLMStudentReward] Waiting for worker {i} init...")
            try:
                result = ray.get(future, timeout=60)  # 60 second timeout
                logger.info(f"[VLLMStudentReward] Worker {i} init done")
            except Exception as e:
                logger.error(f"[VLLMStudentReward] Worker {i} init FAILED: {e}")
                raise
        logger.info(f"[VLLMStudentReward] All {self.num_actors} actors initialized successfully (TP={self.tensor_parallel_size})")

        self._initialized = True

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
        Compute cognitive rewards for a batch using multi-GPU vLLM Ray actors.

        Each Ray actor runs vLLM on a separate GPU. We split the batch across
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
        logger.info(f"[VLLMStudentReward] Computing rewards for {batch_size} samples on {self.num_actors} actors (TP={self.tensor_parallel_size})")

        start_time = time.time()

        # Phase 1: Build all prompts (CPU)
        logger.info(f"[VLLMStudentReward] Phase 1: Building prompts...")
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
        logger.info(f"[VLLMStudentReward] Phase 1 done: {prep_time:.2f}s")

        # Phase 2: Distribute work to Ray actors
        logger.info(f"[VLLMStudentReward] Phase 2: Distributing work to {self.num_actors} Ray actors...")
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
        logger.info(f"[VLLMStudentReward] Waiting for {len(rewrite_futures)} rewrite tasks...")
        rewrite_results = [None] * batch_size
        for worker_idx, chunk_size, future in rewrite_futures:
            logger.info(f"[VLLMStudentReward] Waiting for rewrite result from worker {worker_idx}...")
            get_start = time.time()
            chunk_results = ray.get(future)
            get_elapsed = time.time() - get_start
            logger.info(f"[VLLMStudentReward] Got rewrite result from worker {worker_idx} in {get_elapsed:.2f}s")
            # Place results back in original order
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_results[start_idx + i] = result

        # Gather original results
        original_results_map = {}
        if original_futures:
            logger.info(f"[VLLMStudentReward] Waiting for {len(original_futures)} original tasks...")
            for worker_idx, chunk_size, future in original_futures:
                logger.info(f"[VLLMStudentReward] Waiting for original result from worker {worker_idx}...")
                get_start = time.time()
                chunk_results = ray.get(future)
                get_elapsed = time.time() - get_start
                logger.info(f"[VLLMStudentReward] Got original result from worker {worker_idx} in {get_elapsed:.2f}s")
                start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
                for i, result in enumerate(chunk_results):
                    original_idx = original_indices[start_idx + i]
                    original_results_map[original_idx] = result

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        logger.info(f"[VLLMStudentReward] Phase 2 done: {inference_time:.2f}s inference, {total_time:.2f}s total")

        # Phase 3: Compute rewards
        logger.info(f"[VLLMStudentReward] Phase 3: Computing rewards with alpha_cog={alpha_cog}...")
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

            # ===== 旧的实现（仅使用 rewrite log prob，不减去 original）=====
            # R_cog = rewrite_log_prob (不减去 original_log_prob)
            # 直接使用 rewrite 的 answer log prob 作为 reward
            # reward = rewrite_answer_log_prob

            rewards.append(reward)
            rewrite_mean_log_probs.append(rewrite_mean)
            original_mean_log_probs.append(original_mean)
            all_token_log_probs.append(rewrite_result.get("token_log_probs", []))
            all_token_ids.append([])  # Token IDs not needed for reward

        logger.info(f"[VLLMStudentReward] All {batch_size} rewards computed")

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
