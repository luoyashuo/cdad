"""
Student Reward (Cognitive Reward) Module - vLLM Version

Computes R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
using vLLM for fast log prob computation.

其中 answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
compared to the original CoT.

Uses Ray actors for multi-GPU parallelism, with each GPU running
a separate vLLM instance.
"""

from typing import Any, Dict, List, Optional
import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger


@ray.remote
class VLLMWorkerActor:
    """
    Ray actor that runs vLLM on a specific GPU.
    Each actor is a separate process with its own vLLM instance.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        logical_gpu_id: int,  # NEW: logical GPU ID within CUDA_VISIBLE_DEVICES
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        micro_batch_size: int = 4,  # NEW: micro batch size to avoid OOM
    ):
        import os
        import subprocess

        # Ray actor inherits CUDA_VISIBLE_DEVICES from parent process
        # e.g., CUDA_VISIBLE_DEVICES="0,5" means GPU 0 -> logical 0, GPU 5 -> logical 1
        # We set CUDA_VISIBLE_DEVICES to the logical GPU ID to isolate this actor
        parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[VLLMWorker-{gpu_id}] Parent CUDA_VISIBLE_DEVICES={parent_cuda_devices}", flush=True)

        # CRITICAL: Set CUDA_VISIBLE_DEVICES to only the logical GPU for this actor
        # This isolates vLLM to a single GPU and avoids interference
        os.environ["CUDA_VISIBLE_DEVICES"] = str(logical_gpu_id)
        print(f"[VLLMWorker-{gpu_id}] Set CUDA_VISIBLE_DEVICES={logical_gpu_id} (logical GPU for physical GPU {gpu_id})", flush=True)

        # Force vLLM V0 engine to avoid V1 serialization bugs
        # (msgspec.ValidationError: Expected `float`, got `str`)
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

        print(f"[VLLMWorker-{gpu_id}] ========== GPU MEMORY CHECK ==========", flush=True)
        mem_before = get_nvidia_smi_memory_for_gpu(gpu_id)
        print(f"[VLLMWorker-{gpu_id}] BEFORE loading Student vLLM: {mem_before}", flush=True)
        print(f"[VLLMWorker-{gpu_id}] gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}, micro_batch_size={micro_batch_size}", flush=True)
        print(f"[VLLMWorker-{gpu_id}] Initializing vLLM on GPU {gpu_id}...", flush=True)

        self.gpu_id = gpu_id
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu  # Save for later use

        # Log CUDA environment
        print(f"[VLLMWorker-{gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[VLLMWorker-{gpu_id}] PyTorch sees {torch.cuda.device_count()} GPU(s)", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.micro_batch_size = micro_batch_size
        print(f"[VLLMWorker-{gpu_id}] vLLM initialized successfully", flush=True)
        print(f"[VLLMWorker-{gpu_id}] nvidia-smi AFTER loading Student vLLM:", flush=True)
        print(f"[VLLMWorker-{gpu_id}] {get_nvidia_smi_memory_for_gpu(gpu_id)}", flush=True)
        print(f"[VLLMWorker-{gpu_id}] ========================================", flush=True)

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

        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0,
        )

        # Log GPU memory before generation
        print(f"[VLLMWorker-{self.gpu_id}] Processing {len(prompts)} prompts (micro_batch_size={self.micro_batch_size})...", flush=True)
        if hasattr(self, '_get_nvidia_smi'):
            print(f"[VLLMWorker-{self.gpu_id}] GPU memory BEFORE generate: {self._get_nvidia_smi(self.gpu_id)}", flush=True)

        # Micro-batch processing to avoid OOM
        all_outputs = []
        total_time = 0
        micro_batch_size = self.micro_batch_size
        
        for i in range(0, len(prompts), micro_batch_size):
            batch_prompts = prompts[i:i + micro_batch_size]
            batch_ans_starts = answer_starts[i:i + micro_batch_size]
            
            print(f"[VLLMWorker-{self.gpu_id}] Processing micro-batch {i//micro_batch_size + 1}/{(len(prompts)-1)//micro_batch_size + 1} ({len(batch_prompts)} prompts)...", flush=True)
            
            start_time = time.time()
            batch_outputs = self.llm.generate(batch_prompts, sampling_params)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            all_outputs.extend(batch_outputs)
            
            # Log memory after each micro-batch
            if hasattr(self, '_get_nvidia_smi'):
                print(f"[VLLMWorker-{self.gpu_id}] Micro-batch done in {elapsed:.2f}s, GPU: {self._get_nvidia_smi(self.gpu_id)}", flush=True)

        speed = len(prompts) / total_time if total_time > 0 else 0
        print(f"[VLLMWorker-{self.gpu_id}] Done: {len(prompts)} prompts in {total_time:.2f}s ({speed:.1f} samples/s)")

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

    Each GPU runs a separate vLLM instance in its own Ray actor process.
    This avoids conflicts with the main Ray cluster and allows true parallelism.

    Benefits:
    1. Each GPU has its own vLLM instance (no TP coordination overhead)
    2. True data parallelism across GPUs
    3. vLLM's efficient batching within each GPU
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,  # Ignored, always 1 per actor
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
        devices: Optional[List[str]] = None,
        micro_batch_size: int = 4,  # NEW: micro batch size per GPU worker
    ):
        """
        Initialize Multi-GPU vLLM Student Reward Client with Ray actors.

        Args:
            model_path: Path to the student model
            tensor_parallel_size: Ignored (each actor uses TP=1)
            gpu_memory_utilization: Fraction of GPU memory to use per actor
            dtype: Data type for the model
            max_model_len: Maximum sequence length
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
            micro_batch_size: Batch size for each micro-batch to avoid OOM
        """
        import torch

        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.micro_batch_size = micro_batch_size

        # Auto-detect devices if not specified
        if devices is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
        self.devices = devices
        self.num_gpus = len(devices)
        self.gpu_ids = [int(d.split(":")[-1]) for d in devices]

        self.workers: List[ray.actor.ActorHandle] = []
        self.tokenizer = None
        self._initialized = False

        logger.info(f"[VLLMStudentReward] Will use {self.num_gpus} Ray actors on GPUs: {self.gpu_ids}")
        logger.info(f"[VLLMStudentReward] Micro batch size per worker: {self.micro_batch_size}")

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

        logger.info(f"[VLLMStudentReward] Initializing {self.num_gpus} vLLM Ray actors...")

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

        # Create Ray actors, one per GPU
        for gpu_id in self.gpu_ids:
            logical_gpu_id = physical_to_logical.get(gpu_id, 0)
            logger.info(f"[VLLMStudentReward] Creating actor for physical GPU {gpu_id} (logical GPU {logical_gpu_id})...")
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = VLLMWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_id=gpu_id,
                logical_gpu_id=logical_gpu_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                micro_batch_size=self.micro_batch_size,
            )
            self.workers.append(worker)

        # Wait for all actors to initialize
        logger.info(f"[VLLMStudentReward] Waiting for actors to initialize...")
        ray.get([w.compute_logprobs.remote(["test"], [0]) for w in self.workers])
        logger.info(f"[VLLMStudentReward] All {self.num_gpus} actors initialized successfully")

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
        logger.info(f"[VLLMStudentReward] Computing rewards for {batch_size} samples on {self.num_gpus} GPUs")

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
        logger.info(f"[VLLMStudentReward] Phase 2: Distributing work to {self.num_gpus} Ray actors...")
        inference_start = time.time()

        # Split rewrite prompts across workers
        rewrite_chunks = self._split_into_chunks(rewrite_prompts, self.num_gpus)
        rewrite_ans_start_chunks = self._split_into_chunks(rewrite_ans_starts, self.num_gpus)

        # Submit rewrite tasks to all workers in parallel
        rewrite_futures = []
        for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(rewrite_chunks, rewrite_ans_start_chunks)):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                rewrite_futures.append((worker_idx, len(prompts_chunk), future))

        # Split original prompts across workers (if any)
        original_futures = []
        if original_prompts:
            original_chunks = self._split_into_chunks(original_prompts, self.num_gpus)
            original_ans_start_chunks = self._split_into_chunks(original_ans_starts, self.num_gpus)

            for worker_idx, (prompts_chunk, ans_starts_chunk) in enumerate(zip(original_chunks, original_ans_start_chunks)):
                if prompts_chunk:
                    future = self.workers[worker_idx].compute_logprobs.remote(prompts_chunk, ans_starts_chunk)
                    original_futures.append((worker_idx, len(prompts_chunk), future))

        # Gather rewrite results
        logger.info(f"[VLLMStudentReward] Waiting for {len(rewrite_futures)} rewrite tasks...")
        rewrite_results = [None] * batch_size
        for worker_idx, chunk_size, future in rewrite_futures:
            chunk_results = ray.get(future)
            # Place results back in original order
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_results[start_idx + i] = result

        # Gather original results
        original_results_map = {}
        if original_futures:
            logger.info(f"[VLLMStudentReward] Waiting for {len(original_futures)} original tasks...")
            for worker_idx, chunk_size, future in original_futures:
                chunk_results = ray.get(future)
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
