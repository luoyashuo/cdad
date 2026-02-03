"""
Student Reward (Cognitive Reward) Module - vLLM Version

Computes R_cog = log P_student(Answer | Question, Rewrite_CoT)
using vLLM for fast log prob computation.

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence.

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
        gpu_memory_utilization: float = 0.4,
        dtype: str = "float16",
        max_model_len: int = 8192,
    ):
        import os
        import subprocess
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
        print(f"[VLLMWorker-{gpu_id}] gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}", flush=True)
        print(f"[VLLMWorker-{gpu_id}] Initializing vLLM on GPU {gpu_id}...", flush=True)

        self.gpu_id = gpu_id
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu  # Save for later use
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
        print(f"[VLLMWorker-{self.gpu_id}] Processing {len(prompts)} prompts...", flush=True)
        if hasattr(self, '_get_nvidia_smi'):
            print(f"[VLLMWorker-{self.gpu_id}] GPU memory BEFORE generate: {self._get_nvidia_smi(self.gpu_id)}", flush=True)

        start_time = time.time()

        outputs = self.llm.generate(prompts, sampling_params)

        elapsed = time.time() - start_time
        speed = len(prompts) / elapsed if elapsed > 0 else 0
        print(f"[VLLMWorker-{self.gpu_id}] Done: {len(prompts)} prompts in {elapsed:.2f}s ({speed:.1f} samples/s)")

        # Extract log probs with mean and min
        results = []
        for output, ans_start in zip(outputs, answer_starts):
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
        """
        import torch

        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.max_model_len = max_model_len

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

        # Create Ray actors, one per GPU
        for gpu_id in self.gpu_ids:
            logger.info(f"[VLLMStudentReward] Creating actor for GPU {gpu_id}...")
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = VLLMWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_id=gpu_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
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

            # R_cog = rewrite_log_prob (不再减去 original_log_prob)
            # 直接使用 rewrite 的 answer log prob 作为 reward
            reward = rewrite_answer_log_prob

            # 仍然记录 original 的 log prob 用于日志分析（如果有的话）
            if i in original_results_map:
                original_result = original_results_map[i]
                original_mean = original_result["mean_log_prob"]
            else:
                original_mean = -100.0

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
