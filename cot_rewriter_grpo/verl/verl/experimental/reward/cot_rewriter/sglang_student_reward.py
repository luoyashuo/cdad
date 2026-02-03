"""
Student Reward (Cognitive Reward) Module - SGLang Version

Computes R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
using SGLang for fast log prob computation.

其中 answer_log_prob = avg(token_log_probs) + alpha_cog * min(token_log_probs)

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence
compared to the original CoT.

Uses Ray actors for multi-GPU parallelism, with each GPU running
a separate SGLang Engine instance.
"""

from typing import Any, Dict, List, Optional
import ray

from verl.experimental.reward.cot_rewriter.logging_config import logger


@ray.remote
class SGLangWorkerActor:
    """
    Ray actor that runs SGLang Engine on a specific GPU.
    Each actor is a separate process with its own SGLang Engine instance.
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

        from sglang import Engine
        from sglang.srt.server_args import ServerArgs
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

        print(f"[SGLangWorker-{gpu_id}] ========== GPU MEMORY CHECK ==========", flush=True)
        mem_before = get_nvidia_smi_memory_for_gpu(gpu_id)
        print(f"[SGLangWorker-{gpu_id}] BEFORE loading Student SGLang: {mem_before}", flush=True)
        print(f"[SGLangWorker-{gpu_id}] gpu_memory_utilization={gpu_memory_utilization}, max_model_len={max_model_len}", flush=True)
        print(f"[SGLangWorker-{gpu_id}] Initializing SGLang Engine on GPU {gpu_id}...", flush=True)

        self.gpu_id = gpu_id
        self._get_nvidia_smi = get_nvidia_smi_memory_for_gpu

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Initialize SGLang Engine
        # Note: SGLang uses mem_fraction_static instead of gpu_memory_utilization
        self.engine = Engine(
            model_path=model_path,
            tp_size=1,
            mem_fraction_static=gpu_memory_utilization,
            dtype=dtype,
            context_length=max_model_len,
            trust_remote_code=True,
        )

        print(f"[SGLangWorker-{gpu_id}] SGLang Engine initialized successfully", flush=True)
        print(f"[SGLangWorker-{gpu_id}] nvidia-smi AFTER loading Student SGLang:", flush=True)
        print(f"[SGLangWorker-{gpu_id}] {get_nvidia_smi_memory_for_gpu(gpu_id)}", flush=True)
        print(f"[SGLangWorker-{gpu_id}] ========================================", flush=True)

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
        from sglang.srt.sampling.sampling_params import SamplingParams
        import time

        sampling_params = SamplingParams(
            max_new_tokens=1,
            return_logprob=True,
            top_logprobs_num=1,
            temperature=0,
        )

        # Log GPU memory before generation
        print(f"[SGLangWorker-{self.gpu_id}] Processing {len(prompts)} prompts...", flush=True)
        if hasattr(self, '_get_nvidia_smi'):
            print(f"[SGLangWorker-{self.gpu_id}] GPU memory BEFORE generate: {self._get_nvidia_smi(self.gpu_id)}", flush=True)

        start_time = time.time()

        # SGLang generate with logprobs
        outputs = self.engine.generate(
            prompts,
            sampling_params,
        )

        elapsed = time.time() - start_time
        speed = len(prompts) / elapsed if elapsed > 0 else 0
        print(f"[SGLangWorker-{self.gpu_id}] Done: {len(prompts)} prompts in {elapsed:.2f}s ({speed:.1f} samples/s)")

        # Extract log probs with mean and min
        results = []
        for output, ans_start in zip(outputs, answer_starts):
            logprobs = self._extract_answer_logprobs(output, ans_start)
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

    def _extract_answer_logprobs(self, output, answer_start_idx: int) -> List[float]:
        """Extract log probabilities for the answer portion from SGLang output."""
        log_probs = []

        try:
            # SGLang output format: output has meta_info with input_token_logprobs
            meta_info = output.get("meta_info", {})

            # Try input_token_logprobs (prompt logprobs)
            input_token_logprobs = meta_info.get("input_token_logprobs", [])
            if input_token_logprobs:
                for i in range(answer_start_idx, len(input_token_logprobs)):
                    token_info = input_token_logprobs[i]
                    if token_info is not None:
                        # SGLang format: (logprob, token_id, token_text) or similar
                        if isinstance(token_info, (list, tuple)) and len(token_info) > 0:
                            log_probs.append(float(token_info[0]))
                        elif isinstance(token_info, dict) and "logprob" in token_info:
                            log_probs.append(float(token_info["logprob"]))
                        elif isinstance(token_info, (int, float)):
                            log_probs.append(float(token_info))

            # Fallback: try prompt_logprobs if available
            if not log_probs:
                prompt_logprobs = meta_info.get("prompt_logprobs", [])
                if prompt_logprobs:
                    for i in range(answer_start_idx, len(prompt_logprobs)):
                        token_info = prompt_logprobs[i]
                        if token_info is not None:
                            if isinstance(token_info, (list, tuple)) and len(token_info) > 0:
                                log_probs.append(float(token_info[0]))
                            elif isinstance(token_info, dict):
                                lp = token_info.get("logprob", token_info.get("top_logprobs", [{}])[0].get("logprob", -100))
                                log_probs.append(float(lp))
                            elif isinstance(token_info, (int, float)):
                                log_probs.append(float(token_info))

        except Exception as e:
            print(f"[SGLangWorker-{self.gpu_id}] Error extracting logprobs: {e}", flush=True)
            return [-100.0]

        return log_probs if log_probs else [-100.0]

    def shutdown(self):
        """Shutdown the SGLang engine to release GPU memory."""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as e:
                print(f"[SGLangWorker-{self.gpu_id}] Error during shutdown: {e}", flush=True)


class SGLangStudentRewardClient:
    """
    Multi-GPU SGLang-based Student Reward Client using Ray actors.

    Each GPU runs a separate SGLang Engine in its own Ray actor process.
    This avoids conflicts with the main Ray cluster and allows true parallelism.

    Benefits:
    1. Each GPU has its own SGLang Engine (no TP coordination overhead)
    2. True data parallelism across GPUs
    3. SGLang's efficient batching within each GPU
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
        Initialize Multi-GPU SGLang Student Reward Client with Ray actors.

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

        logger.info(f"[SGLangStudentReward] Will use {self.num_gpus} Ray actors on GPUs: {self.gpu_ids}")

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

        logger.info(f"[SGLangStudentReward] Cleaning up {len(self.workers)} SGLang Ray actors...")
        logger.info(f"[SGLangStudentReward] nvidia-smi BEFORE cleanup: {get_nvidia_smi()}")

        # Shutdown and kill all Ray actors to release GPU memory
        for i, worker in enumerate(self.workers):
            try:
                # Try to gracefully shutdown the engine first
                ray.get(worker.shutdown.remote())
            except Exception as e:
                logger.warning(f"[SGLangStudentReward] Error shutting down worker {i}: {e}")
            try:
                ray.kill(worker)
                logger.info(f"[SGLangStudentReward] Killed worker {i}")
            except Exception as e:
                logger.warning(f"[SGLangStudentReward] Error killing worker {i}: {e}")

        self.workers = []
        self._initialized = False

        # Force garbage collection and CUDA cache cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for GPU memory to be released
        logger.info(f"[SGLangStudentReward] Waiting for GPU memory release...")
        time.sleep(2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[SGLangStudentReward] nvidia-smi AFTER cleanup: {get_nvidia_smi()}")
        logger.info(f"[SGLangStudentReward] Cleanup completed")

    def initialize(self):
        """Lazy initialization of Ray SGLang actors."""
        if self._initialized:
            return

        from transformers import AutoTokenizer

        logger.info(f"[SGLangStudentReward] Initializing {self.num_gpus} SGLang Ray actors...")

        # Initialize tokenizer (shared, for prompt building)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Create Ray actors, one per GPU
        for gpu_id in self.gpu_ids:
            logger.info(f"[SGLangStudentReward] Creating actor for GPU {gpu_id}...")
            # Note: num_gpus=0 because we set CUDA_VISIBLE_DEVICES inside the actor
            worker = SGLangWorkerActor.options(num_gpus=0, num_cpus=1).remote(
                model_path=self.model_path,
                gpu_id=gpu_id,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
            )
            self.workers.append(worker)

        # Wait for all actors to initialize
        logger.info(f"[SGLangStudentReward] Waiting for actors to initialize...")
        ray.get([w.compute_logprobs.remote(["test"], [0]) for w in self.workers])
        logger.info(f"[SGLangStudentReward] All {self.num_gpus} actors initialized successfully")

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
        Compute cognitive rewards for a batch using multi-GPU SGLang Ray actors.

        Each Ray actor runs SGLang Engine on a separate GPU. We split the batch across
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
        logger.info(f"[SGLangStudentReward] Computing rewards for {batch_size} samples on {self.num_gpus} GPUs")

        start_time = time.time()

        # Phase 1: Build all prompts (CPU)
        logger.info(f"[SGLangStudentReward] Phase 1: Building prompts...")
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
        logger.info(f"[SGLangStudentReward] Phase 1 done: {prep_time:.2f}s")

        # Phase 2: Distribute work to Ray actors
        logger.info(f"[SGLangStudentReward] Phase 2: Distributing work to {self.num_gpus} Ray actors...")
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
        logger.info(f"[SGLangStudentReward] Waiting for {len(rewrite_futures)} rewrite tasks...")
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
            logger.info(f"[SGLangStudentReward] Waiting for {len(original_futures)} original tasks...")
            for worker_idx, chunk_size, future in original_futures:
                chunk_results = ray.get(future)
                start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
                for i, result in enumerate(chunk_results):
                    original_idx = original_indices[start_idx + i]
                    original_results_map[original_idx] = result

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        logger.info(f"[SGLangStudentReward] Phase 2 done: {inference_time:.2f}s inference, {total_time:.2f}s total")

        # Phase 3: Compute rewards
        logger.info(f"[SGLangStudentReward] Phase 3: Computing rewards with alpha_cog={alpha_cog}...")
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

        logger.info(f"[SGLangStudentReward] All {batch_size} rewards computed")

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
