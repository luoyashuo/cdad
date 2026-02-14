"""
SGLang-based FSF Judge with graphviz DOT format parsing.

Parses the FSF model's DOT format output to calculate FSF:
- FSF = failed_nodes / total_nodes
- failed_nodes are nodes with fillcolor=lightpink
- R_logic = 1 if FSF(rewrite) < FSF(original) else 0

Supports:
- Custom temperature, top_k, top_p
- Auto-detects GPUs and creates multiple instances for data parallelism
- On-demand creation and destruction
- Skips samples with unparseable output (returns None)
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch


def get_nvidia_smi_memory_for_gpu(gpu_idx: int) -> str:
    """Get GPU memory usage from nvidia-smi for a specific GPU."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
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
        return f"GPU {gpu_idx}: error - {e}"


@ray.remote(num_cpus=1)
class SGLangFSFActor:
    """
    Ray Actor for SGLang FSF Judge computation.

    Each actor runs SGLang on a separate GPU (or group of GPUs for TP).
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],
        logical_gpu_ids: List[int],
        tensor_parallel_size: int,
        dtype: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
        max_new_tokens: int,
    ):
        """
        Initialize SGLang FSF Actor.

        Args:
            model_path: Path to FSF Judge Model
            gpu_ids: Physical GPU IDs (e.g., [0, 1] for TP=2)
            logical_gpu_ids: Logical GPU IDs within CUDA_VISIBLE_DEVICES
            tensor_parallel_size: TP size (must equal len(gpu_ids))
            dtype: Data type (float16 or bfloat16)
            gpu_memory_utilization: GPU memory fraction for SGLang
            max_model_len: Maximum context length
            temperature: Sampling temperature
            top_k: Sampling top-k
            top_p: Sampling top-p (nucleus)
            max_new_tokens: Maximum tokens to generate
        """
        import os

        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.logical_gpu_ids = logical_gpu_ids
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        worker_name = f"SGLangFSFActor-{gpu_ids}"

        # Set CUDA_VISIBLE_DEVICES for this actor
        parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        # e.g., logical_gpu_ids=[0, 1] -> "0,1" for TP=2
        cuda_devices_str = ",".join(map(str, logical_gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str

        # Log memory info before creating engine
        memory_info = ", ".join([get_nvidia_smi_memory_for_gpu(gpu) for gpu in gpu_ids])
        logger.info(f"[{worker_name}] GPU Memory: {memory_info}")

        logger.info(f"[{worker_name}] Initializing...")
        logger.info(f"  Model Path: {model_path}")
        logger.info(f"  GPUs: {gpu_ids} (logical: {logical_gpu_ids})")
        logger.info(f"  TP Size: {tensor_parallel_size}")
        logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")
        logger.info(f"  Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}")
        logger.info(f"  Max New Tokens: {max_new_tokens}")

        # Load tokenizer (always keep in memory)
        from transformers import AutoTokenizer
        logger.info(f"[{worker_name}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Engine will be created on first use
        self.engine = None
        self._engine_initialized = False

    def _create_engine(self):
        """Create SGLang Engine."""
        import os

        # CUDA_VISIBLE_DEVICES already set in __init__
        from sglang import Engine

        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Creating SGLang Engine...")

        start_time = time.time()
        self.engine = Engine(
            model_path=self.model_path,
            tp_size=self.tensor_parallel_size,
            mem_fraction_static=self.gpu_memory_utilization,
            dtype=self.dtype,
            context_length=self.max_model_len,
            trust_remote_code=True,
        )

        elapsed = time.time() - start_time
        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Engine created in {elapsed:.2f}s")
        self._engine_initialized = True

    def _destroy_engine(self):
        """Destroy SGLang Engine to release GPU memory."""
        if self.engine is not None:
            logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Destroying SGLang Engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Error during shutdown: {e}")
            self.engine = None
            self._engine_initialized = False

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Engine destroyed")

    def _parse_fsf_from_dot(self, dot_output: str) -> Optional[float]:
        """
        Parse FSF value from DOT graph format output.

        Following the logic in analyze_rg_fsf.py:
        1. Use pydot to parse DOT format
        2. Extract nodes and edges from graph
        3. Count failed nodes (nodes with fillcolor=lightpink)
        4. Calculate FSF = failed_nodes / total_nodes

        Args:
            dot_output: Raw model output text (may contain DOT format)

        Returns:
            FSF value between 0.0 and 1.0, or None if parsing fails
        """
        try:
            import pydot

            # Try to find DOT graph in output
            # DOT format starts with "digraph" or "strict digraph"
            dot_match = re.search(r'(?:strict\s+)?digraph\s+.*?(?:\n|\r|\{)', dot_output, re.DOTALL)
            if not dot_match:
                # Try to find any {...} block that might be DOT-like
                dot_match = re.search(r'\{[^}]*\}', dot_output, re.DOTALL)

            if not dot_match:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] No DOT format found in output")
                return None

            dot_code = dot_match.group(0) if dot_match.lastindex else dot_output

            # Parse DOT format
            graphs = pydot.graph_from_dot_data(dot_code)

            if not graphs or len(graphs) == 0:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Failed to parse DOT graph")
                return None

            graph = graphs[0]
            nodes = graph.get_nodes() or []
            edges = graph.get_edges() or []

            node_count = len(nodes)
            edge_count = len(edges)

            # Count failed nodes (nodes with fillcolor=lightpink)
            failed_nodes = 0
            for node in nodes:
                if node is None:
                    continue
                # Get node's fillcolor attribute
                try:
                    fillcolor = node.get('fillcolor')
                    if fillcolor is not None:
                        fillcolor_str = str(fillcolor).strip('"').lower()
                        if fillcolor_str == 'lightpink':
                            failed_nodes += 1
                except:
                    # If get fails, node might not have fillcolor attribute
                    continue

            # Compute FSF: failed_nodes / total_nodes
            fsf = failed_nodes / max(node_count, 1) if node_count > 0 else 0.0

            logger.debug(
                f"[SGLangFSFActor-{self.gpu_ids}] Parsed DOT: nodes={node_count}, "
                f"edges={edge_count}, failed_nodes={failed_nodes}, fsf={fsf:.4f}"
            )

            return fsf

        except ImportError:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] pydot not installed, using fallback parsing")
            return self._parse_fsf_fallback(dot_output)
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Error parsing DOT format: {e}")
            return self._parse_fsf_fallback(dot_output)

    def _parse_fsf_fallback(self, output: str) -> Optional[float]:
        """
        Fallback FSF parsing when DOT format is not available.

        Tries to extract FSF from:
        1. JSON format with "fsf" key
        2. Plain text pattern matching "0.xx" format

        Args:
            output: Model output text

        Returns:
            FSF value between 0.0 and 1.0, or None if parsing fails
        """
        try:
            # Try to find JSON with fsf field
            json_match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', output)
            if json_match:
                value = float(json_match.group(1))
                if 0.0 <= value <= 1.0:
                    return value

            # Try to parse as complete JSON
            import json
            try:
                data = json.loads(output.strip())
                if "fsf" in data:
                    value = float(data["fsf"])
                    if 0.0 <= value <= 1.0:
                        return value
            except json.JSONDecodeError:
                pass

            # Try to extract any number that looks like FSF (0.00 - 1.00)
            # Pattern: "0.xx" where xx are digits
            fsf_match = re.search(r'\b0\.\d{2,2}\b', output)
            if fsf_match:
                value = float(fsf_match.group(0))
                if 0.0 <= value <= 1.0:
                    return value

            logger.debug(f"[SGLangFSFActor-{self.gpu_ids}] Fallback parsing found no valid FSF")
            return None

        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Fallback parsing error: {e}")
            return None

    def _build_fsf_prompt(self, question: str, reasoning: str) -> str:
        """
        Build FSF prompt with proper format.

        Args:
            question: Problem text
            reasoning: CoT reasoning to evaluate

        Returns:
            Formatted prompt string
        """
        # Use build_cot_rewrite_from_sharegpt.py message format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "reasoning_content": reasoning}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def compute_fsf_batch(
        self,
        questions: List[str],
        reasonings: List[str],
    ) -> List[Optional[float]]:
        """
        Batch computation of FSF values.

        Args:
            questions: List of problem texts
            reasonings: List of reasoning texts

        Returns:
            List of FSF values (None for unparseable outputs)
        """
        from sglang.srt.sampling_params import SamplingParams

        # Build all prompts
        prompts = []
        for question, reasoning in zip(questions, reasonings):
            reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning
            prompt = self._build_fsf_prompt(question, reasoning_truncated)
            prompts.append(prompt)

        # Batch generate
        sampling_params = SamplingParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        try:
            outputs = self.engine.generate(prompts, sampling_params)

            fsf_values = []
            for output in outputs:
                if isinstance(output, dict):
                    generated_text = output.get("text", "")
                else:
                    generated_text = str(output)
                fsf_values.append(self._parse_fsf_from_dot(generated_text))

            return fsf_values

        except Exception as e:
            logger.error(f"[SGLangFSFActor-{self.gpu_ids}] Batch compute error: {e}")
            return [None] * len(reasonings)

    def cleanup(self):
        """Release KV Cache memory, keep weights and engine instance."""
        if self.engine is None:
            return

        try:
            logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Releasing KV Cache memory...")
            # Check if release_memory_occupation method exists
            if hasattr(self.engine, 'release_memory_occupation'):
                self.engine.release_memory_occupation()
                logger.info(f"[SGLangFSFActor-{self.gpu_ids}] KV Cache memory released")
            else:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Engine does not support release_memory_occupation")
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Failed to release memory: {e}")

    def resume(self):
        """Resume KV Cache memory after engine is created."""
        if self.engine is None:
            logger.debug(f"[SGLangFSFActor-{self.gpu_ids}] Memory not resumed, skipping")
            return

        try:
            if hasattr(self.engine, 'resume_memory_occupation'):
                self.engine.resume_memory_occupation()
                logger.debug(f"[SGLangFSFActor-{self.gpu_ids}] Resumed KV Cache memory")
            else:
                logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Engine does not support resume_memory_occupation")
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Failed to resume memory: {e}")

    def shutdown(self):
        """Shutdown the engine and release resources."""
        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Starting shutdown...")

        try:
            if self.engine is not None:
                self.engine.shutdown()
        except Exception as e:
            logger.warning(f"[SGLangFSFActor-{self.gpu_ids}] Error during shutdown: {e}")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[SGLangFSFActor-{self.gpu_ids}] Shutdown complete")


class SGLangFSFJudge:
    """
    SGLang-based FSF Judge with multi-instance data parallelism.

    This class manages multiple Ray actors for parallel FSF computation.
    Similar to SGLangStudentRewardClient architecture.
    """

    def __init__(
        self,
        model_path: str = "./models/Qwen3-4B",
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        max_new_tokens: int = 8000,
    ):
        """
        Initialize Multi-GPU SGLang FSF Judge with Ray actors.

        Args:
            model_path: Path to FSF Judge Model
            dtype: Data type (float16 or bfloat16)
            gpu_memory_utilization: GPU memory fraction for SGLang
            max_model_len: Maximum context length
            tensor_parallel_size: TP size per SGLang instance.
                - TP=1: Each GPU runs independent SGLang (Data Parallelism)
                - TP=2: Every 2 GPUs share one SGLang (model sharded)
                - TP=4: Every 4 GPUs share one SGLang
            temperature: Sampling temperature
            top_k: Sampling top-k
            top_p: Sampling top-p (nucleus)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        # Auto-detect devices
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Calculate number of actors based on TP size
        # num_actors = num_gpus / tensor_parallel_size
        assert num_gpus >= tensor_parallel_size, \
            f"num_gpus ({num_gpus}) must be >= tensor_parallel_size ({tensor_parallel_size})"
        assert num_gpus % tensor_parallel_size == 0, \
            f"num_gpus ({num_gpus}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
        self.num_actors = num_gpus // tensor_parallel_size

        # Assign GPUs to actors
        # e.g., gpu_ids=[0,1,2,3], TP=2 -> [[0,1], [2,3]]
        self.gpu_ids = [int(d.split(":")[-1]) for d in [f"cuda:{i}" for i in range(num_gpus)]]

        # Group GPUs for each actor
        gpu_groups = []
        for i in range(self.num_actors):
            group = self.gpu_ids[i * self.tensor_parallel_size : (i + 1) * self.tensor_parallel_size]
            gpu_groups.append(group)
        self.gpu_groups = gpu_groups

        logger.info(f"[SGLangFSFJudge] Initializing Multi-Instance FSF Judge (DOT parser)")
        logger.info(f"  Model Path: {model_path}")
        logger.info(f"  Total GPUs: {num_gpus}, Actors: {self.num_actors}, TP per actor: {tensor_parallel_size}")
        logger.info(f"  GPU groups: {self.gpu_groups}")
        logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")
        logger.info(f"  Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}")

        # Load tokenizer once (shared by all actors)
        from transformers import AutoTokenizer
        logger.info(f"[SGLangFSFJudge] Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create Ray actors
        self.workers: List[ray.actor.ActorHandle] = []
        self._initialized = False

    def _build_fsf_prompt(self, question: str, reasoning: str) -> str:
        """
        Build FSF prompt with proper format.

        Args:
            question: Problem text
            reasoning: CoT reasoning to evaluate

        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "reasoning_content": reasoning}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _split_into_chunks(self, items: List, num_chunks: int) -> List[List]:
        """Split items into roughly equal chunks."""
        if num_chunks <= 0:
            num_chunks = 1

        chunk_size = max(1, len(items) // num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            if i == num_chunks - 1:
                end = min(start + chunk_size, len(items))
            else:
                end = start + chunk_size
            chunks.append(items[start:end])
        return chunks

    def initialize_workers(self):
        """Initialize all Ray actors in parallel."""
        if self._initialized:
            return

        import subprocess

        # Assign ports for each actor to avoid conflicts
        # SGLang uses ports 8000-8999 for TP communication
        # Each actor needs unique ports
        base_port = 8000
        port_per_actor = 100  # Reserve 100 ports per actor
        assigned_ports = [base_port + i * port_per_actor for i in range(self.num_actors)]

        logger.info(f"[SGLangFSFJudge] Initializing {self.num_actors} workers IN PARALLEL with unique ports...")
        init_futures = []
        for i, worker in enumerate(self.workers):
            logger.info(f"[SGLangFSFJudge] Submitting init for worker {i}/{self.num_actors} on GPUs {self.gpu_groups[i]} (base_port={assigned_ports[i]})...")
            init_futures.append(worker.initialize.remote(
                [assigned_ports[i]],
                [8000 + i * port_per_actor]  # additional ports
            ))

        # Wait for all workers to complete initialization in parallel
        try:
            results = ray.get(init_futures, timeout=180)
            logger.info(f"[SGLangFSFJudge] All {self.num_actors} workers initialized successfully in parallel")
        except Exception as e:
            logger.error(f"[SGLangFSFJudge] Parallel init FAILED: {e}")
            raise

        self._initialized = True
        logger.info(f"[SGLangFSFJudge] All {self.num_actors} workers initialized")

    def compute_logic_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
    ) -> Dict[str, Any]:
        """
        Compute logic reward based on FSF comparison using multiple workers.

        R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

        For samples where FSF parsing fails, R_logic = None (caller should skip these).

        Args:
            questions: List of problem texts
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings

        Returns:
            Dictionary with:
                - original_fsfs: FSF values for original CoTs (None if unparseable)
                - rewrite_fsfs: FSF values for rewritten CoTs (None if unparseable)
                - logic_rewards: Binary rewards (1 if improved and parseable, 0 otherwise, None if unparseable)
        """
        # Ensure workers are initialized
        if not self._initialized:
            # Create actors
            for i, gpu_group in enumerate(self.gpu_groups):
                logical_gpu_ids = list(range(len(gpu_group)))
                actor = ray.remote(
                    num_cpus=len(gpu_group) * 4,  # Allocate CPU per GPU
                    num_gpus=0,
                )(SGLangFSFActor).remote(
                    model_path=self.model_path,
                    gpu_ids=gpu_group,
                    logical_gpu_ids=logical_gpu_ids,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype=self.dtype,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                )
                self.workers.append(actor)

            # Initialize all workers
            self.initialize_workers()

        # Build prompts
        original_prompts = []
        for question, cot in zip(questions, original_cots):
            cot_truncated = cot[:3000] if len(cot) > 3000 else cot
            prompt = self._build_fsf_prompt(question, cot_truncated)
            original_prompts.append(prompt)

        rewrite_prompts = []
        for question, cot in zip(questions, rewrite_cots):
            cot_truncated = cot[:3000] if len(cot) > 3000 else cot
            prompt = self._build_fsf_prompt(question, cot_truncated)
            rewrite_prompts.append(prompt)

        # Split data across workers
        original_chunks = self._split_into_chunks(original_prompts, self.num_actors)
        rewrite_chunks = self._split_into_chunks(rewrite_prompts, self.num_actors)

        # Compute original FSFs
        logger.info(f"[SGLangFSFJudge] Computing FSF for {len(original_cots)} original CoTs...")
        original_futures = []
        for worker_idx, prompts_chunk in enumerate(original_chunks):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_fsf_batch.remote(
                    questions[len(original_chunks[0]):][:len(prompts_chunk)],
                    prompts_chunk,
                )
                original_futures.append((worker_idx, len(prompts_chunk), future))

        # Aggregate original results
        original_fsfs = [None] * len(original_cots)
        for worker_idx, chunk_size, future in original_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(original_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                original_fsfs[start_idx + i] = result

        # Compute rewrite FSFs
        logger.info(f"[SGLangFSFJudge] Computing FSF for {len(rewrite_cots)} rewrite CoTs...")
        rewrite_futures = []
        for worker_idx, prompts_chunk in enumerate(rewrite_chunks):
            if prompts_chunk:
                future = self.workers[worker_idx].compute_fsf_batch.remote(
                    questions[len(rewrite_chunks[0]):][:len(prompts_chunk)],
                    prompts_chunk,
                )
                rewrite_futures.append((worker_idx, len(prompts_chunk), future))

        # Aggregate rewrite results
        rewrite_fsfs = [None] * len(rewrite_cots)
        for worker_idx, chunk_size, future in rewrite_futures:
            chunk_results = ray.get(future)
            start_idx = sum(len(rewrite_chunks[i]) for i in range(worker_idx))
            for i, result in enumerate(chunk_results):
                rewrite_fsfs[start_idx + i] = result

        # Compute logic rewards
        logic_rewards = []
        for o_fsf, r_fsf in zip(original_fsfs, rewrite_fsfs):
            if o_fsf is None or r_fsf is None:
                # Unparseable output, skip this sample
                logic_rewards.append(None)
            elif r_fsf < o_fsf:
                logic_rewards.append(1.0)
            else:
                logic_rewards.append(0.0)

        return {
            "original_fsfs": original_fsfs,
            "rewrite_fsfs": rewrite_fsfs,
            "logic_rewards": logic_rewards,
        }

    def cleanup(self):
        """Release KV Cache for all workers."""
        if not self._initialized:
            return

        logger.info("[SGLangFSFJudge] Cleaning up all workers...")
        for worker in self.workers:
            try:
                worker.cleanup.remote()
            except Exception as e:
                logger.warning(f"[SGLangFSFJudge] Error during worker cleanup: {e}")

    def shutdown(self):
        """Shutdown all workers and release resources."""
        logger.info("[SGLangFSFJudge] Starting shutdown...")

        for worker in self.workers:
            try:
                worker.shutdown.remote()
            except Exception as e:
                logger.warning(f"[SGLangFSFJudge] Error during worker shutdown: {e}")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[SGLangFSFJudge] Shutdown complete")


# Global singleton
_fsf_judge: Optional[SGLangFSFJudge] = None
_fsf_judge_kwargs: Optional[dict] = None


def get_fsf_judge(
    model_path: str = "./models/Qwen3-4B",
    **kwargs,
) -> SGLangFSFJudge:
    """Get or create SGLang FSF Judge singleton."""
    global _fsf_judge, _fsf_judge_kwargs
    if _fsf_judge is None:
        _fsf_judge_kwargs = kwargs.copy()
        _fsf_judge = SGLangFSFJudge(model_path=model_path, **kwargs)
    else:
        # Log if kwargs differ from initial creation
        # (For now, we just reuse the existing instance)
        pass
    return _fsf_judge


def cleanup_fsf_judge():
    """Cleanup the global FSF judge to release GPU memory."""
    global _fsf_judge, _fsf_judge_kwargs
    if _fsf_judge is not None:
        logger.info("[FSF] Cleaning up FSF judge...")
        _fsf_judge.shutdown()
        _fsf_judge = None
        _fsf_judge_kwargs = None

        # Force garbage collection
        import gc
        import torch
        import time
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Wait for GPU memory to be released
        logger.info("[FSF] Waiting for GPU memory to be released...")
        time.sleep(3)
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("[FSF] FSF judge cleanup completed")
