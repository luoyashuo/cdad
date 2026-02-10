"""
FSF (Failed Step Fraction) Judge Reward Module - SGLang Version

Computes R_logic based on "Failed Step Fraction" (FSF) analysis.
Uses SGLang Engine for fast LLM-as-a-Judge inference.

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

This encourages the model to produce more efficient reasoning with
fewer failed/abandoned branches.

Supports two modes:
1. Persistent mode (enable_offload=False): Engine stays in memory
2. Offload mode (enable_offload=True): Engine created on demand, destroyed after use
"""

import gc
import json
import re
from typing import Any, Dict, List, Optional
import torch

from verl.experimental.reward.cot_rewriter.logging_config import logger


class SGLangFSFJudge:
    """
    SGLang-based FSF (Failed Step Fraction) Judge

    Uses SGLang Engine for fast LLM inference to analyze the "failed branch ratio"
    in reasoning. A failed branch is defined as a reasoning path that was attempted
    but ultimately abandoned.

    Supports two modes:
    1. Persistent mode (enable_offload=False): Engine stays in memory
    2. Offload mode (enable_offload=True): Engine created on demand
    """

    # System prompt for FSF analysis
    SYSTEM_PROMPT = """You are an expert at analyzing reasoning processes.
Your task is to identify "failed branches" in a chain-of-thought reasoning.

A "failed branch" is a reasoning step or approach that:
1. Was attempted but then abandoned
2. Led to a wrong direction and had to backtrack
3. Was explicitly marked as incorrect or unhelpful

Analyze the given reasoning and output a JSON with:
{
    "total_steps": <number of distinct reasoning steps>,
    "failed_steps": <number of failed/abandoned steps>,
    "fsf": <failed_steps / total_steps, a value between 0.0 and 1.0>
}

Only output the JSON, nothing else."""

    USER_PROMPT_TEMPLATE = """Analyze the following reasoning process and calculate the Failed Step Fraction (FSF):

{reasoning}

Output JSON only:"""

    def __init__(
        self,
        model_path: str = "./models/Qwen3-8B",
        device: str = "cuda:3",
        dtype: str = "float16",
        allow_cpu: bool = True,
        enable_offload: bool = False,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
    ):
        """
        Initialize SGLang FSF Judge.

        Args:
            model_path: Path to FSF Judge Model
            device: Device to run model on (e.g., "cuda:3")
            dtype: Data type (float16 or bfloat16)
            allow_cpu: If True, allow fallback when CUDA not available
            enable_offload: If True, create engine on demand and destroy after use
            gpu_memory_utilization: GPU memory fraction for SGLang
            max_model_len: Maximum context length
        """
        self.model_path = model_path
        self.dtype = dtype
        self.enable_offload = enable_offload
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # Parse device
        if device.startswith("cuda:"):
            self.gpu_id = int(device.split(":")[1])
        else:
            self.gpu_id = 0
        self.device = device

        logger.info(f"[SGLangFSFJudge] Initializing FSF Judge")
        logger.info(f"  Model Path: {model_path}")
        logger.info(f"  Device: {device} (GPU {self.gpu_id})")
        logger.info(f"  Offload Enabled: {enable_offload}")
        logger.info(f"  GPU Memory Utilization: {gpu_memory_utilization}")

        # Load tokenizer (always keep in memory)
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize engine
        self.engine = None
        if not enable_offload:
            logger.info(f"Loading SGLang Engine (persistent mode)...")
            self._create_engine()

    def _create_engine(self):
        """Create SGLang Engine."""
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        from sglang import Engine

        logger.info(f"[SGLangFSFJudge] Creating SGLang Engine on GPU {self.gpu_id}...")

        self.engine = Engine(
            model_path=self.model_path,
            tp_size=1,
            mem_fraction_static=self.gpu_memory_utilization,
            dtype=self.dtype,
            context_length=self.max_model_len,
            trust_remote_code=True,
        )

        logger.info(f"[SGLangFSFJudge] SGLang Engine created successfully")

    def _destroy_engine(self):
        """Destroy SGLang Engine to release GPU memory."""
        if self.engine is not None:
            logger.info(f"[SGLangFSFJudge] Destroying SGLang Engine...")
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"[SGLangFSFJudge] Error during engine shutdown: {e}")
            self.engine = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"[SGLangFSFJudge] Engine destroyed")

    def load_to_gpu(self):
        """Load engine (for offload mode compatibility)."""
        if self.enable_offload and self.engine is None:
            self._create_engine()

    def offload_to_cpu(self):
        """Offload engine (for offload mode compatibility)."""
        if self.enable_offload:
            self._destroy_engine()

    def _parse_fsf(self, text: str) -> float:
        """
        Parse FSF value from model output.

        Args:
            text: Model output text

        Returns:
            FSF value between 0.0 and 1.0
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if json_match:
                return float(json_match.group(1))

            # Try to parse as complete JSON
            try:
                data = json.loads(text.strip())
                if "fsf" in data:
                    return float(data["fsf"])
            except json.JSONDecodeError:
                pass

            # Try to extract any number
            num_match = re.search(r"(\d+\.?\d*)", text)
            if num_match:
                value = float(num_match.group(1))
                return min(value, 1.0)

        except Exception:
            pass

        return 0.5  # Default to 0.5 if parsing fails

    def _compute_single_fsf(self, reasoning: str) -> float:
        """
        Compute FSF for a single reasoning text.

        Args:
            reasoning: The reasoning text to analyze

        Returns:
            FSF value between 0.0 and 1.0
        """
        from sglang.srt.sampling.sampling_params import SamplingParams

        # Truncate if too long
        reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning

        user_content = self.USER_PROMPT_TEMPLATE.format(reasoning=reasoning_truncated)

        try:
            # Build chat prompt
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate with SGLang
            sampling_params = SamplingParams(
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.9,
            )

            outputs = self.engine.generate([prompt], sampling_params)

            # Extract generated text
            if outputs and len(outputs) > 0:
                output = outputs[0]
                if isinstance(output, dict):
                    generated_text = output.get("text", "")
                else:
                    generated_text = str(output)

                return self._parse_fsf(generated_text)

        except Exception as e:
            logger.error(f"[SGLangFSFJudge] Error computing FSF: {e}")
            return 0.5  # Default on error

        return 0.5

    def compute_fsf_batch(self, reasonings: List[str]) -> List[float]:
        """
        Batch computation of FSF values.

        Args:
            reasonings: List of reasoning texts

        Returns:
            List of FSF values
        """
        from sglang.srt.sampling.sampling_params import SamplingParams

        # Build all prompts
        prompts = []
        for reasoning in reasonings:
            reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning
            user_content = self.USER_PROMPT_TEMPLATE.format(reasoning=reasoning_truncated)

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Batch generate
        sampling_params = SamplingParams(
            max_new_tokens=150,
            temperature=0.1,
            top_p=0.9,
        )

        try:
            outputs = self.engine.generate(prompts, sampling_params)

            fsf_values = []
            for output in outputs:
                if isinstance(output, dict):
                    generated_text = output.get("text", "")
                else:
                    generated_text = str(output)
                fsf_values.append(self._parse_fsf(generated_text))

            return fsf_values

        except Exception as e:
            logger.error(f"[SGLangFSFJudge] Batch compute error: {e}")
            return [0.5] * len(reasonings)

    def compute_logic_reward(
        self,
        original_cots: List[str],
        rewrite_cots: List[str],
    ) -> Dict[str, Any]:
        """
        Compute logic reward based on FSF comparison.

        R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

        When enable_offload=True:
        - Creates engine before computation
        - Destroys engine after computation

        Args:
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings

        Returns:
            Dictionary with:
                - original_fsfs: FSF values for original CoTs
                - rewrite_fsfs: FSF values for rewritten CoTs
                - logic_rewards: Binary rewards (1 if improved, 0 otherwise)
        """
        # Create engine if in offload mode
        if self.enable_offload:
            self._create_engine()

        try:
            logger.info(f"[SGLangFSFJudge] Computing FSF for {len(original_cots)} original CoTs...")
            original_fsfs = self.compute_fsf_batch(original_cots)

            logger.info(f"[SGLangFSFJudge] Computing FSF for {len(rewrite_cots)} rewrite CoTs...")
            rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

            # R_logic = 1 if rewrite FSF < original FSF (improvement)
            logic_rewards = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]

            return {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "logic_rewards": logic_rewards,
            }
        finally:
            # Destroy engine if in offload mode
            if self.enable_offload:
                self._destroy_engine()


# Global singleton
_sglang_fsf_judge: Optional[SGLangFSFJudge] = None


def get_sglang_fsf_judge(
    model_path: str = "./models/Qwen3-8B",
    **kwargs,
) -> SGLangFSFJudge:
    """Get or create SGLang FSF Judge singleton."""
    global _sglang_fsf_judge
    if _sglang_fsf_judge is None:
        _sglang_fsf_judge = SGLangFSFJudge(model_path=model_path, **kwargs)
    return _sglang_fsf_judge
