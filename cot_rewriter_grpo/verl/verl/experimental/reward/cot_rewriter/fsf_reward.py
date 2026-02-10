"""
FSF (Failed Step Fraction) Judge Reward Module

Computes R_logic based on "Failed Step Fraction" (FSF) analysis.
Uses LLM-as-a-Judge to analyze the proportion of failed reasoning branches.

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

This encourages the model to produce more efficient reasoning with
fewer failed/abandoned branches.

Supports two modes:
1. Persistent mode (enable_offload=False): Model stays on GPU
2. Offload mode (enable_offload=True): Model on CPU, loads to GPU on demand
"""

import gc
import json
import re
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from verl.experimental.reward.cot_rewriter.logging_config import logger


def get_available_device(preferred_device: str = "cuda:3", allow_cpu: bool = True) -> torch.device:
    """
    Get available device for model loading.

    Args:
        preferred_device: Preferred CUDA device (e.g., "cuda:3", "cuda:0")
        allow_cpu: If True, fallback to CPU when CUDA is not available.
                  If False, raise RuntimeError when CUDA is not available.

    Returns:
        torch.device: Available device (CUDA or CPU)

    Raises:
        RuntimeError: If CUDA is not available and allow_cpu is False
    """
    if not torch.cuda.is_available():
        if allow_cpu:
            logger.info(f"[HFFSFJudge] CUDA is not available. Using CPU as fallback.")
            return torch.device("cpu")
        else:
            raise RuntimeError(
                "[HFFSFJudge] CUDA is not available. "
                "FSF Judge model requires GPU to run. "
                "Please ensure CUDA_VISIBLE_DEVICES is set correctly in ray_kwargs.ray_init.runtime_env.env_vars "
                "Or set reward_kwargs.allow_cpu=true to allow CPU fallback (slower)."
            )

    try:
        device = torch.device(preferred_device)
        # Test if device is accessible by creating a small tensor
        _ = torch.tensor([1.0], device=device)
        return device
    except RuntimeError as e:
        # Preferred device not available, try cuda:0
        try:
            device = torch.device("cuda:0")
            _ = torch.tensor([1.0], device=device)
            logger.info(f"[HFFSFJudge] Preferred device {preferred_device} not available, using cuda:0")
            return device
        except RuntimeError:
            if allow_cpu:
                logger.info(f"[HFFSFJudge] No CUDA device available, using CPU as fallback")
                return torch.device("cpu")
            raise RuntimeError(
                f"[HFFSFJudge] No CUDA device available. "
                f"Preferred device {preferred_device} failed with: {e}"
            ) from e


class HFFSFJudge:
    """
    HuggingFace FSF (Failed Step Fraction) Judge

    Uses LLM-as-a-Judge to analyze the "failed branch ratio" in reasoning.
    A failed branch is defined as a reasoning path that was attempted but
    ultimately abandoned.

    Supports two modes:
    1. Persistent mode (enable_offload=False): Model stays on GPU
    2. Offload mode (enable_offload=True): Model on CPU, loads to GPU on demand
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
        dtype: torch.dtype = torch.bfloat16,
        allow_cpu: bool = True,
        enable_offload: bool = False,
        offload_device: str = "cpu",
        auto_offload: bool = False,
    ):
        """
        Initialize FSF Judge client.

        Args:
            model_path: Path to FSF Judge Model
            device: Preferred device to run model on (will fall back to CPU if not available)
            dtype: Precision (bfloat16 recommended for GPU, float32 for CPU)
            allow_cpu: If True, allow fallback to CPU when CUDA is not available
            enable_offload: If True, enable load/offload mechanism for GPU memory saving
            offload_device: Device to offload model when not in use (default: "cpu")
            auto_offload: If True, automatically offload after each batch (slower but simpler).
                         If False (default), caller must manually call load_to_gpu/offload_to_cpu.
        """
        self.model_path = model_path
        self.enable_offload = enable_offload
        self.offload_device = torch.device(offload_device)
        self.auto_offload = auto_offload

        # Determine compute device
        if enable_offload:
            try:
                self.compute_device = torch.device(device)
                _ = torch.tensor([1.0], device=self.compute_device)
            except RuntimeError:
                if allow_cpu:
                    logger.info(f"[HFFSFJudge] Device {device} not available, using CPU")
                    self.compute_device = torch.device("cpu")
                    self.enable_offload = False
                else:
                    raise
        else:
            self.compute_device = get_available_device(device, allow_cpu=allow_cpu)

        # Alias for compatibility
        self.device = self.compute_device

        # Use float32 for CPU (bfloat16 not well supported on CPU)
        if self.compute_device.type == "cpu":
            self.dtype = torch.float32
            logger.info(f"[HFFSFJudge] Using float32 for CPU")
        else:
            self.dtype = dtype

        logger.info(f"[HFFSFJudge] Initializing FSF Judge")
        logger.info(f"  Model Path: {model_path}")
        logger.info(f"  Compute Device: {self.compute_device}")
        logger.info(f"  Offload Enabled: {self.enable_offload}")
        if self.enable_offload:
            logger.info(f"  Offload Device: {self.offload_device}")

        # Load tokenizer (always keep in memory)
        logger.info(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set pad_token if not set (important for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = None
        if self.enable_offload:
            # Load model to offload device (CPU) initially
            logger.info(f"Loading model to {self.offload_device} (will load to GPU on demand)...")
            self._load_model_to_device(self.offload_device)
        else:
            # Load model to compute device and keep it there
            logger.info(f"Loading model to {self.compute_device} (persistent mode)...")
            self._load_model_to_device(self.compute_device)

    def _load_model_to_device(self, target_device: torch.device):
        """Load model to specified device."""
        if target_device.type == "cpu":
            # For CPU, load with float32 and move to CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(target_device)
        else:
            # For GPU, use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=target_device,
            )
        self.model.eval()

    def load_to_gpu(self):
        """Load model from CPU to GPU (only effective when enable_offload=True)."""
        if not self.enable_offload:
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot load to GPU")

        logger.info(f"[HFFSFJudge] Loading model to {self.compute_device}...")

        # Convert dtype if needed (CPU uses float32, GPU uses bfloat16)
        if self.compute_device.type != "cpu" and self.model.dtype != self.dtype:
            self.model = self.model.to(dtype=self.dtype)

        # Move to GPU
        self.model = self.model.to(self.compute_device)

        # Clear CPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[HFFSFJudge] Model loaded to {self.compute_device}")

    def offload_to_cpu(self):
        """Offload model from GPU to CPU (only effective when enable_offload=True)."""
        if not self.enable_offload:
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot offload")

        logger.info(f"[HFFSFJudge] Offloading model to {self.offload_device}...")

        # Convert to float32 for CPU
        if self.offload_device.type == "cpu":
            self.model = self.model.to(dtype=torch.float32)

        # Move to CPU
        self.model = self.model.to(self.offload_device)

        # Clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"[HFFSFJudge] Model offloaded to {self.offload_device}")

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

    @torch.no_grad()
    def _compute_single_fsf(
        self,
        reasoning: str,
    ) -> float:
        """
        Compute FSF for a single reasoning text.

        Args:
            reasoning: The reasoning text to analyze

        Returns:
            FSF value between 0.0 and 1.0
        """
        # Truncate if too long
        reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning

        user_content = self.USER_PROMPT_TEMPLATE.format(reasoning=reasoning_truncated)

        try:
            # Build chat prompt
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=3500,
            ).to(self.model.device)

            # Generate (only need small output)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            return self._parse_fsf(generated_text)

        except Exception as e:
            logger.error(f"[FSFJudge] Error computing FSF: {e}")
            return 0.5  # Default on error

    @torch.no_grad()
    def compute_fsf_batch(
        self,
        reasonings: List[str],
    ) -> List[float]:
        """
        Batch computation of FSF values.

        Args:
            reasonings: List of reasoning texts

        Returns:
            List of FSF values
        """
        fsf_values = []
        for reasoning in reasonings:
            fsf_values.append(self._compute_single_fsf(reasoning))
        return fsf_values

    def compute_logic_reward(
        self,
        original_cots: List[str],
        rewrite_cots: List[str],
    ) -> Dict[str, Any]:
        """
        Compute logic reward based on FSF comparison.

        R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

        When enable_offload=True:
        - Automatically loads model to GPU before computation
        - Automatically offloads model to CPU after computation

        Args:
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings

        Returns:
            Dictionary with:
                - original_fsfs: FSF values for original CoTs
                - rewrite_fsfs: FSF values for rewritten CoTs
                - logic_rewards: Binary rewards (1 if improved, 0 otherwise)
        """
        # Auto load to GPU if offload enabled and auto_offload mode
        if self.enable_offload and self.auto_offload:
            self.load_to_gpu()

        try:
            original_fsfs = self.compute_fsf_batch(original_cots)
            rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

            # R_logic = 1 if rewrite FSF < original FSF (improvement)
            logic_rewards = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]

            return {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "logic_rewards": logic_rewards,
            }
        finally:
            # Auto offload to CPU after computation (only if auto_offload enabled)
            if self.enable_offload and self.auto_offload:
                self.offload_to_cpu()


# Global singleton
_fsf_judge: Optional[HFFSFJudge] = None


def get_fsf_judge(
    model_path: str = "./models/Qwen3-8B",
    **kwargs,
) -> HFFSFJudge:
    """Get or create FSF Judge singleton."""
    global _fsf_judge
    if _fsf_judge is None:
        _fsf_judge = HFFSFJudge(model_path=model_path, **kwargs)
    return _fsf_judge
