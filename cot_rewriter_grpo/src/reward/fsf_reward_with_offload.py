"""
FSF Judge with Load/Offload Support

这个版本支持动态加载和卸载模型，避免占用 GPU 显存。
"""

import json
import re
import torch
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_available_device(preferred_device: str = "cuda:0", allow_cpu: bool = True) -> torch.device:
    """Get available device for model loading."""
    if not torch.cuda.is_available():
        if allow_cpu:
            print(f"[HFFSFJudge] CUDA is not available. Using CPU as fallback.")
            return torch.device("cpu")
        else:
            raise RuntimeError("[HFFSFJudge] CUDA is not available.")

    try:
        device = torch.device(preferred_device)
        _ = torch.tensor([1.0], device=device)
        return device
    except RuntimeError as e:
        try:
            device = torch.device("cuda:0")
            _ = torch.tensor([1.0], device=device)
            print(f"[HFFSFJudge] Preferred device {preferred_device} not available, using cuda:0")
            return device
        except RuntimeError:
            if allow_cpu:
                print(f"[HFFSFJudge] No CUDA device available, using CPU as fallback")
                return torch.device("cpu")
            raise RuntimeError(f"[HFFSFJudge] No CUDA device available.") from e


class HFFSFJudgeWithOffload:
    """
    FSF Judge with Load/Offload Support

    支持两种模式：
    1. 持久化模式 (enable_offload=False)：模型始终在 GPU 上
    2. Load/Offload 模式 (enable_offload=True)：模型初始在 CPU，计算时临时加载到 GPU
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
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        allow_cpu: bool = True,
        enable_offload: bool = True,
        offload_device: str = "cpu",
    ):
        """
        Initialize FSF Judge with Offload Support.

        Args:
            model_path: Path to FSF Judge Model
            device: GPU device to load model on during computation
            dtype: Precision (bfloat16 recommended for GPU, float32 for CPU)
            allow_cpu: If True, allow fallback to CPU when CUDA is not available
            enable_offload: If True, enable load/offload mechanism
            offload_device: Device to offload model when not in use (default: "cpu")
        """
        self.model_path = model_path
        self.enable_offload = enable_offload
        self.offload_device = torch.device(offload_device)

        # Determine compute device
        if enable_offload:
            try:
                self.compute_device = torch.device(device)
                _ = torch.tensor([1.0], device=self.compute_device)
            except RuntimeError:
                if allow_cpu:
                    print(f"[HFFSFJudge] Device {device} not available, using CPU")
                    self.compute_device = torch.device("cpu")
                    enable_offload = False
                else:
                    raise
        else:
            self.compute_device = get_available_device(device, allow_cpu=allow_cpu)

        # Use float32 for CPU
        if self.compute_device.type == "cpu":
            self.dtype = torch.float32
            print(f"[HFFSFJudge] Using float32 for CPU")
        else:
            self.dtype = dtype

        print(f"[HFFSFJudge] Initializing FSF Judge")
        print(f"  Model Path: {model_path}")
        print(f"  Compute Device: {self.compute_device}")
        print(f"  Offload Enabled: {self.enable_offload}")
        if self.enable_offload:
            print(f"  Offload Device: {self.offload_device}")

        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = None
        if self.enable_offload:
            print(f"Loading model to {self.offload_device} (will load to GPU on demand)...")
            self._load_model_to_device(self.offload_device)
        else:
            print(f"Loading model to {self.compute_device} (persistent mode)...")
            self._load_model_to_device(self.compute_device)

    def _load_model_to_device(self, target_device: torch.device):
        """Load model to specified device."""
        if target_device.type == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(target_device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=target_device,
            )
        self.model.eval()

    def load_to_gpu(self):
        """Load model from CPU to GPU."""
        if not self.enable_offload:
            print("[HFFSFJudge] Offload disabled, model already on GPU")
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot load to GPU")

        print(f"[HFFSFJudge] Loading model to {self.compute_device}...")

        if self.compute_device.type != "cpu" and self.model.dtype != self.dtype:
            self.model = self.model.to(dtype=self.dtype)

        self.model = self.model.to(self.compute_device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[HFFSFJudge] Model loaded to {self.compute_device}")

    def offload_to_cpu(self):
        """Offload model from GPU to CPU."""
        if not self.enable_offload:
            print("[HFFSFJudge] Offload disabled, model stays on GPU")
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot offload")

        print(f"[HFFSFJudge] Offloading model to {self.offload_device}...")

        if self.offload_device.type == "cpu":
            self.model = self.model.to(dtype=torch.float32)

        self.model = self.model.to(self.offload_device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[HFFSFJudge] Model offloaded to {self.offload_device}")

    def _parse_fsf(self, text: str) -> float:
        """Parse FSF value from model output."""
        try:
            json_match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if json_match:
                return float(json_match.group(1))

            try:
                data = json.loads(text.strip())
                if "fsf" in data:
                    return float(data["fsf"])
            except json.JSONDecodeError:
                pass

            num_match = re.search(r"(\d+\.?\d*)", text)
            if num_match:
                value = float(num_match.group(1))
                return min(value, 1.0)

        except Exception:
            pass

        return 0.5

    @torch.no_grad()
    def _compute_single_fsf(self, reasoning: str) -> float:
        """Compute FSF for a single reasoning text."""
        reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning
        user_content = self.USER_PROMPT_TEMPLATE.format(reasoning=reasoning_truncated)

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=3500,
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            return self._parse_fsf(generated_text)

        except Exception as e:
            print(f"[FSFJudge] Error computing FSF: {e}")
            return 0.5

    @torch.no_grad()
    def compute_fsf_batch(self, reasonings: List[str]) -> List[float]:
        """Batch computation of FSF values."""
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

        自动处理 load/offload：
        - 计算前自动加载模型到 GPU
        - 计算后自动卸载模型到 CPU

        Args:
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings

        Returns:
            Dictionary with FSF values and logic rewards
        """
        # Auto load to GPU if offload enabled
        if self.enable_offload:
            self.load_to_gpu()

        try:
            original_fsfs = self.compute_fsf_batch(original_cots)
            rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

            logic_rewards = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]

            return {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "logic_rewards": logic_rewards,
            }
        finally:
            # Auto offload to CPU after computation
            if self.enable_offload:
                self.offload_to_cpu()


# Global singleton
_fsf_judge: Optional[HFFSFJudgeWithOffload] = None


def get_fsf_judge_with_offload(
    model_path: str = "./models/Qwen3-8B",
    **kwargs,
) -> HFFSFJudgeWithOffload:
    """Get or create FSF Judge singleton with offload support."""
    global _fsf_judge
    if _fsf_judge is None:
        _fsf_judge = HFFSFJudgeWithOffload(model_path=model_path, **kwargs)
    return _fsf_judge
