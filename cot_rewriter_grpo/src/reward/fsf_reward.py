"""
FSF (Failed Step Fraction) Judge Reward Module

Computes R_logic based on "Failed Step Fraction" (FSF) analysis.
Uses LLM-as-a-Judge to analyze the proportion of failed reasoning branches.

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

This encourages the model to produce more efficient reasoning with
fewer failed/abandoned branches.
"""

import json
import re
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFFSFJudge:
    """
    HuggingFace FSF (Failed Step Fraction) Judge

    Uses LLM-as-a-Judge to analyze the "failed branch ratio" in reasoning.
    A failed branch is defined as a reasoning path that was attempted but
    ultimately abandoned.
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
    ):
        """
        Initialize FSF Judge client.

        Args:
            model_path: Path to FSF Judge Model
            device: Device to run model on
            dtype: Precision (bfloat16 recommended for GPU)
        """
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading FSF Judge Model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

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
            ).to(self.device)

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
            print(f"[FSFJudge] Error computing FSF: {e}")
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

        Args:
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings

        Returns:
            Dictionary with:
                - original_fsfs: FSF values for original CoTs
                - rewrite_fsfs: FSF values for rewritten CoTs
                - logic_rewards: Binary rewards (1 if improved, 0 otherwise)
        """
        original_fsfs = self.compute_fsf_batch(original_cots)
        rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

        # R_logic = 1 if rewrite FSF < original FSF (improvement)
        logic_rewards = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]

        return {
            "original_fsfs": original_fsfs,
            "rewrite_fsfs": rewrite_fsfs,
            "logic_rewards": logic_rewards,
        }


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
