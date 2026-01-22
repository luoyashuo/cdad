"""
FSF (Failed Step Fraction) Judge Reward Module

Computes R_logic based on "Failed Step Fraction" (FSF) analysis.
Uses LLM-as-a-Judge to analyze the proportion of failed reasoning branches.

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

This encourages the model to produce more efficient reasoning with
fewer failed/abandoned branches.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import httpx


class VLLMFSFJudge:
    """
    vLLM FSF (Failed Step Fraction) Judge

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
        vllm_base_url: str = "http://localhost:8101/v1",
        model_name: str = "Qwen2.5-7B-Instruct",
        timeout: float = 60.0,
        max_concurrent: int = 32,
    ):
        """
        Initialize FSF Judge client.

        Args:
            vllm_base_url: vLLM service base URL
            model_name: Judge model name
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # vLLM chat endpoint
        self.chat_url = f"{self.vllm_base_url}/chat/completions"

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
            num_match = re.search(r'(\d+\.?\d*)', text)
            if num_match:
                value = float(num_match.group(1))
                return min(value, 1.0)

        except Exception:
            pass

        return 0.5  # Default to 0.5 if parsing fails

    async def _compute_single_fsf_async(
        self,
        reasoning: str,
        client: httpx.AsyncClient,
    ) -> float:
        """
        Compute FSF for a single reasoning text.

        Args:
            reasoning: The reasoning text to analyze
            client: HTTP client

        Returns:
            FSF value between 0.0 and 1.0
        """
        # Truncate if too long
        reasoning_truncated = reasoning[:3000] if len(reasoning) > 3000 else reasoning

        user_content = self.USER_PROMPT_TEMPLATE.format(reasoning=reasoning_truncated)

        async with self.semaphore:
            try:
                response = await client.post(
                    self.chat_url,
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "max_tokens": 150,
                        "temperature": 0.1,  # Low temperature for consistency
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                text = result["choices"][0]["message"]["content"]
                return self._parse_fsf(text)

            except Exception as e:
                print(f"[FSFJudge] Error computing FSF: {e}")
                return 0.5  # Default on error

    async def compute_fsf_batch_async(
        self,
        reasonings: List[str],
    ) -> List[float]:
        """
        Async batch computation of FSF values.

        Args:
            reasonings: List of reasoning texts

        Returns:
            List of FSF values
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                self._compute_single_fsf_async(r, client)
                for r in reasonings
            ]
            return await asyncio.gather(*tasks)

    def compute_fsf_batch(
        self,
        reasonings: List[str],
    ) -> List[float]:
        """
        Sync batch computation of FSF values.

        Args:
            reasonings: List of reasoning texts

        Returns:
            List of FSF values
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            pass

        return asyncio.run(self.compute_fsf_batch_async(reasonings))

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
        logic_rewards = [
            1.0 if r < o else 0.0
            for o, r in zip(original_fsfs, rewrite_fsfs)
        ]

        return {
            "original_fsfs": original_fsfs,
            "rewrite_fsfs": rewrite_fsfs,
            "logic_rewards": logic_rewards,
        }

    async def health_check(self) -> bool:
        """Check if the vLLM service is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vllm_base_url}/models",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False


# Global singleton
_fsf_judge: Optional[VLLMFSFJudge] = None


def get_fsf_judge(
    vllm_base_url: str = "http://localhost:8101/v1",
    **kwargs,
) -> VLLMFSFJudge:
    """Get or create FSF Judge singleton."""
    global _fsf_judge
    if _fsf_judge is None:
        _fsf_judge = VLLMFSFJudge(vllm_base_url=vllm_base_url, **kwargs)
    return _fsf_judge
