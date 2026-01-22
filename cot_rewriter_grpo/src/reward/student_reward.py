"""
Student Reward (Cognitive Reward) Module

Computes R_cog = log P_student(Answer | Question, Rewrite_CoT)
using a frozen Student Model deployed as a vLLM service.

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional

import httpx


class VLLMStudentRewardClient:
    """
    vLLM Student Reward Client

    Calculates cognitive reward using Student Model's log probability.
    Communicates with vLLM service via HTTP API.
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8100/v1",
        model_name: str = "Qwen2.5-3B-Instruct",
        timeout: float = 60.0,
        max_concurrent: int = 32,
    ):
        """
        Initialize vLLM Student Reward Client.

        Args:
            vllm_base_url: vLLM service base URL
            model_name: Student model name
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # vLLM endpoints
        self.completions_url = f"{self.vllm_base_url}/completions"

    async def _compute_single_log_prob_async(
        self,
        question: str,
        rewrite_cot: str,
        answer: str,
        client: httpx.AsyncClient,
    ) -> float:
        """
        Compute log P(Answer | Question, CoT) for a single sample.

        Uses vLLM's prompt_logprobs feature to get token-level log probabilities.

        Args:
            question: The input question
            rewrite_cot: The rewritten chain-of-thought
            answer: The ground truth answer
            client: HTTP client

        Returns:
            Sum of log probabilities for the answer tokens
        """
        # Build prompt with the rewritten CoT
        prompt_template = """Question: {question}

Reasoning Process:
{reasoning}

Based on the above reasoning, the answer is: """

        prefix = prompt_template.format(question=question, reasoning=rewrite_cot)
        full_text = f"{prefix}{answer}"

        async with self.semaphore:
            try:
                response = await client.post(
                    self.completions_url,
                    json={
                        "model": self.model_name,
                        "prompt": full_text,
                        "max_tokens": 1,  # No generation needed
                        "prompt_logprobs": 1,  # Return prompt logprobs
                        "echo": True,  # Return full sequence
                        "temperature": 0,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                choice = result["choices"][0]

                if "logprobs" in choice and choice["logprobs"]:
                    logprobs_data = choice["logprobs"]
                    tokens = logprobs_data.get("tokens", [])
                    token_logprobs = logprobs_data.get("token_logprobs", [])

                    if not tokens or not token_logprobs:
                        return -100.0

                    # Estimate where the answer starts
                    prefix_tokens_count = self._estimate_prefix_tokens(tokens, prefix)

                    # Extract answer logprobs
                    answer_logprobs = token_logprobs[prefix_tokens_count:]
                    answer_logprobs = [lp for lp in answer_logprobs if lp is not None]

                    if answer_logprobs:
                        return sum(answer_logprobs)

                return -100.0

            except Exception as e:
                print(f"[StudentReward] Error computing log_prob: {e}")
                return -100.0

    def _estimate_prefix_tokens(self, tokens: List[str], prefix: str) -> int:
        """
        Estimate the number of tokens in the prefix.

        Args:
            tokens: List of tokens
            prefix: The prefix string

        Returns:
            Estimated token count for the prefix
        """
        accumulated = ""
        for i, token in enumerate(tokens):
            accumulated += token
            if len(accumulated) >= len(prefix) - 10:  # Allow some tolerance
                return i + 1
        return len(tokens) // 2  # Fallback

    async def compute_cognitive_reward_batch_async(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """
        Async batch computation of cognitive rewards.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers

        Returns:
            List of cognitive reward scores
        """
        async with httpx.AsyncClient() as client:
            tasks = [
                self._compute_single_log_prob_async(q, cot, ans, client)
                for q, cot, ans in zip(questions, rewrite_cots, answers)
            ]
            return await asyncio.gather(*tasks)

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """
        Sync batch computation of cognitive rewards.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers

        Returns:
            List of cognitive reward scores
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running in async context, create new loop
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            pass

        return asyncio.run(
            self.compute_cognitive_reward_batch_async(questions, rewrite_cots, answers)
        )

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
_student_client: Optional[VLLMStudentRewardClient] = None


def get_student_client(
    vllm_base_url: str = "http://localhost:8100/v1",
    **kwargs,
) -> VLLMStudentRewardClient:
    """Get or create Student Reward client singleton."""
    global _student_client
    if _student_client is None:
        _student_client = VLLMStudentRewardClient(vllm_base_url=vllm_base_url, **kwargs)
    return _student_client
