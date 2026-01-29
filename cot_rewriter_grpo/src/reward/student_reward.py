"""
Student Reward (Cognitive Reward) Module

Computes R_cog = log P_student(Answer | Question, Rewrite_CoT)
using a frozen Student Model with HuggingFace Transformers directly.

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence.
"""

import torch
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFStudentRewardClient:
    """
    HuggingFace Student Reward Client

    Calculates cognitive reward using Student Model's log probability.
    Uses HuggingFace Transformers directly (no vLLM dependency).
    """

    def __init__(
        self,
        model_path: str = "./models/Qwen3-1.7B",
        device: str = "cuda:3",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize HuggingFace Student Reward Client.

        Args:
            model_path: Path to Student Model
            device: Device to run model on
            dtype: Precision (bfloat16 recommended for GPU)
        """
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Student Model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    @torch.no_grad()
    def _compute_single_log_prob(
        self,
        question: str,
        rewrite_cot: str,
        answer: str,
    ) -> float:
        """
        Compute log P(Answer | Question, CoT) for a single sample.

        Args:
            question: The input question
            rewrite_cot: The rewritten chain-of-thought
            answer: The ground truth answer

        Returns:
            Sum of log probabilities for the answer tokens
        """
        prompt_template = """Question: {question}

Reasoning Process:
{reasoning}

Based on the above reasoning, the answer is: """

        prefix = prompt_template.format(question=question, reasoning=rewrite_cot)
        full_text = f"{prefix}{answer}"

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Forward pass to get logits
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        token_ids = inputs["input_ids"][:, 1:]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, token_ids.unsqueeze(2)).squeeze(2)

        # Estimate where the answer starts
        prefix_token_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"][0]
        prefix_len = len(prefix_token_ids) - 1  # exclude eos token

        # Extract answer logprobs
        answer_log_probs = token_log_probs[0, prefix_len:].tolist()
        if answer_log_probs:
            return sum(answer_log_probs)

        return -100.0

    @torch.no_grad()
    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """
        Batch computation of cognitive rewards.

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers

        Returns:
            List of cognitive reward scores
        """
        rewards = []
        for q, cot, ans in zip(questions, rewrite_cots, answers):
            try:
                reward = self._compute_single_log_prob(q, cot, ans)
                rewards.append(reward)
            except Exception as e:
                print(f"[StudentReward] Error computing log_prob: {e}")
                rewards.append(-100.0)
        return rewards


# Global singleton
_student_client: Optional[HFStudentRewardClient] = None


def get_student_client(
    model_path: str = "./models/Qwen3-1.7B",
    **kwargs,
) -> HFStudentRewardClient:
    """Get or create Student Reward client singleton."""
    global _student_client
    if _student_client is None:
        _student_client = HFStudentRewardClient(model_path=model_path, **kwargs)
    return _student_client
