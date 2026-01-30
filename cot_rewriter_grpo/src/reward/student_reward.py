"""
Student Reward (Cognitive Reward) Module

Computes R_cog = log P_student(Answer | Question, Rewrite_CoT)
using a frozen Student Model with HuggingFace Transformers directly.

The cognitive reward encourages the rewritten CoT to help the
Student Model predict the correct answer with higher confidence.

Supports two modes:
1. Local mode: Load model directly (uses CPU if GPU not available in Ray worker)
2. Remote mode: Call a separate GPU server via HTTP (recommended for GPU usage)
"""

import torch
import requests
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


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
            print(f"[HFStudentRewardClient] CUDA is not available. Using CPU as fallback.")
            return torch.device("cpu")
        else:
            raise RuntimeError(
                "[HFStudentRewardClient] CUDA is not available. "
                "Student Reward model requires GPU to run. "
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
            print(f"[HFStudentRewardClient] Preferred device {preferred_device} not available, using cuda:0")
            return device
        except RuntimeError:
            if allow_cpu:
                print(f"[HFStudentRewardClient] No CUDA device available, using CPU as fallback")
                return torch.device("cpu")
            raise RuntimeError(
                f"[HFStudentRewardClient] No CUDA device available. "
                f"Preferred device {preferred_device} failed with: {e}"
            ) from e


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
        allow_cpu: bool = True,
    ):
        """
        Initialize HuggingFace Student Reward Client.

        Args:
            model_path: Path to Student Model
            device: Preferred device to run model on (will fall back to CPU if not available)
            dtype: Precision (bfloat16 recommended for GPU, float32 for CPU)
            allow_cpu: If True, allow fallback to CPU when CUDA is not available
        """
        # Get available device (fallback to CPU if CUDA not available and allow_cpu=True)
        self.device = get_available_device(device, allow_cpu=allow_cpu)
        
        # Use float32 for CPU (bfloat16 not well supported on CPU)
        if self.device.type == "cpu":
            self.dtype = torch.float32
            print(f"[HFStudentRewardClient] Using float32 for CPU")
        else:
            self.dtype = dtype

        print(f"Loading Student Model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad_token if not set (important for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate device mapping
        if self.device.type == "cpu":
            # For CPU, don't use device_map, load to CPU directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            )
            self.model = self.model.to(self.device)
        else:
            # For GPU, use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
        self.model.eval()

    @torch.no_grad()
    def _compute_single_log_prob(
        self,
        question: str,
        cot: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Compute log P(Answer | Question, CoT) for a single sample.

        Uses Qwen3 chat template with reasoning_content field:
        - User message: question
        - Assistant message: answer with reasoning_content=cot

        Format:
            <|im_start|>user
            {question}<|im_end|>
            <|im_start|>assistant
            <think>
            {cot}
            </think>

            {answer}<|im_end|>

        Args:
            question: The input question
            cot: The chain-of-thought reasoning (will be in <think> block)
            answer: The ground truth answer

        Returns:
            dict with:
                - mean_log_prob: float, answer 部分 token logprob 的平均值
                - sum_log_prob: float, answer 部分 token logprob 的总和
                - token_log_probs: List[float], 每个 answer token 的 logprob
                - token_ids: List[int], answer 部分的 token IDs
        """
        # 使用 Qwen3 chat template 格式
        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer, 'reasoning_content': cot}
        ]

        # 应用 chat template
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 构建 prefix（不包含 answer）用于定位 answer 开始位置
        # prefix 格式: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n{cot}\n</think>\n\n
        prefix_messages = [
            {'role': 'user', 'content': question},
        ]
        prefix_base = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,  # 添加 <|im_start|>assistant\n
        )
        # 手动添加 <think> 部分
        prefix_with_think = prefix_base + f"<think>\n{cot}\n</think>\n\n"

        # Tokenize 完整文本
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        # Forward pass to get logits
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        all_token_ids = inputs["input_ids"][:, 1:]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, all_token_ids.unsqueeze(2)).squeeze(2)

        # 计算 prefix 的 token 长度（用于定位 answer 开始位置）
        prefix_token_ids = self.tokenizer(
            prefix_with_think,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )["input_ids"][0]
        prefix_len = len(prefix_token_ids) - 1  # -1 因为 logits 是 shifted

        # Extract answer token logprobs and IDs
        answer_log_probs = token_log_probs[0, prefix_len:].tolist()
        answer_token_ids = all_token_ids[0, prefix_len:].tolist()

        # 移除末尾的特殊 token（如 <|im_end|>）
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_token_id = self.tokenizer.eos_token_id
        special_ids = {im_end_token_id, eos_token_id}
        if im_end_token_id is None:
            special_ids = {eos_token_id}

        # 从末尾移除特殊 token
        while answer_token_ids and answer_token_ids[-1] in special_ids:
            answer_token_ids.pop()
            answer_log_probs.pop()

        if answer_log_probs:
            return {
                "mean_log_prob": sum(answer_log_probs) / len(answer_log_probs),
                "sum_log_prob": sum(answer_log_probs),
                "token_log_probs": answer_log_probs,
                "token_ids": answer_token_ids,
            }

        return {
            "mean_log_prob": -100.0,
            "sum_log_prob": -100.0,
            "token_log_probs": [],
            "token_ids": [],
        }

    @torch.no_grad()
    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch computation of cognitive rewards.

        计算 rewrite_cot 和 original_cot 条件下 answer 的 mean log prob，
        然后比较两者来决定 reward。

        Reward 逻辑:
            如果 rewrite_mean_log_prob > original_mean_log_prob → reward = 1
            否则 → reward = 0

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs (Actor 生成)
            answers: List of ground truth answers
            original_cots: List of original CoTs (用于比较，可选)

        Returns:
            dict with:
                - rewards: List[float], 0 或 1 的 reward
                - rewrite_mean_log_probs: List[float], rewrite_cot 条件下的 mean log prob
                - original_mean_log_probs: List[float], original_cot 条件下的 mean log prob
                - rewrite_token_log_probs: List[List[float]], rewrite 的 token-level logprob
                - rewrite_token_ids: List[List[int]], rewrite 的 answer token IDs
        """
        rewards = []
        rewrite_mean_log_probs = []
        original_mean_log_probs = []
        rewrite_token_log_probs = []
        rewrite_token_ids = []

        for i, (q, rewrite_cot, ans) in enumerate(zip(questions, rewrite_cots, answers)):
            try:
                # 计算 rewrite_cot 条件下的 log prob
                rewrite_result = self._compute_single_log_prob(q, rewrite_cot, ans)
                rewrite_mean = rewrite_result["mean_log_prob"]
                rewrite_mean_log_probs.append(rewrite_mean)
                rewrite_token_log_probs.append(rewrite_result["token_log_probs"])
                rewrite_token_ids.append(rewrite_result["token_ids"])

                # 如果提供了 original_cot，计算比较
                if original_cots and i < len(original_cots) and original_cots[i]:
                    original_result = self._compute_single_log_prob(q, original_cots[i], ans)
                    original_mean = original_result["mean_log_prob"]
                    original_mean_log_probs.append(original_mean)

                    # Reward: 如果 rewrite 的 log prob 更高（更好），给 reward = 1
                    # log prob 越接近 0 越好，所以 > 表示更好
                    reward = 1.0 if rewrite_mean > original_mean else 0.0
                    rewards.append(reward)
                else:
                    # 没有 original_cot，直接使用 rewrite 的 mean log prob 作为 reward
                    original_mean_log_probs.append(-100.0)
                    rewards.append(rewrite_mean)  # 直接用 log prob 作为 reward

            except Exception as e:
                print(f"[StudentReward] Error computing log_prob for sample {i}: {e}")
                rewards.append(0.0)
                rewrite_mean_log_probs.append(-100.0)
                original_mean_log_probs.append(-100.0)
                rewrite_token_log_probs.append([])
                rewrite_token_ids.append([])

        return {
            "rewards": rewards,
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "rewrite_token_log_probs": rewrite_token_log_probs,
            "rewrite_token_ids": rewrite_token_ids,
            # 向后兼容
            "log_probs": rewrite_mean_log_probs,
            "token_log_probs": rewrite_token_log_probs,
            "token_ids": rewrite_token_ids,
        }


class RemoteStudentRewardClient:
    """
    Remote Student Reward Client

    Calls a separate HTTP server running on GPU to compute log probabilities.
    This allows reward computation to run on GPU even when the calling process
    (RewardLoopWorker) doesn't have GPU access.

    Usage:
        1. Start the server: CUDA_VISIBLE_DEVICES=5 python -m src.reward.student_reward_server --port 8765
        2. Use this client in reward computation
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8765",
        timeout: float = 30.0,
    ):
        """
        Initialize Remote Student Reward Client.

        Args:
            server_url: URL of the student reward server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._check_server_health()

    def _check_server_health(self):
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            print(f"[RemoteStudentRewardClient] Server healthy: {health}")
        except Exception as e:
            print(f"[RemoteStudentRewardClient] Warning: Server health check failed: {e}")
            print(f"[RemoteStudentRewardClient] Make sure the server is running at {self.server_url}")

    def compute_cognitive_reward_batch(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        original_cots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch computation of cognitive rewards via HTTP.

        如果提供了 original_cots，会比较 rewrite 和 original 的 mean log prob：
        - 如果 rewrite_mean > original_mean → reward = 1
        - 否则 → reward = 0

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs
            answers: List of ground truth answers
            original_cots: List of original CoTs (用于比较，可选)

        Returns:
            dict with:
                - rewards: List[float], 0 或 1 的 reward
                - rewrite_mean_log_probs: List[float], rewrite 的 mean log prob
                - original_mean_log_probs: List[float], original 的 mean log prob
                - rewrite_token_log_probs: List[List[float]], token-level logprob
                - rewrite_token_ids: List[List[int]], token IDs
        """
        try:
            request_data = {
                "questions": questions,
                "rewrite_cots": rewrite_cots,
                "answers": answers,
            }
            if original_cots:
                request_data["original_cots"] = original_cots

            response = requests.post(
                f"{self.server_url}/compute_log_probs",
                json=request_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return {
                "rewards": result["rewards"],
                "rewrite_mean_log_probs": result["rewrite_mean_log_probs"],
                "original_mean_log_probs": result["original_mean_log_probs"],
                "rewrite_token_log_probs": result["rewrite_token_log_probs"],
                "rewrite_token_ids": result["rewrite_token_ids"],
                # 向后兼容
                "log_probs": result["log_probs"],
                "token_log_probs": result["token_log_probs"],
                "token_ids": result["token_ids"],
            }
        except Exception as e:
            print(f"[RemoteStudentRewardClient] Error calling server: {e}")
            batch_size = len(questions)
            return {
                "rewards": [0.0] * batch_size,
                "rewrite_mean_log_probs": [-100.0] * batch_size,
                "original_mean_log_probs": [-100.0] * batch_size,
                "rewrite_token_log_probs": [[] for _ in range(batch_size)],
                "rewrite_token_ids": [[] for _ in range(batch_size)],
                "log_probs": [-100.0] * batch_size,
                "token_log_probs": [[] for _ in range(batch_size)],
                "token_ids": [[] for _ in range(batch_size)],
            }


# Global singleton
_student_client: Optional[HFStudentRewardClient] = None
_remote_student_client: Optional[RemoteStudentRewardClient] = None


def get_student_client(
    model_path: str = "./models/Qwen3-1.7B",
    use_remote: bool = False,
    server_url: str = "http://localhost:8765",
    **kwargs,
) -> HFStudentRewardClient | RemoteStudentRewardClient:
    """
    Get or create Student Reward client singleton.

    Args:
        model_path: Path to the student model (for local mode)
        use_remote: If True, use remote HTTP server instead of local model
        server_url: URL of the remote server (only used if use_remote=True)
        **kwargs: Additional arguments for HFStudentRewardClient

    Returns:
        Student reward client (local or remote)
    """
    global _student_client, _remote_student_client

    if use_remote:
        if _remote_student_client is None:
            _remote_student_client = RemoteStudentRewardClient(server_url=server_url)
        return _remote_student_client
    else:
        if _student_client is None:
            _student_client = HFStudentRewardClient(model_path=model_path, **kwargs)
        return _student_client
