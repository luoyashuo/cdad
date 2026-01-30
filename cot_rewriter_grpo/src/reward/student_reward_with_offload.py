"""
Student Reward with Load/Offload Support

这个版本支持动态加载和卸载模型，避免占用 GPU 显存。
使用场景：在 GRPO 训练过程中，只在计算 reward 时临时加载模型到 GPU。
"""

import torch
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc


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


class HFStudentRewardClientWithOffload:
    """
    HuggingFace Student Reward Client with Load/Offload Support

    支持两种模式：
    1. 持久化模式 (enable_offload=False)：模型始终在 GPU 上
    2. Load/Offload 模式 (enable_offload=True)：模型初始在 CPU，计算时临时加载到 GPU

    Load/Offload 模式适用于：
    - GPU 显存紧张的情况
    - Reward 计算不频繁的场景
    - 需要在多个任务之间共享 GPU 的场景
    """

    def __init__(
        self,
        model_path: str = "./models/Qwen3-1.7B",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        allow_cpu: bool = True,
        enable_offload: bool = True,
        offload_device: str = "cpu",
    ):
        """
        Initialize HuggingFace Student Reward Client with Offload Support.

        Args:
            model_path: Path to Student Model
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
            # In offload mode, always use the specified device for computation
            try:
                self.compute_device = torch.device(device)
                # Test if device is accessible
                _ = torch.tensor([1.0], device=self.compute_device)
            except RuntimeError:
                if allow_cpu:
                    print(f"[HFStudentRewardClient] Device {device} not available, using CPU")
                    self.compute_device = torch.device("cpu")
                    enable_offload = False  # Disable offload if using CPU
                else:
                    raise
        else:
            # In persistent mode, use get_available_device logic
            self.compute_device = get_available_device(device, allow_cpu=allow_cpu)

        # Use float32 for CPU (bfloat16 not well supported on CPU)
        if self.compute_device.type == "cpu":
            self.dtype = torch.float32
            print(f"[HFStudentRewardClient] Using float32 for CPU")
        else:
            self.dtype = dtype

        print(f"[HFStudentRewardClient] Initializing Student Reward Client")
        print(f"  Model Path: {model_path}")
        print(f"  Compute Device: {self.compute_device}")
        print(f"  Offload Enabled: {self.enable_offload}")
        if self.enable_offload:
            print(f"  Offload Device: {self.offload_device}")

        # Load tokenizer (always keep in memory)
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set pad_token if not set (important for batch processing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = None
        if self.enable_offload:
            # Load model to offload device (CPU) initially
            print(f"Loading model to {self.offload_device} (will load to GPU on demand)...")
            self._load_model_to_device(self.offload_device)
        else:
            # Load model to compute device and keep it there
            print(f"Loading model to {self.compute_device} (persistent mode)...")
            self._load_model_to_device(self.compute_device)

    def _load_model_to_device(self, target_device: torch.device):
        """Load model to specified device."""
        if target_device.type == "cpu":
            # For CPU, load with float32 and move to CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU doesn't support bfloat16 well
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
        """
        Load model from CPU to GPU.
        只在 enable_offload=True 时有效。
        """
        if not self.enable_offload:
            print("[HFStudentRewardClient] Offload disabled, model already on GPU")
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot load to GPU")

        print(f"[HFStudentRewardClient] Loading model to {self.compute_device}...")

        # Convert dtype if needed (CPU uses float32, GPU uses bfloat16)
        if self.compute_device.type != "cpu" and self.model.dtype != self.dtype:
            self.model = self.model.to(dtype=self.dtype)

        # Move to GPU
        self.model = self.model.to(self.compute_device)

        # Clear CPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[HFStudentRewardClient] Model loaded to {self.compute_device}")

    def offload_to_cpu(self):
        """
        Offload model from GPU to CPU.
        只在 enable_offload=True 时有效。
        """
        if not self.enable_offload:
            print("[HFStudentRewardClient] Offload disabled, model stays on GPU")
            return

        if self.model is None:
            raise RuntimeError("Model is None, cannot offload")

        print(f"[HFStudentRewardClient] Offloading model to {self.offload_device}...")

        # Convert to float32 for CPU
        if self.offload_device.type == "cpu":
            self.model = self.model.to(dtype=torch.float32)

        # Move to CPU
        self.model = self.model.to(self.offload_device)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[HFStudentRewardClient] Model offloaded to {self.offload_device}")

    @torch.no_grad()
    def _compute_single_log_prob(
        self,
        question: str,
        cot: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Compute log P(Answer | Question, CoT) for a single sample.

        Uses Qwen3 chat template with reasoning_content field.

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
        prefix_messages = [
            {'role': 'user', 'content': question},
        ]
        prefix_base = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_with_think = prefix_base + f"<think>\n{cot}\n</think>\n\n"

        # Tokenize 完整文本
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Forward pass to get logits
        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        all_token_ids = inputs["input_ids"][:, 1:]

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, all_token_ids.unsqueeze(2)).squeeze(2)

        # 计算 prefix 的 token 长度
        prefix_token_ids = self.tokenizer(
            prefix_with_think,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )["input_ids"][0]
        prefix_len = len(prefix_token_ids) - 1

        # Extract answer token logprobs and IDs
        answer_log_probs = token_log_probs[0, prefix_len:].tolist()
        answer_token_ids = all_token_ids[0, prefix_len:].tolist()

        # 移除末尾的特殊 token
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_token_id = self.tokenizer.eos_token_id
        special_ids = {im_end_token_id, eos_token_id}
        if im_end_token_id is None:
            special_ids = {eos_token_id}

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

        自动处理 load/offload：
        - 计算前自动加载模型到 GPU
        - 计算后自动卸载模型到 CPU

        Args:
            questions: List of questions
            rewrite_cots: List of rewritten CoTs (Actor 生成)
            answers: List of ground truth answers
            original_cots: List of original CoTs (用于比较，可选)

        Returns:
            dict with rewards and metrics
        """
        # Auto load to GPU if offload enabled
        if self.enable_offload:
            self.load_to_gpu()

        try:
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
                        reward = 1.0 if rewrite_mean > original_mean else 0.0
                        rewards.append(reward)
                    else:
                        original_mean_log_probs.append(-100.0)
                        rewards.append(rewrite_mean)

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
        finally:
            # Auto offload to CPU after computation
            if self.enable_offload:
                self.offload_to_cpu()
