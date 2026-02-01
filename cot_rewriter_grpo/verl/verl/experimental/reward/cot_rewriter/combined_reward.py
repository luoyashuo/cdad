"""
Combined Reward Function for CoT Rewriter GRPO

Reward 计算逻辑:
- 使用 Qwen3 chat template 格式计算 Student Model 的 log prob
- R_cog = avg(log P_student(Answer | Question, Rewrite_CoT)) + alpha_cog * min(batch内的avg log prob)
- alpha_cog 为可配置参数，默认为 0.01
- (可选) R_logic: FSF Judge 比较失败分支比例

This module provides the VERL-compatible interface for reward computation.
"""

from typing import Any, Dict, List, Optional

import torch

from verl import DataProto

from verl.experimental.reward.cot_rewriter.student_reward import HFStudentRewardClient, RemoteStudentRewardClient
from verl.experimental.reward.cot_rewriter.fsf_reward import HFFSFJudge
from verl.experimental.reward.cot_rewriter.logging_config import logger


class CombinedRewardComputer:
    """
    Combined Reward Computer

    Computes R = R_cog + lambda * R_logic using HuggingFace Transformers directly.

    Supports multiple modes:
    1. Local persistent mode: Load models to GPU and keep them there
    2. Local offload mode: Load models to CPU, temporarily move to GPU for computation
    3. Remote mode: Call separate GPU server via HTTP for student reward
    """

    def __init__(
        self,
        student_model_path: str = "./models/Qwen3-1.7B",
        fsf_model_path: Optional[str] = None,
        device: str = "cuda:3",
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
        allow_cpu: bool = True,
        use_remote_student: bool = False,
        student_server_url: str = "http://localhost:8765",
        enable_offload: bool = False,
        offload_device: str = "cpu",
        alpha_cog: float = 0.01,
    ):
        """
        Initialize Combined Reward Computer.

        Args:
            student_model_path: Path to Student Model (for local mode)
            fsf_model_path: Path to FSF Judge Model (only required if use_logic_reward=True)
            device: Device to run reward models on (should be a single GPU)
            lambda_logic: Weight for logic reward
            use_logic_reward: Whether to use logic reward
            allow_cpu: If True, allow fallback to CPU when CUDA is not available
            use_remote_student: If True, use remote HTTP server for student reward
            student_server_url: URL of the student reward server (only used if use_remote_student=True)
            enable_offload: If True, models are kept on CPU and loaded to GPU only during computation
            offload_device: Device to offload models when not in use (default: "cpu")
            alpha_cog: Weight for min log prob term in cognitive reward (default: 0.01)
                       R_cog = avg(log_prob) + alpha_cog * min(batch内的avg log_prob)
        """
        self.lambda_logic = lambda_logic
        self.use_logic_reward = use_logic_reward
        self.use_remote_student = use_remote_student
        self.enable_offload = enable_offload
        self.alpha_cog = alpha_cog

        logger.info(f"[CombinedRewardComputer] Initializing")
        logger.info(f"  Device: {device}")
        logger.info(f"  Offload Enabled: {enable_offload}")
        logger.info(f"  Alpha Cog: {alpha_cog}")
        if enable_offload:
            logger.info(f"  Offload Device: {offload_device}")

        # Initialize Student Reward client
        # Note: auto_offload=False so we can manually control load/offload timing
        # This avoids repeated load/offload overhead when processing multiple batches
        if use_remote_student:
            logger.info(f"[CombinedRewardComputer] Using remote student server at {student_server_url}")
            self.student_client = RemoteStudentRewardClient(
                server_url=student_server_url,
            )
        else:
            self.student_client = HFStudentRewardClient(
                model_path=student_model_path,
                device=device,
                allow_cpu=allow_cpu,
                enable_offload=enable_offload,
                offload_device=offload_device,
                auto_offload=False,  # Manual control for efficiency
            )

        # Initialize FSF Judge client (if needed)
        if use_logic_reward:
            if not fsf_model_path:
                raise ValueError("fsf_model_path must be provided when use_logic_reward=True")
            self.fsf_judge = HFFSFJudge(
                model_path=fsf_model_path,
                device=device,
                allow_cpu=allow_cpu,
                enable_offload=enable_offload,
                offload_device=offload_device,
                auto_offload=False,  # Manual control for efficiency
            )
        else:
            self.fsf_judge = None

    def load_models(self):
        """
        Load all reward models to GPU.
        Call this before processing batches to avoid repeated load/offload overhead.
        """
        if not self.enable_offload:
            return

        logger.info(f"[CombinedRewardComputer] Loading models to GPU...")
        if hasattr(self.student_client, 'load_to_gpu'):
            self.student_client.load_to_gpu()
        if self.fsf_judge is not None:
            self.fsf_judge.load_to_gpu()
        logger.info(f"[CombinedRewardComputer] Models loaded to GPU")

    def offload_models(self):
        """
        Offload all reward models to CPU.
        Call this after all batches are processed to free GPU memory.
        """
        if not self.enable_offload:
            return

        logger.info(f"[CombinedRewardComputer] Offloading models to CPU...")
        if hasattr(self.student_client, 'offload_to_cpu'):
            self.student_client.offload_to_cpu()
        if self.fsf_judge is not None:
            self.fsf_judge.offload_to_cpu()
        logger.info(f"[CombinedRewardComputer] Models offloaded to CPU")

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> Dict[str, Any]:
        """
        Compute combined reward.

        Reward 逻辑:
        - R_cog = avg(log P_student(Answer | Question, Rewrite_CoT)) + alpha_cog * min(batch内的avg log prob)
        - (可选) R_logic = 1 if FSF(rewrite) < FSF(original) else 0
        - Total = R_cog + lambda * R_logic

        Args:
            questions: List of questions
            original_cots: List of original CoT reasonings
            rewrite_cots: List of rewritten CoT reasonings (Actor 生成)
            answers: List of ground truth answers

        Returns:
            Dictionary with reward info and extra metrics
        """
        batch_size = len(questions)
        logger.info(f"[REWARD] Computing reward for batch_size={batch_size}")

        # Compute cognitive reward (获取 mean log prob)
        logger.info(f"[REWARD] Computing cognitive reward...")
        cog_result = self.student_client.compute_cognitive_reward_batch(
            questions=questions,
            rewrite_cots=rewrite_cots,
            answers=answers,
            original_cots=original_cots,  # 传入 original_cots 用于记录
        )
        logger.info(f"[REWARD] Cognitive reward computed")

        # 获取 rewrite_cot 条件下的 mean log prob
        rewrite_mean_log_probs = cog_result["rewrite_mean_log_probs"]
        original_mean_log_probs = cog_result["original_mean_log_probs"]
        token_log_probs = cog_result["rewrite_token_log_probs"]
        token_ids = cog_result["rewrite_token_ids"]

        # 新的 R_cog 计算: R_cog = avg(log_prob) + alpha_cog * min(batch内的avg log_prob)
        # 过滤掉无效值 (-100.0) 来计算 min
        valid_log_probs = [lp for lp in rewrite_mean_log_probs if lp > -99.0]
        if valid_log_probs:
            min_log_prob = min(valid_log_probs)
        else:
            min_log_prob = -100.0

        r_cog = [
            lp + self.alpha_cog * min_log_prob if lp > -99.0 else 0.0
            for lp in rewrite_mean_log_probs
        ]
        logger.info(f"[REWARD] R_cog formula: avg(log_prob) + {self.alpha_cog} * min(log_prob)")
        logger.info(f"[REWARD] min_log_prob in batch: {min_log_prob:.4f}")

        # Compute logic reward (if enabled)
        if (
            self.use_logic_reward
            and self.fsf_judge
            and original_cots
            and all(c for c in original_cots)
        ):
            logger.info(f"[REWARD] Computing logic reward...")
            logic_result = self.fsf_judge.compute_logic_reward(
                original_cots=original_cots,
                rewrite_cots=rewrite_cots,
            )
            r_logic = logic_result["logic_rewards"]
            original_fsfs = logic_result["original_fsfs"]
            rewrite_fsfs = logic_result["rewrite_fsfs"]
            logger.info(f"[REWARD] Logic reward computed")
        else:
            logger.info(f"[REWARD] Skipping logic reward (disabled or no original cots)")
            r_logic = [0.0] * batch_size
            original_fsfs = [0.0] * batch_size
            rewrite_fsfs = [0.0] * batch_size

        # Combine rewards: R = R_cog + lambda * R_logic
        logger.info(f"[REWARD] Combining rewards with lambda_logic={self.lambda_logic}")
        total_rewards = [c + self.lambda_logic * l for c, l in zip(r_cog, r_logic)]
        logger.info(f"[REWARD] Reward computation completed. Returning results...")

        return {
            "cognitive_rewards": r_cog,  # 0 或 1 的比较 reward
            "rewrite_mean_log_probs": rewrite_mean_log_probs,
            "original_mean_log_probs": original_mean_log_probs,
            "logic_rewards": r_logic,
            "total_rewards": total_rewards,
            "token_log_probs": token_log_probs,
            "token_ids": token_ids,
            "extra_info": {
                "original_fsfs": original_fsfs,
                "rewrite_fsfs": rewrite_fsfs,
                "lambda_logic": self.lambda_logic,
            },
        }


# Global singleton
_reward_computer: Optional[CombinedRewardComputer] = None


def get_reward_computer(**kwargs) -> CombinedRewardComputer:
    """Get or create Combined Reward Computer singleton."""
    global _reward_computer
    if _reward_computer is None:
        # Set default allow_cpu to True for Ray worker compatibility
        if "allow_cpu" not in kwargs:
            kwargs["allow_cpu"] = True
        _reward_computer = CombinedRewardComputer(**kwargs)
    return _reward_computer


def compute_cot_rewriter_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen3-1.7B",
    fsf_model_path: Optional[str] = None,
    device: str = "cuda:5",
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    allow_cpu: bool = True,
    use_remote_student: bool = False,
    student_server_url: str = "http://localhost:8765",
    enable_offload: bool = False,
    offload_device: str = "cpu",
    alpha_cog: float = 0.01,
    field_mapping: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    VERL-compatible reward function interface.

    This function is called by VERL's reward manager during training.

    Args:
        data: VERL DataProto containing batch data
        tokenizer: Tokenizer for decoding
        student_model_path: Path to Student Model
        fsf_model_path: Path to FSF Judge Model
        device: Device to run reward models on. Default is cuda:5.
        lambda_logic: Weight for logic reward
        use_logic_reward: Whether to use logic reward
        allow_cpu: If True, allow fallback to CPU when CUDA is not available (for Ray workers)
        use_remote_student: If True, use remote HTTP server for student reward computation
            (recommended for GPU usage since RewardLoopWorker doesn't have GPU access)
        student_server_url: URL of the student reward server
        enable_offload: If True, models are kept on CPU and loaded to GPU only during computation.
            This allows sharing GPU memory with training. Recommended for 4-GPU setups.
        offload_device: Device to offload models when not in use (default: "cpu")
        alpha_cog: Weight for min log prob term in cognitive reward (default: 0.01)
                   R_cog = avg(log_prob) + alpha_cog * min(batch内的avg log_prob)
        field_mapping: Optional dict to customize field names in data.
            Keys: "prompt_key", "reward_model_key", "question_key",
                  "original_cot_key", "ground_truth_key"
            Example: {"prompt_key": "problem", "ground_truth_key": "solution"}

    Returns:
        Dictionary with:
            - reward_tensor: Token-level reward tensor [batch_size, seq_len]
            - reward_extra_info: Additional metrics for logging
    """
    # Default field mapping
    default_mapping = {
        "prompt_key": "prompt",
        "reward_model_key": "reward_model",
        "question_key": "question",
        "original_cot_key": "original_cot",
        "ground_truth_key": "ground_truth",
    }

    # Merge with user-provided mapping
    if field_mapping:
        default_mapping.update(field_mapping)

    prompt_key = default_mapping["prompt_key"]
    reward_model_key = default_mapping.get("reward_model_key", "reward_model")
    question_key = default_mapping["question_key"]
    original_cot_key = default_mapping["original_cot_key"]
    ground_truth_key = default_mapping["ground_truth_key"]

    # Handle case where reward_model_key is None or empty (flat structure)
    use_flat_structure = not reward_model_key or reward_model_key == "null"

    batch_size = data.batch["responses"].shape[0]
    logger.info(f"[REWARD_FN] compute_cot_rewriter_reward called with batch_size={batch_size}")

    # Decode prompts and responses
    logger.info(f"[REWARD_FN] Decoding prompts and responses...")
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True,
    )
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True,
    )
    logger.info(f"[REWARD_FN] Decoded {len(prompts)} prompts and {len(rewrite_cots)} responses")

    # Extract metadata from non_tensor_batch
    logger.info(f"[REWARD_FN] Extracting metadata from non_tensor_batch...")
    questions = []
    answers = []
    original_cots = []

    for i in range(batch_size):
        # Get reward info from non_tensor_batch
        if use_flat_structure:
            # Use flat structure: read directly from non_tensor_batch
            reward_info = {}
        else:
            # Use nested structure: read from reward_model_key
            reward_info_data = data.non_tensor_batch.get(reward_model_key, {})
            # Handle numpy array or list
            if hasattr(reward_info_data, '__len__') and not isinstance(reward_info_data, dict):
                reward_info = reward_info_data[i] if i < len(reward_info_data) else {}
            else:
                reward_info = reward_info_data
            # Ensure reward_info is a dict
            if not isinstance(reward_info, dict):
                reward_info = {}

        # Extract ground truth answer (try mapped key first, then fallbacks)
        answer = reward_info.get(ground_truth_key, "")
        if not answer:
            answer = reward_info.get("ground_truth", "")
        if not answer:
            answer = reward_info.get("answer", "")
        # Also check top-level non_tensor_batch
        if not answer and ground_truth_key in data.non_tensor_batch:
            gt_list = data.non_tensor_batch[ground_truth_key]
            if isinstance(gt_list, list) and i < len(gt_list):
                answer = gt_list[i]
        answers.append(answer)

        # Extract original CoT
        orig_cot = ""
        # Check top-level non_tensor_batch first (may be numpy array)
        if original_cot_key in data.non_tensor_batch:
            orig_cots_data = data.non_tensor_batch[original_cot_key]
            # Handle numpy array
            if hasattr(orig_cots_data, '__len__') and i < len(orig_cots_data):
                orig_cot = orig_cots_data[i]
                if hasattr(orig_cot, 'item'):  # numpy scalar
                    orig_cot = orig_cot.item()
        # Then check reward_info
        if not orig_cot and original_cot_key in reward_info:
            orig_cot = reward_info.get(original_cot_key, "")
        # Fallback to default keys
        if not orig_cot:
            orig_cot = reward_info.get("original_cot", "")
        original_cots.append(orig_cot if orig_cot else "")

        # Extract question (from prompt or metadata)
        question = reward_info.get(question_key, "")
        if not question:
            question = reward_info.get("question", "")
        if not question:
            question = prompts[i]
        questions.append(question)

    logger.info(f"[REWARD_FN] Extracted {len(questions)} questions, {len(answers)} answers, {len(original_cots)} original_cots")

    # Debug: Check if original_cots were extracted correctly
    non_empty_cots = sum(1 for c in original_cots if c)
    logger.info(f"[REWARD_FN] Non-empty original_cots: {non_empty_cots}/{len(original_cots)}")
    if non_empty_cots == 0:
        logger.warning(f"[REWARD_FN] WARNING: No original_cots found! Check field_mapping.original_cot_key")
        logger.warning(f"[REWARD_FN] Available keys in non_tensor_batch: {list(data.non_tensor_batch.keys())}")

    # Number of samples to print for review
    print_samples = kwargs.get("print_samples", 2)

    # Get or create reward computer
    logger.info(f"[REWARD_FN] Getting reward computer...")
    reward_computer = get_reward_computer(
        student_model_path=student_model_path,
        fsf_model_path=fsf_model_path,
        device=device,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
        allow_cpu=allow_cpu,
        use_remote_student=use_remote_student,
        student_server_url=student_server_url,
        enable_offload=enable_offload,
        offload_device=offload_device,
        alpha_cog=alpha_cog,
    )

    # Load models to GPU before computing rewards (if offload enabled)
    # This is done once per batch to avoid repeated load/offload overhead
    reward_computer.load_models()

    try:
        # Compute rewards
        logger.info(f"[REWARD_FN] Calling reward_computer.compute_reward()...")
        result = reward_computer.compute_reward(
            questions=questions,
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            answers=answers,
        )
        logger.info(f"[REWARD_FN] Reward computer returned results")
    finally:
        # Offload models to CPU after computing rewards (if offload enabled)
        # This frees GPU memory for training
        reward_computer.offload_models()

    # Convert to tensor - use the same device as the input data
    logger.info(f"[REWARD_FN] Converting rewards to tensors...")
    device = data.batch["responses"].device
    total_rewards = torch.tensor(
        result["total_rewards"],
        device=device,
        dtype=torch.bfloat16,
    )

    # Handle NaN and extreme values
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0)
    total_rewards = total_rewards.clamp(-100, 100)

    # Convert to token-level rewards (assign reward to last token)
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size,
        response_length,
        device=device,
        dtype=total_rewards.dtype,
    )

    # Get actual sequence lengths from response mask
    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full(
            (batch_size,),
            response_length - 1,
            dtype=torch.long,
            device=device,
        )

    # Assign reward to the last valid token of each sequence
    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    # Print rollout samples with rewards for review
    if print_samples > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"[ROLLOUT OUTPUT WITH REWARDS] Showing first {min(print_samples, batch_size)} samples:")
        logger.info(f"{'='*60}")
        for i in range(min(print_samples, batch_size)):
            logger.info(f"\n[Sample {i}]")
            logger.info(f"[question] {prompts[i][:500]}..." if len(prompts[i]) > 500 else f"[question] {prompts[i]}")
            logger.info(f"[rollout_output] {rewrite_cots[i][:1000]}..." if len(rewrite_cots[i]) > 1000 else f"[rollout_output] {rewrite_cots[i]}")
            if i < len(original_cots) and original_cots[i]:
                logger.info(f"[old_cot] {original_cots[i][:500]}..." if len(original_cots[i]) > 500 else f"[old_cot] {original_cots[i]}")
            logger.info(f"[total_reward] {result['total_rewards'][i]}")
            logger.info(f"[cognitive_reward] {result['cognitive_rewards'][i]}")
            logger.info("-" * 50)
        logger.info(f"{'='*60}\n")

    logger.info(f"[REWARD_FN] Building reward_extra_info dict...")
    # Note: Do NOT include variable-length fields (token_log_probs, token_ids) in reward_extra_info
    # VERL's agent_loop.py:778 tries to convert all extra_info values to numpy arrays,
    # which fails for variable-length lists (different samples have different token counts).
    # Error: "ValueError: setting an array element with a sequence. The requested array
    # has an inhomogeneous shape after 2 dimensions."
    #
    # IMPORTANT: Only include list-type values with length == batch_size in reward_extra_info.
    # VERL's ray_trainer.py:1226 converts all reward_extra_info values to numpy arrays and adds
    # them to non_tensor_batch. Scalar values (like mean_*) become 0-dim arrays, which causes
    # np.array_split to fail in protocol.py:897 with "IndexError: tuple index out of range".
    # Scalar values should be logged separately, not included in reward_extra_info.
    reward_result = {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            # Sample-level info (lists with length == batch_size) - REQUIRED for VERL
            "cognitive_rewards": result["cognitive_rewards"],  # 0 或 1 的比较 reward
            "rewrite_mean_log_probs": result["rewrite_mean_log_probs"],
            "original_mean_log_probs": result["original_mean_log_probs"],
            "logic_rewards": result["logic_rewards"],
            "total_rewards": result["total_rewards"],
            "original_fsfs": result["extra_info"]["original_fsfs"],
            "rewrite_fsfs": result["extra_info"]["rewrite_fsfs"],
            # NOTE: Scalar values (mean_*, lambda_logic) are intentionally excluded from
            # reward_extra_info to avoid 0-dim numpy array issues in VERL's chunk operation.
        },
    }

    logger.info(f"[REWARD_FN] reward_extra_info keys: {list(reward_result['reward_extra_info'].keys())}")
    logger.info(f"[REWARD_FN] Sample-level info (lists): {[k for k, v in reward_result['reward_extra_info'].items() if isinstance(v, list)]}")
    logger.info(f"[REWARD_FN] Batch-level info (scalars): {[k for k, v in reward_result['reward_extra_info'].items() if not isinstance(v, list)]}")
    logger.info(f"[REWARD_FN] compute_cot_rewriter_reward completed, returning result")

    return reward_result
