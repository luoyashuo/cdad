"""
Student Reward Worker

将 Student Model（含 FSF LoRA）作为 Reward Model 集成到 VERL 中
采用"时分复用"策略：在 Rollout 结束后加载，计算完成后释放

关键设计：
1. 延迟加载：只在需要计算 reward 时才加载模型
2. 显存释放：计算完成后可选择释放到 CPU
3. 与 Rollout 解耦：不影响 vLLM/SGLang 的 KV cache

集成方式：
- 方案 A：作为 custom_reward_function（推荐，简单）
- 方案 B：作为独立的 RewardModelWorker（适合大规模）
- 方案 C：使用 RewardLoopWorker（适合异步场景）
"""

from typing import Any, Optional, Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc
import json
import re

from verl import DataProto


class StudentRewardWorker:
    """
    Student Model Reward Worker

    采用延迟加载 + 显存管理策略，与 Rollout 引擎共存
    """

    # FSF 分析 prompt
    FSF_PROMPT_TEMPLATE = """<|im_start|>system
你是一个推理过程分析专家。请分析推理过程中的"失败分支"占比。
失败分支定义：尝试了某种方法但最终放弃的分支。
请输出 JSON 格式：{{"total_steps": N, "failed_steps": M, "fsf": 0.0到1.0的数值}}<|im_end|>
<|im_start|>user
分析以下推理过程的失败分支占比：

{reasoning}

只输出 JSON，不要其他文字。<|im_end|>
<|im_start|>assistant
"""

    def __init__(
        self,
        student_model_path: str,
        fsf_lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
        # 显存管理参数
        offload_to_cpu: bool = False,  # 计算完成后是否 offload 到 CPU
        lazy_load: bool = True,  # 是否延迟加载
        pin_memory: bool = True,  # CPU 端是否 pin memory（加速加载）
    ):
        """
        初始化 Student Reward Worker

        Args:
            student_model_path: Student Model 路径
            fsf_lora_path: FSF LoRA adapter 路径
            device: 目标设备
            dtype: 数据类型
            max_length: 最大序列长度
            offload_to_cpu: 计算完成后是否 offload 到 CPU（节省显存）
            lazy_load: 是否延迟加载（等到第一次调用时才加载）
            pin_memory: CPU 端是否使用 pinned memory
        """
        self.student_model_path = student_model_path
        self.fsf_lora_path = fsf_lora_path
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.offload_to_cpu = offload_to_cpu
        self.lazy_load = lazy_load
        self.pin_memory = pin_memory

        # 模型状态
        self._model_loaded = False
        self._model_on_gpu = False

        # 模型和 tokenizer
        self.tokenizer = None
        self.base_model = None
        self.fsf_model = None

        # CPU 缓存（用于 offload）
        self._cpu_state_dict = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """加载模型到 GPU"""
        if self._model_loaded and self._model_on_gpu:
            return

        print(f"[StudentRewardWorker] 加载模型: {self.student_model_path}")

        # 加载 tokenizer（只需加载一次）
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.student_model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 如果有 CPU 缓存，从缓存加载
        if self._cpu_state_dict is not None:
            print("[StudentRewardWorker] 从 CPU 缓存加载模型...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.student_model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                state_dict=self._cpu_state_dict,
            )
        else:
            # 首次加载
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.student_model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

        self.base_model.to(self.device)
        self.base_model.eval()

        # 冻结参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 加载 FSF LoRA（如果有）
        if self.fsf_lora_path:
            print(f"[StudentRewardWorker] 加载 FSF LoRA: {self.fsf_lora_path}")
            self.fsf_model = PeftModel.from_pretrained(
                self.base_model,
                self.fsf_lora_path,
                is_trainable=False,
            )
            self.fsf_model.eval()

        self._model_loaded = True
        self._model_on_gpu = True

        print(f"[StudentRewardWorker] 模型加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _offload_to_cpu(self):
        """将模型 offload 到 CPU"""
        if not self._model_on_gpu:
            return

        print("[StudentRewardWorker] Offload 模型到 CPU...")

        # 保存 state dict 到 CPU
        if self._cpu_state_dict is None and self.pin_memory:
            self._cpu_state_dict = {
                k: v.cpu().pin_memory() if self.pin_memory else v.cpu()
                for k, v in self.base_model.state_dict().items()
            }

        # 清理 GPU 上的模型
        del self.base_model
        del self.fsf_model
        self.base_model = None
        self.fsf_model = None

        # 强制 GC 和 CUDA 缓存清理
        gc.collect()
        torch.cuda.empty_cache()

        self._model_on_gpu = False

        print(f"[StudentRewardWorker] Offload 完成, 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def ensure_model_ready(self):
        """确保模型在 GPU 上就绪"""
        if not self._model_on_gpu:
            self._load_model()

    def release_gpu_memory(self):
        """释放 GPU 显存（offload 到 CPU）"""
        if self.offload_to_cpu:
            self._offload_to_cpu()

    # ==================== 认知奖励 ====================

    def compute_cognitive_reward(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算认知奖励: log P(Answer | Question, CoT)

        使用原始 Student Model（不带 LoRA adapter）
        """
        self.ensure_model_ready()

        batch_size = len(questions)

        full_texts = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: {ans}"
            for q, cot, ans in zip(questions, rewrite_cots, answers)
        ]

        prefixes = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: "
            for q, cot in zip(questions, rewrite_cots)
        ]

        prefix_enc = self.tokenizer(
            prefixes, padding=True, truncation=True,
            max_length=self.max_length - 100, return_tensors="pt"
        ).to(self.device)

        full_enc = self.tokenizer(
            full_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        prefix_lengths = prefix_enc.attention_mask.sum(dim=1)

        # 使用基座模型（不带 LoRA）
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=full_enc.input_ids,
                attention_mask=full_enc.attention_mask,
            )
            logits = outputs.logits

        log_probs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            start_idx = prefix_lengths[i].item()
            end_idx = full_enc.attention_mask[i].sum().item()

            if start_idx >= end_idx:
                log_probs[i] = -100.0
                continue

            ans_logits = logits[i, start_idx-1:end_idx-1, :]
            ans_labels = full_enc.input_ids[i, start_idx:end_idx]

            ans_log_probs = F.log_softmax(ans_logits, dim=-1)
            token_log_probs = ans_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
            log_probs[i] = token_log_probs.sum()

        return log_probs

    # ==================== FSF 奖励 ====================

    def _parse_fsf(self, text: str) -> float:
        """解析 FSF 输出"""
        try:
            match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if match:
                return float(match.group(1))
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return min(float(match.group(1)), 1.0)
        except (ValueError, AttributeError):
            pass
        return 0.5

    def compute_fsf_batch(self, reasonings: list[str]) -> list[float]:
        """
        批量计算 FSF

        使用 Student Model + FSF LoRA adapter
        """
        self.ensure_model_ready()

        if self.fsf_model is None:
            return [0.5] * len(reasonings)

        fsf_values = []

        for reasoning in reasonings:
            prompt = self.FSF_PROMPT_TEMPLATE.format(reasoning=reasoning[:2000])

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_length - 100
            ).to(self.device)

            with torch.no_grad():
                outputs = self.fsf_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            fsf_values.append(self._parse_fsf(response))

        return fsf_values

    def compute_logic_reward(
        self,
        original_cots: list[str],
        rewrite_cots: list[str],
    ) -> torch.Tensor:
        """计算逻辑奖励"""
        original_fsfs = self.compute_fsf_batch(original_cots)
        rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)

        rewards = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]
        return torch.tensor(rewards, device=self.device, dtype=self.dtype)

    # ==================== 组合奖励 ====================

    def compute_reward(
        self,
        questions: list[str],
        original_cots: list[str],
        rewrite_cots: list[str],
        answers: list[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算组合奖励: R = R_cog + λ * R_logic
        """
        # 确保模型在 GPU
        self.ensure_model_ready()

        # 计算认知奖励
        r_cog = self.compute_cognitive_reward(questions, rewrite_cots, answers)

        # 计算逻辑奖励
        if use_logic_reward and self.fsf_model is not None and original_cots and len(original_cots[0]) > 0:
            r_logic = self.compute_logic_reward(original_cots, rewrite_cots)
        else:
            r_logic = torch.zeros_like(r_cog)

        total = r_cog + lambda_logic * r_logic

        info = {
            "cognitive_rewards": r_cog.cpu().tolist(),
            "logic_rewards": r_logic.cpu().tolist(),
            "total_rewards": total.cpu().tolist(),
        }

        # 可选：释放显存
        self.release_gpu_memory()

        return total, info


# ==================== 全局单例 ====================

_student_reward_worker: Optional[StudentRewardWorker] = None


def get_student_reward_worker(
    student_model_path: str,
    fsf_lora_path: Optional[str] = None,
    **kwargs
) -> StudentRewardWorker:
    """获取或创建 StudentRewardWorker 单例"""
    global _student_reward_worker
    if _student_reward_worker is None:
        _student_reward_worker = StudentRewardWorker(
            student_model_path,
            fsf_lora_path=fsf_lora_path,
            **kwargs
        )
    return _student_reward_worker


# ==================== VERL 接口 ====================

def compute_student_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    fsf_lora_path: Optional[str] = None,
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    offload_to_cpu: bool = True,  # 默认开启 offload
    **kwargs,
) -> dict[str, Any]:
    """
    VERL 奖励函数接口

    在 Rollout 完成后调用，计算完成后自动释放显存

    Args:
        data: Verl DataProto
        tokenizer: 分词器
        student_model_path: Student Model 路径
        fsf_lora_path: FSF LoRA adapter 路径
        lambda_logic: 逻辑奖励权重
        use_logic_reward: 是否使用逻辑奖励
        offload_to_cpu: 计算完成后是否 offload 到 CPU

    Returns:
        {"reward_tensor": Tensor, "reward_extra_info": dict}
    """
    batch_size = data.batch.batch_size[0]

    # 解码
    prompts = tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
    rewrite_cots = tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    # 提取额外信息
    answers = []
    original_cots = []
    questions = []

    for i in range(batch_size):
        reward_info = data.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_info, list):
            reward_info = reward_info[i] if i < len(reward_info) else {}

        answers.append(reward_info.get("ground_truth", ""))

        orig_cot = ""
        if "original_cot" in data.non_tensor_batch:
            orig_cots_list = data.non_tensor_batch["original_cot"]
            if isinstance(orig_cots_list, list) and i < len(orig_cots_list):
                orig_cot = orig_cots_list[i]
        original_cots.append(orig_cot)

        questions.append(prompts[i])

    # 获取 Worker（支持显存管理）
    worker = get_student_reward_worker(
        student_model_path,
        fsf_lora_path=fsf_lora_path,
        offload_to_cpu=offload_to_cpu,
        lazy_load=True,
    )

    # 计算奖励
    total_rewards, info = worker.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
    )

    # 处理异常值
    total_rewards = torch.nan_to_num(total_rewards, nan=0.0).clamp(-100, 100)

    # 转为 token-level
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size, response_length,
        device=total_rewards.device,
        dtype=total_rewards.dtype
    )

    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full((batch_size,), response_length - 1, dtype=torch.long)

    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": info,
    }
