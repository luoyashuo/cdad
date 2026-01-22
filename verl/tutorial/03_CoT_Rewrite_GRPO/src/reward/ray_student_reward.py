"""
Ray Student Reward Worker

使用 Ray Actor 将 Student Model 部署在独立的 GPU/进程上
与 Policy Model 的 Rollout 完全隔离，避免显存竞争

架构：
┌────────────────────────────────────────────────────────────────┐
│ GPU 0-2: Policy Model + Rollout                                │
│  ┌──────────────┐  ┌────────────────────────────────────────┐  │
│  │ Actor (LoRA) │  │ vLLM/SGLang Rollout Engine            │  │
│  │ Qwen2.5-7B   │  │ KV Cache + Attention                  │  │
│  └──────────────┘  └────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ GPU 3: Student Reward Model (独立)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Student Model (Qwen2.5-3B) + FSF LoRA Adapter            │  │
│  │  - 认知奖励: 使用基座                                      │  │
│  │  - FSF 奖励: 使用基座 + LoRA                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

使用方式：
1. 配置 reward_model.enable_resource_pool = true
2. 配置 reward_model.n_gpus_per_node = 1
3. 使用 RayStudentRewardManager 管理 Ray Actor
"""

import ray
from typing import Any, Optional, Dict, List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import re
import os

from verl import DataProto


@ray.remote(num_gpus=1)
class RayStudentRewardActor:
    """
    Ray Actor: Student Reward Model

    部署在独立的 GPU 上，通过 Ray RPC 调用
    """

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
        dtype: str = "bfloat16",
        max_length: int = 4096,
    ):
        """初始化 Ray Actor"""
        self.student_model_path = student_model_path
        self.fsf_lora_path = fsf_lora_path
        self.max_length = max_length

        # 确定设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, dtype)

        print(f"[RayStudentRewardActor] 初始化在设备: {self.device}")
        print(f"[RayStudentRewardActor] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

        self._load_model()

    def _load_model(self):
        """加载模型"""
        print(f"[RayStudentRewardActor] 加载 Student Model: {self.student_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.base_model.to(self.device)
        self.base_model.eval()

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.fsf_model = None
        if self.fsf_lora_path:
            print(f"[RayStudentRewardActor] 加载 FSF LoRA: {self.fsf_lora_path}")
            self.fsf_model = PeftModel.from_pretrained(
                self.base_model,
                self.fsf_lora_path,
                is_trainable=False,
            )
            self.fsf_model.eval()

        print(f"[RayStudentRewardActor] 加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def compute_cognitive_reward(
        self,
        questions: List[str],
        rewrite_cots: List[str],
        answers: List[str],
    ) -> List[float]:
        """计算认知奖励"""
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

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=full_enc.input_ids,
                attention_mask=full_enc.attention_mask,
            )
            logits = outputs.logits

        log_probs = []

        for i in range(batch_size):
            start_idx = prefix_lengths[i].item()
            end_idx = full_enc.attention_mask[i].sum().item()

            if start_idx >= end_idx:
                log_probs.append(-100.0)
                continue

            ans_logits = logits[i, start_idx-1:end_idx-1, :]
            ans_labels = full_enc.input_ids[i, start_idx:end_idx]

            ans_log_probs = F.log_softmax(ans_logits, dim=-1)
            token_log_probs = ans_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
            log_probs.append(token_log_probs.sum().item())

        return log_probs

    def _parse_fsf(self, text: str) -> float:
        """解析 FSF"""
        try:
            match = re.search(r'\{[^{}]*"fsf"\s*:\s*([\d.]+)[^{}]*\}', text)
            if match:
                return float(match.group(1))
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return min(float(match.group(1)), 1.0)
        except:
            pass
        return 0.5

    def compute_fsf_batch(self, reasonings: List[str]) -> List[float]:
        """批量计算 FSF"""
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

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> Dict[str, Any]:
        """
        计算组合奖励

        Returns:
            {
                "cognitive_rewards": List[float],
                "logic_rewards": List[float],
                "total_rewards": List[float],
            }
        """
        # 认知奖励
        r_cog = self.compute_cognitive_reward(questions, rewrite_cots, answers)

        # 逻辑奖励
        if use_logic_reward and self.fsf_model is not None and original_cots and len(original_cots[0]) > 0:
            original_fsfs = self.compute_fsf_batch(original_cots)
            rewrite_fsfs = self.compute_fsf_batch(rewrite_cots)
            r_logic = [1.0 if r < o else 0.0 for o, r in zip(original_fsfs, rewrite_fsfs)]
        else:
            r_logic = [0.0] * len(questions)

        # 组合
        total = [c + lambda_logic * l for c, l in zip(r_cog, r_logic)]

        return {
            "cognitive_rewards": r_cog,
            "logic_rewards": r_logic,
            "total_rewards": total,
        }

    def health_check(self) -> bool:
        """健康检查"""
        return True


class RayStudentRewardManager:
    """
    管理 Ray Student Reward Actor

    支持：
    1. 懒加载 Ray Actor
    2. 批量计算
    3. 异步调用
    """

    def __init__(
        self,
        student_model_path: str,
        fsf_lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        max_length: int = 4096,
        num_actors: int = 1,
    ):
        self.student_model_path = student_model_path
        self.fsf_lora_path = fsf_lora_path
        self.dtype = dtype
        self.max_length = max_length
        self.num_actors = num_actors

        self.actors: List[ray.actor.ActorHandle] = []
        self._initialized = False

    def init_actors(self):
        """初始化 Ray Actors"""
        if self._initialized:
            return

        print(f"[RayStudentRewardManager] 创建 {self.num_actors} 个 Ray Actor...")

        for i in range(self.num_actors):
            actor = RayStudentRewardActor.remote(
                student_model_path=self.student_model_path,
                fsf_lora_path=self.fsf_lora_path,
                dtype=self.dtype,
                max_length=self.max_length,
            )
            self.actors.append(actor)

        # 等待初始化完成
        ray.get([actor.health_check.remote() for actor in self.actors])

        self._initialized = True
        print(f"[RayStudentRewardManager] {self.num_actors} 个 Actor 初始化完成")

    def compute_reward(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> Dict[str, Any]:
        """同步计算奖励"""
        self.init_actors()

        # 使用第一个 actor（可扩展为负载均衡）
        actor = self.actors[0]

        result = ray.get(actor.compute_reward.remote(
            questions=questions,
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            answers=answers,
            lambda_logic=lambda_logic,
            use_logic_reward=use_logic_reward,
        ))

        return result

    def compute_reward_async(
        self,
        questions: List[str],
        original_cots: List[str],
        rewrite_cots: List[str],
        answers: List[str],
        lambda_logic: float = 0.5,
        use_logic_reward: bool = True,
    ) -> ray.ObjectRef:
        """异步计算奖励（返回 Future）"""
        self.init_actors()

        actor = self.actors[0]

        return actor.compute_reward.remote(
            questions=questions,
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            answers=answers,
            lambda_logic=lambda_logic,
            use_logic_reward=use_logic_reward,
        )

    def shutdown(self):
        """关闭所有 Actors"""
        for actor in self.actors:
            ray.kill(actor)
        self.actors = []
        self._initialized = False


# ==================== 全局单例 ====================

_ray_reward_manager: Optional[RayStudentRewardManager] = None


def get_ray_reward_manager(
    student_model_path: str,
    fsf_lora_path: Optional[str] = None,
    **kwargs
) -> RayStudentRewardManager:
    """获取 RayStudentRewardManager 单例"""
    global _ray_reward_manager
    if _ray_reward_manager is None:
        _ray_reward_manager = RayStudentRewardManager(
            student_model_path,
            fsf_lora_path=fsf_lora_path,
            **kwargs
        )
    return _ray_reward_manager


# ==================== VERL 接口 ====================

def compute_ray_student_reward(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    fsf_lora_path: Optional[str] = None,
    lambda_logic: float = 0.5,
    use_logic_reward: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    VERL 奖励函数接口（使用 Ray Actor）

    Student Model 部署在独立的 GPU 上，不与 Rollout 引擎竞争显存
    """
    batch_size = data.batch.batch_size[0]

    # 解码
    prompts = tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
    rewrite_cots = tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    # 提取信息
    answers, original_cots, questions = [], [], []

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

    # 获取 Manager
    manager = get_ray_reward_manager(student_model_path, fsf_lora_path)

    # 计算奖励
    result = manager.compute_reward(
        questions=questions,
        original_cots=original_cots,
        rewrite_cots=rewrite_cots,
        answers=answers,
        lambda_logic=lambda_logic,
        use_logic_reward=use_logic_reward,
    )

    # 转为 Tensor
    total_rewards = torch.tensor(
        result["total_rewards"],
        device="cuda",
        dtype=torch.bfloat16
    )
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
        "reward_extra_info": result,
    }
