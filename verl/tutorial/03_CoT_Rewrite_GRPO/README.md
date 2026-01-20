# CoT Rewriter GRPO 训练实战

> 本章节详细讲解如何实现一个基于 GRPO 算法的 CoT Rewriter 训练系统，包括复杂的组合奖励函数设计和显存优化策略。

---

## 快速导航

| 文档 | 说明 |
|------|------|
| **[最优配置方案](docs/optimal_config_guide.md)** | 7B 全参 + 1.7B 统一奖励模型，4×A100-80G 详细配置 |
| [configs/optimal_7b_fullparam_grpo.yaml](configs/optimal_7b_fullparam_grpo.yaml) | 最优配置文件 |
| [scripts/run_optimal_training.sh](scripts/run_optimal_training.sh) | 一键启动脚本 |

---

## 1. 项目目标

### 1.1 任务定义

训练一个 **CoT Rewriter (改写模型)**，将原始的 Chain-of-Thought 推理过程改写得更清晰、更有逻辑性，从而提升下游 Student Model 的预测能力。

```
输入:  原始 CoT (可能冗长、有歧路)
输出:  改写后的 CoT (精简、逻辑清晰)
目标:  最大化 Student Model 对正确答案的预测概率
```

### 1.2 奖励函数设计

组合奖励: **R = R_cog + λ · R_logic**

| 奖励类型 | 公式 | 说明 |
|---------|------|------|
| 认知奖励 R_cog | log P_student(Answer \| Question, Rewrite_CoT) | 让 Student 更容易预测正确答案 |
| 逻辑奖励 R_logic | 1 if FSF(Rewrite) < FSF(Original) else 0 | 减少推理失败分支 |

### 1.3 模型角色

| 模型 | 用途 | 训练状态 | 显存位置 |
|------|------|---------|---------|
| Actor (Qwen-2.5-7B-Instruct) | 改写 CoT | 可训练 (LoRA) | GPU 0-3 |
| Student (Qwen-2.5-3B-Instruct) | 计算 R_cog | 冻结 | GPU 0-3 (共享) |
| LLM Judge (API/Local) | 计算 FSF | 冻结/外部 | CPU/API |

## 2. 项目结构

```
verl_grpo_practice/
├── configs/
│   ├── base_config.yaml           # 基础配置
│   └── cot_rewriter_grpo.yaml     # CoT Rewriter 专用配置
├── src/
│   ├── __init__.py
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── cognitive_reward.py    # 认知奖励计算
│   │   ├── logic_reward.py        # 逻辑奖励计算
│   │   └── combined_reward.py     # 组合奖励函数
│   ├── data/
│   │   ├── __init__.py
│   │   └── cot_dataset.py         # 数据集定义
│   └── utils/
│       ├── __init__.py
│       └── fsf_analyzer.py        # FSF 分析工具
├── scripts/
│   ├── prepare_data.py            # 数据预处理
│   ├── train.py                   # 训练入口
│   └── evaluate.py                # 评估脚本
└── pyproject.toml
```

## 3. 核心代码实现

### 3.1 配置文件

```yaml
# configs/cot_rewriter_grpo.yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer  # 继承 Verl 默认 PPO 配置
  - _self_

# ============ 实验信息 ============
trainer:
  project_name: cot_rewriter_grpo
  experiment_name: qwen7b_lora_4xA100
  total_epochs: 3
  save_freq: 100
  val_freq: 50

# ============ 数据配置 ============
data:
  train_files: data/processed/train.parquet
  val_files: data/processed/val.parquet
  train_batch_size: 64        # 每个 step 的 prompt 数量
  max_prompt_length: 1024
  max_response_length: 2048
  return_raw_chat: true       # 返回原始对话格式

# ============ Actor/Rollout 配置 ============
actor_rollout_ref:
  hybrid_engine: true         # 使用混合引擎

  model:
    path: ./models/Qwen2.5-7B-Instruct
    trust_remote_code: true
    # LoRA 配置 (节省显存，且 Ref Policy 无需额外显存)
    lora:
      rank: 64
      alpha: 128
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      dropout: 0.05

  actor:
    optim:
      lr: 1e-5
      weight_decay: 0.01
      lr_scheduler: cosine
      warmup_ratio: 0.03
    ppo_epochs: 1             # GRPO 通常只需要 1 个 epoch
    clip_ratio: 0.2

  rollout:
    name: sglang              # 使用 SGLang 推理
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.35
    n: 8                      # 每个 prompt 生成 8 个响应 (GRPO 需要)
    temperature: 0.7
    top_p: 0.95
    max_new_tokens: 2048

  ref:
    # 使用 LoRA 时，Ref Policy 复用 Actor 基座，无需额外配置
    enable: false

# ============ 算法配置 ============
algorithm:
  adv_estimator: grpo         # 使用 GRPO
  norm_adv_by_std_in_grpo: true
  gamma: 1.0
  use_kl_in_reward: false     # GRPO 不使用 KL 惩罚

# ============ 奖励配置 ============
reward_model:
  use_reward_loop: false
  custom_reward_function:
    path: src/reward/combined_reward.py
    name: compute_cot_rewriter_reward
    reward_kwargs:
      student_model_path: ./models/Qwen2.5-3B-Instruct
      lambda_logic: 0.5       # R_logic 权重
      use_fsf_reward: true
      fsf_judge_type: "api"   # "api" 或 "local"
      fsf_api_url: "http://localhost:8000/v1/chat/completions"

# ============ 禁用 Critic (GRPO 不需要) ============
critic:
  enable: false
```

### 3.2 认知奖励实现

```python
# src/reward/cognitive_reward.py
"""
认知奖励 (Cognitive Reward) 计算模块

R_cog = log P_student(Answer | Question, Rewrite_CoT)

核心思想: 改写后的 CoT 应该让 Student Model 更容易预测正确答案
"""

from typing import Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CognitiveRewardComputer:
    """
    认知奖励计算器

    使用冻结的 Student Model 计算改写后 CoT 对答案预测的增益
    """

    def __init__(
        self,
        student_model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 4096,
    ):
        """
        初始化 Student Model

        Args:
            student_model_path: Student Model 路径
            device: 设备 (默认 CUDA)
            dtype: 数据类型 (默认 bfloat16)
            max_length: 最大序列长度
        """
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        # 加载 Student Model (冻结状态)
        print(f"[CognitiveReward] 加载 Student Model: {student_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            student_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            # 注意: 不使用 device_map="auto"，手动控制设备
        )
        self.model.to(device)
        self.model.eval()

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[CognitiveReward] Student Model 加载完成, 显存占用: {self._get_memory_usage():.2f} GB")

    def _get_memory_usage(self) -> float:
        """获取当前 GPU 显存占用 (GB)"""
        return torch.cuda.memory_allocated() / 1e9

    def _build_student_input(self, question: str, rewrite_cot: str, answer: str) -> str:
        """
        构造 Student Model 输入

        格式:
        Question: {question}
        Reasoning: {rewrite_cot}
        Answer: {answer}

        我们计算 P(Answer | Question, Reasoning)
        """
        return f"""Question: {question}

Reasoning: {rewrite_cot}

Answer: {answer}"""

    def compute_answer_log_prob(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算 log P_student(Answer | Question, Rewrite_CoT)

        Args:
            questions: 问题列表
            rewrite_cots: 改写后的 CoT 列表
            answers: 正确答案列表

        Returns:
            log_probs: shape (batch_size,), 每个样本的 log 概率
        """
        batch_size = len(questions)

        # 构造输入
        full_texts = [
            self._build_student_input(q, cot, ans)
            for q, cot, ans in zip(questions, rewrite_cots, answers)
        ]

        # 分别构造 prefix (Question + Reasoning) 和 target (Answer)
        prefixes = [
            f"Question: {q}\n\nReasoning: {cot}\n\nAnswer: "
            for q, cot in zip(questions, rewrite_cots)
        ]

        # Tokenize
        prefix_encodings = self.tokenizer(
            prefixes,
            padding=True,
            truncation=True,
            max_length=self.max_length - 100,  # 留出空间给 answer
            return_tensors="pt"
        ).to(self.device)

        full_encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # 计算 prefix 长度 (不含 padding)
        prefix_lengths = prefix_encodings.attention_mask.sum(dim=1)

        with torch.no_grad():
            # 前向传播
            outputs = self.model(
                input_ids=full_encodings.input_ids,
                attention_mask=full_encodings.attention_mask,
            )
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # 计算 Answer 部分的 log 概率
        log_probs = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            # Answer 开始位置 (prefix 之后)
            start_idx = prefix_lengths[i].item()
            # Answer 结束位置 (非 padding)
            end_idx = full_encodings.attention_mask[i].sum().item()

            if start_idx >= end_idx:
                # 没有 answer token
                log_probs[i] = -100.0
                continue

            # 提取 answer 部分的 logits 和 labels
            # logits[t] 预测 token[t+1]，所以用 [start_idx-1:end_idx-1]
            answer_logits = logits[i, start_idx-1:end_idx-1, :]  # (answer_len, vocab_size)
            answer_labels = full_encodings.input_ids[i, start_idx:end_idx]  # (answer_len,)

            # 计算 cross-entropy loss (取负得到 log prob)
            answer_log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = answer_log_probs.gather(1, answer_labels.unsqueeze(1)).squeeze(1)

            # 求和得到序列 log 概率
            log_probs[i] = token_log_probs.sum()

        return log_probs

    def __call__(
        self,
        questions: list[str],
        rewrite_cots: list[str],
        answers: list[str],
    ) -> torch.Tensor:
        """
        计算认知奖励

        Args:
            questions: 问题列表
            rewrite_cots: 改写后的 CoT 列表
            answers: 正确答案列表

        Returns:
            rewards: shape (batch_size,), 归一化后的奖励值
        """
        log_probs = self.compute_answer_log_prob(questions, rewrite_cots, answers)

        # 可选: 对 log prob 进行归一化或缩放
        # 这里简单地使用 log prob 作为奖励
        rewards = log_probs

        return rewards


# 全局单例 (避免重复加载模型)
_cognitive_reward_computer: Optional[CognitiveRewardComputer] = None


def get_cognitive_reward_computer(student_model_path: str, **kwargs) -> CognitiveRewardComputer:
    """获取或创建 CognitiveRewardComputer 单例"""
    global _cognitive_reward_computer
    if _cognitive_reward_computer is None:
        _cognitive_reward_computer = CognitiveRewardComputer(student_model_path, **kwargs)
    return _cognitive_reward_computer
```

### 3.3 逻辑奖励实现

```python
# src/reward/logic_reward.py
"""
逻辑奖励 (Logic Reward) 计算模块

R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0

FSF (Failed Step Fraction): 推理过程中失败分支的比例
"""

import asyncio
import json
import re
from typing import Optional
import httpx


class FSFAnalyzer:
    """
    FSF (Failed Step Fraction) 分析器

    使用 LLM-as-a-Judge 分析推理过程中的失败分支占比
    """

    # FSF 分析 prompt
    ANALYSIS_PROMPT = """你是一个推理过程分析专家。请分析以下推理过程中的"失败分支"占比。

**失败分支定义**: 在推理过程中，尝试了某种方法但最终发现不可行而放弃的分支。

**请按照以下步骤分析**:
1. 识别推理过程中的所有步骤
2. 标记哪些步骤是"失败分支"(尝试后放弃)
3. 计算 FSF = 失败分支数 / 总步骤数

**推理过程**:
{reasoning}

**输出格式** (JSON):
```json
{{
  "total_steps": <总步骤数>,
  "failed_steps": <失败分支数>,
  "fsf": <失败分支占比, 0.0-1.0>,
  "failed_step_descriptions": ["<失败分支1描述>", "<失败分支2描述>", ...]
}}
```

请只输出 JSON，不要有其他文字。"""

    def __init__(
        self,
        judge_type: str = "api",
        api_url: str = "http://localhost:8000/v1/chat/completions",
        api_key: str = "EMPTY",
        model_name: str = "gpt-4",
        timeout: float = 30.0,
        max_concurrent: int = 16,
    ):
        """
        初始化 FSF 分析器

        Args:
            judge_type: "api" (外部 API) 或 "local" (本地模型)
            api_url: API 地址
            api_key: API 密钥
            model_name: 模型名称
            timeout: 超时时间 (秒)
            max_concurrent: 最大并发数
        """
        self.judge_type = judge_type
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)

    def _parse_fsf_response(self, response_text: str) -> float:
        """解析 FSF 响应"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return float(data.get("fsf", 0.5))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # 解析失败，返回默认值
        return 0.5

    async def _analyze_single_async(self, reasoning: str) -> float:
        """异步分析单个推理过程的 FSF"""
        async with self.semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model_name,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": self.ANALYSIS_PROMPT.format(reasoning=reasoning)
                                }
                            ],
                            "temperature": 0.0,
                            "max_tokens": 512,
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    return self._parse_fsf_response(response_text)
            except Exception as e:
                print(f"[FSFAnalyzer] API 调用失败: {e}")
                return 0.5  # 返回中间值

    def analyze_single(self, reasoning: str) -> float:
        """同步分析单个推理过程的 FSF"""
        return asyncio.run(self._analyze_single_async(reasoning))

    async def analyze_batch_async(self, reasonings: list[str]) -> list[float]:
        """异步批量分析 FSF"""
        tasks = [self._analyze_single_async(r) for r in reasonings]
        return await asyncio.gather(*tasks)

    def analyze_batch(self, reasonings: list[str]) -> list[float]:
        """同步批量分析 FSF"""
        return asyncio.run(self.analyze_batch_async(reasonings))


def compute_logic_reward(
    original_cots: list[str],
    rewrite_cots: list[str],
    fsf_analyzer: FSFAnalyzer,
) -> list[float]:
    """
    计算逻辑奖励

    R_logic = 1 if FSF(Rewrite) < FSF(Original) else 0

    Args:
        original_cots: 原始 CoT 列表
        rewrite_cots: 改写后 CoT 列表
        fsf_analyzer: FSF 分析器

    Returns:
        rewards: 逻辑奖励列表 (0 或 1)
    """
    # 批量分析 FSF
    original_fsfs = fsf_analyzer.analyze_batch(original_cots)
    rewrite_fsfs = fsf_analyzer.analyze_batch(rewrite_cots)

    # 计算奖励
    rewards = []
    for orig_fsf, rewrite_fsf in zip(original_fsfs, rewrite_fsfs):
        if rewrite_fsf < orig_fsf:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# 全局单例
_fsf_analyzer: Optional[FSFAnalyzer] = None


def get_fsf_analyzer(**kwargs) -> FSFAnalyzer:
    """获取或创建 FSFAnalyzer 单例"""
    global _fsf_analyzer
    if _fsf_analyzer is None:
        _fsf_analyzer = FSFAnalyzer(**kwargs)
    return _fsf_analyzer
```

### 3.4 组合奖励函数

```python
# src/reward/combined_reward.py
"""
组合奖励函数

R = R_cog + λ · R_logic

这个文件定义了 Verl 需要的奖励函数接口
"""

from typing import Any
import torch
import numpy as np

from verl import DataProto
from .cognitive_reward import get_cognitive_reward_computer
from .logic_reward import get_fsf_analyzer, compute_logic_reward


def compute_cot_rewriter_reward(
    data: DataProto,
    tokenizer: Any,
    # 以下参数从 config.reward_model.custom_reward_function.reward_kwargs 传入
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    lambda_logic: float = 0.5,
    use_fsf_reward: bool = True,
    fsf_judge_type: str = "api",
    fsf_api_url: str = "http://localhost:8000/v1/chat/completions",
    **kwargs,
) -> dict[str, Any]:
    """
    CoT Rewriter 组合奖励函数

    这是 Verl 调用的入口函数，签名需要符合 Verl 的 RewardFn 接口:
    - 输入: DataProto (包含 batch 和 non_tensor_batch)
    - 输出: {"reward_tensor": Tensor, "reward_extra_info": dict}

    Args:
        data: Verl DataProto，包含:
            - batch["prompts"]: Prompt token IDs
            - batch["responses"]: 生成的响应 token IDs
            - non_tensor_batch["reward_model"]["ground_truth"]: 正确答案
            - non_tensor_batch["original_cot"]: 原始 CoT
        tokenizer: 分词器
        student_model_path: Student Model 路径
        lambda_logic: 逻辑奖励权重
        use_fsf_reward: 是否使用 FSF 奖励
        fsf_judge_type: FSF Judge 类型 ("api" 或 "local")
        fsf_api_url: FSF API 地址

    Returns:
        {
            "reward_tensor": shape (batch_size, response_length),
            "reward_extra_info": {
                "cognitive_rewards": list,
                "logic_rewards": list,
                "fsf_original": list,
                "fsf_rewrite": list,
            }
        }
    """
    batch_size = data.batch.batch_size[0]

    # ============ 1. 解码文本 ============
    # 解码 prompts (Question)
    prompts = tokenizer.batch_decode(
        data.batch["prompts"],
        skip_special_tokens=True
    )

    # 解码 responses (Rewrite CoT)
    rewrite_cots = tokenizer.batch_decode(
        data.batch["responses"],
        skip_special_tokens=True
    )

    # 从 non_tensor_batch 获取额外信息
    answers = []
    original_cots = []
    questions = []

    for i in range(batch_size):
        # 获取正确答案
        reward_info = data.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_info, list):
            reward_info = reward_info[i] if i < len(reward_info) else {}
        answer = reward_info.get("ground_truth", "")
        answers.append(answer)

        # 获取原始 CoT
        orig_cot = data.non_tensor_batch.get("original_cot", [""])[i] \
            if "original_cot" in data.non_tensor_batch else ""
        original_cots.append(orig_cot)

        # 从 prompt 中提取 question (简化处理)
        questions.append(prompts[i])

    # ============ 2. 计算认知奖励 R_cog ============
    cognitive_computer = get_cognitive_reward_computer(student_model_path)
    cognitive_rewards = cognitive_computer(
        questions=questions,
        rewrite_cots=rewrite_cots,
        answers=answers,
    )  # shape: (batch_size,)

    # ============ 3. 计算逻辑奖励 R_logic (可选) ============
    if use_fsf_reward and len(original_cots[0]) > 0:
        fsf_analyzer = get_fsf_analyzer(
            judge_type=fsf_judge_type,
            api_url=fsf_api_url,
        )
        logic_rewards = compute_logic_reward(
            original_cots=original_cots,
            rewrite_cots=rewrite_cots,
            fsf_analyzer=fsf_analyzer,
        )
        logic_rewards = torch.tensor(logic_rewards, device=cognitive_rewards.device)
    else:
        logic_rewards = torch.zeros(batch_size, device=cognitive_rewards.device)

    # ============ 4. 组合奖励 ============
    # R = R_cog + λ · R_logic
    total_rewards = cognitive_rewards + lambda_logic * logic_rewards

    # ============ 5. 转换为 token-level 奖励 ============
    # GRPO 使用 outcome reward，将标量奖励放在最后一个 token
    response_length = data.batch["responses"].shape[1]
    token_level_rewards = torch.zeros(
        batch_size, response_length,
        device=cognitive_rewards.device,
        dtype=cognitive_rewards.dtype
    )

    # 获取每个样本的实际响应长度
    response_mask = data.batch.get("response_mask", None)
    if response_mask is not None:
        # 找到每个样本最后一个有效 token 的位置
        seq_lengths = response_mask.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0, max=response_length - 1)
    else:
        seq_lengths = torch.full((batch_size,), response_length - 1, dtype=torch.long)

    # 将奖励放在最后一个有效 token 位置
    for i in range(batch_size):
        token_level_rewards[i, seq_lengths[i]] = total_rewards[i]

    # ============ 6. 返回结果 ============
    return {
        "reward_tensor": token_level_rewards,
        "reward_extra_info": {
            "cognitive_rewards": cognitive_rewards.cpu().tolist(),
            "logic_rewards": logic_rewards.cpu().tolist(),
            "total_rewards": total_rewards.cpu().tolist(),
        }
    }


# ============ 简化版: 只使用认知奖励 ============
def compute_cognitive_reward_only(
    data: DataProto,
    tokenizer: Any,
    student_model_path: str = "./models/Qwen2.5-3B-Instruct",
    **kwargs,
) -> dict[str, Any]:
    """
    只使用认知奖励的简化版本

    适用于不需要 FSF 分析的场景
    """
    return compute_cot_rewriter_reward(
        data=data,
        tokenizer=tokenizer,
        student_model_path=student_model_path,
        use_fsf_reward=False,
        lambda_logic=0.0,
        **kwargs,
    )
```

### 3.5 训练脚本

```python
# scripts/train.py
"""
CoT Rewriter GRPO 训练入口
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import ray

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@hydra.main(config_path="../configs", config_name="cot_rewriter_grpo", version_base=None)
def main(config: DictConfig):
    """
    训练主函数

    使用 Hydra 加载配置，启动 Verl 训练
    """
    print("=" * 60)
    print("CoT Rewriter GRPO Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(
            # 根据实际情况配置
            num_cpus=32,
            num_gpus=4,
            object_store_memory=50 * 1024**3,  # 50GB
        )

    # 导入 Verl 训练入口
    from verl.trainer.main_ppo import main as verl_main

    # 启动训练
    verl_main(config)


if __name__ == "__main__":
    main()
```

### 3.6 启动命令

```bash
# 单机 4 卡训练
python scripts/train.py \
    trainer.total_epochs=3 \
    data.train_batch_size=64 \
    actor_rollout_ref.rollout.n=8

# 使用 torchrun (多进程)
torchrun --nproc_per_node=4 scripts/train.py

# 使用 Ray 分布式 (推荐)
RAY_DEDUP_LOGS=1 python scripts/train.py
```

## 4. 显存优化策略

### 4.1 显存分析

单机 4 × A100 (80G) 显存分配:

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| Actor (Qwen-7B, LoRA) | ~16 GB × 4 | FSDP 分片 |
| Rollout Engine (SGLang) | ~14 GB × 4 | 共享 Actor 权重 |
| Student Model (Qwen-3B) | ~6 GB × 1 | 仅在 GPU 0 |
| 激活值 + 梯度 | ~20 GB × 4 | 取决于 batch size |
| **总计** | ~56 GB × 4 | 有 ~24 GB 余量 |

### 4.2 关键优化配置

```yaml
# 1. 使用 LoRA 减少可训练参数
actor_rollout_ref:
  model:
    lora:
      rank: 64        # 较小的 rank
      alpha: 128

# 2. 控制 Rollout 显存
  rollout:
    gpu_memory_utilization: 0.35  # 留出空间给其他组件
    enforce_eager: true           # 如果 CUDA Graphs 有问题

# 3. 梯度检查点 (用 时间换空间)
  actor:
    gradient_checkpointing: true

# 4. 混合精度
  model:
    dtype: bfloat16

# 5. 序列长度平衡
data:
  use_seqlen_balancing: true
```

### 4.3 Student Model 显存优化

```python
# 延迟加载 Student Model (只在需要时加载)
class LazyStudentModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},  # 固定在 GPU 0
            )
            self._model.eval()

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    def offload(self):
        """卸载模型到 CPU"""
        if self._model is not None:
            self._model.to("cpu")
            torch.cuda.empty_cache()

    def reload(self):
        """重新加载到 GPU"""
        if self._model is not None:
            self._model.to("cuda:0")
```

## 5. 训练监控

### 5.1 关键指标

```python
# Verl 自动记录的指标
metrics = {
    # 奖励相关
    "reward/mean": "平均奖励",
    "reward/std": "奖励标准差",
    "reward/cognitive_mean": "认知奖励均值",
    "reward/logic_mean": "逻辑奖励均值",

    # GRPO 相关
    "grpo/advantage_mean": "优势函数均值",
    "grpo/advantage_std": "优势函数标准差",

    # Actor 训练
    "actor/loss": "策略损失",
    "actor/entropy": "策略熵",
    "actor/clip_ratio": "PPO 裁剪比例",

    # 系统指标
    "system/gpu_memory": "GPU 显存使用",
    "system/throughput": "吞吐量 (tokens/s)",
}
```

### 5.2 WandB 集成

```yaml
# config.yaml
trainer:
  logger:
    - wandb

wandb:
  project: cot_rewriter_grpo
  entity: your_entity
  tags: ["grpo", "cot-rewrite", "qwen-7b"]
```

### 5.3 自定义回调

```python
# src/callbacks/logging_callback.py
from verl.trainer.callbacks import Callback


class CoTRewriterCallback(Callback):
    """自定义训练回调"""

    def on_step_end(self, trainer, batch, metrics):
        """每个 step 结束时调用"""
        # 记录自定义指标
        if "cognitive_rewards" in metrics.get("reward_extra_info", {}):
            cog_rewards = metrics["reward_extra_info"]["cognitive_rewards"]
            trainer.log({
                "custom/cognitive_reward_max": max(cog_rewards),
                "custom/cognitive_reward_min": min(cog_rewards),
            })

    def on_validation_end(self, trainer, val_metrics):
        """验证结束时调用"""
        # 打印改写样例
        if "samples" in val_metrics:
            print("\n===== CoT Rewrite Samples =====")
            for sample in val_metrics["samples"][:3]:
                print(f"Original: {sample['original'][:100]}...")
                print(f"Rewrite:  {sample['rewrite'][:100]}...")
                print(f"Reward:   {sample['reward']:.3f}")
                print("-" * 40)
```

## 6. 评估与分析

### 6.1 评估脚本

```python
# scripts/evaluate.py
"""
评估 CoT Rewriter 性能
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.reward.cognitive_reward import CognitiveRewardComputer


def evaluate_rewriter(
    model_path: str,
    test_file: str,
    student_model_path: str,
    output_file: str,
):
    """
    评估改写模型性能

    指标:
    1. 认知增益: avg(R_cog(rewrite) - R_cog(original))
    2. 改写率: 改写后文本与原始文本的差异度
    3. 长度变化: avg(len(rewrite) / len(original))
    """
    # 加载改写模型
    print(f"Loading rewriter model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # 加载评估器
    cognitive_computer = CognitiveRewardComputer(student_model_path)

    # 加载测试数据
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f]

    results = []
    cognitive_gains = []

    for sample in tqdm(test_data, desc="Evaluating"):
        # 生成改写
        prompt = f"请将以下推理过程改写得更清晰:\n\n{sample['original_cot']}\n\n改写后:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )

        rewrite_cot = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 计算认知奖励
        original_reward = cognitive_computer(
            questions=[sample["question"]],
            rewrite_cots=[sample["original_cot"]],
            answers=[sample["answer"]],
        )[0].item()

        rewrite_reward = cognitive_computer(
            questions=[sample["question"]],
            rewrite_cots=[rewrite_cot],
            answers=[sample["answer"]],
        )[0].item()

        cognitive_gain = rewrite_reward - original_reward
        cognitive_gains.append(cognitive_gain)

        results.append({
            "question": sample["question"],
            "original_cot": sample["original_cot"],
            "rewrite_cot": rewrite_cot,
            "answer": sample["answer"],
            "original_reward": original_reward,
            "rewrite_reward": rewrite_reward,
            "cognitive_gain": cognitive_gain,
        })

    # 保存结果
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印统计
    print("\n===== Evaluation Results =====")
    print(f"Samples: {len(results)}")
    print(f"Avg Cognitive Gain: {sum(cognitive_gains) / len(cognitive_gains):.4f}")
    print(f"Positive Gain Rate: {sum(1 for g in cognitive_gains if g > 0) / len(cognitive_gains):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--student_model_path", default="./models/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_file", default="eval_results.jsonl")
    args = parser.parse_args()

    evaluate_rewriter(
        model_path=args.model_path,
        test_file=args.test_file,
        student_model_path=args.student_model_path,
        output_file=args.output_file,
    )
```

## 7. 常见问题与调试

### 7.1 奖励为 NaN 或 Inf

```python
# 在 combined_reward.py 中添加检查
def compute_cot_rewriter_reward(...):
    ...
    # 检查 NaN
    if torch.isnan(cognitive_rewards).any():
        print("[WARN] Cognitive rewards contain NaN!")
        cognitive_rewards = torch.nan_to_num(cognitive_rewards, nan=0.0)

    # 裁剪极端值
    cognitive_rewards = cognitive_rewards.clamp(-100, 100)
    ...
```

### 7.2 Rollout 生成质量差

```yaml
# 调整生成参数
actor_rollout_ref:
  rollout:
    temperature: 0.8        # 稍高的温度增加多样性
    top_p: 0.95
    repetition_penalty: 1.1 # 减少重复
    n: 16                   # 增加每个 prompt 的采样数
```

### 7.3 训练不稳定

```yaml
# 减小学习率
actor_rollout_ref:
  actor:
    optim:
      lr: 5e-6              # 更小的学习率
      warmup_ratio: 0.1     # 更长的 warmup

# 使用梯度裁剪
    grad_clip: 1.0
```

### 7.4 显存 OOM

```yaml
# 减小 batch size
data:
  train_batch_size: 32      # 减半

# 减小 rollout.n
actor_rollout_ref:
  rollout:
    n: 4                    # 减少采样数

# 使用梯度累积
  actor:
    gradient_accumulation_steps: 2
```

## 8. 小结

本章完整实现了 CoT Rewriter 的 GRPO 训练系统：

1. **组合奖励函数**: R = R_cog + λ · R_logic
2. **认知奖励**: 使用冻结的 Student Model 计算答案预测概率
3. **逻辑奖励**: 使用 LLM-as-a-Judge 分析 FSF
4. **显存优化**: LoRA、延迟加载、显存控制
5. **训练监控**: 自定义指标、WandB 集成
6. **评估工具**: 认知增益评估

关键要点：
- GRPO 不需要 Critic，适合 Outcome Reward 场景
- LoRA 显著减少显存，且 Reference Policy 零成本
- Student Model 显存需要单独管理
- FSF 分析可以异步化，避免阻塞训练

祝你训练顺利!
