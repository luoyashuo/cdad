# VERL框架自定义Reward Model开发指南

> 参考来源：https://zhuanlan.zhihu.com/p/1931076626940139506

## 1. 概述

VERL框架支持自定义Reward Model来计算强化学习训练中的奖励信号。本文档介绍如何在VERL中加载自定义的Reward Model并计算Reward。

### 1.1 两种计算模式

| 模式 | 描述 | 特点 |
|------|------|------|
| **同步方式** | Batch Rollout + Batch Reward | 串行执行，先完成整个Batch采样，再计算Reward |
| **异步方式** | 单Prompt Rollout + 单Prompt Reward | 流水线并行，每条采样完立即计算Reward，效率更高 |

### 1.2 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        VERL Reward 计算流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  样本数据  │───▶│  RewardManager   │───▶│  Reward Function │  │
│  │(DataProto)│    │  (管理计算流程)    │    │   (具体打分逻辑)   │  │
│  └──────────┘    └──────────────────┘    └──────────────────┘  │
│        │                   │                       │           │
│        │                   │                       │           │
│        ▼                   ▼                       ▼           │
│   ground_truth        解码response            compute_score    │
│   extra_info          调用打分函数              返回分数         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 开发步骤

### 步骤1：实现自定义RewardManager

RewardManager负责管理Reward的计算流程，根据计算模式选择不同的基类。

#### 2.1.1 同步方式 - 继承 `AbstractRewardManager`

文件路径：`/verl/experimental/reward_loop/reward_manager/`

```python
from verl.experimental.reward_loop.reward_manager import AbstractRewardManager
from verl.protocol import DataProto
import torch

class MyRewardManager(AbstractRewardManager):

    def __init__(self, tokenizer, reward_fn, **kwargs):
        """
        初始化RewardManager

        Args:
            tokenizer: 分词器，用于解码模型输出
            reward_fn: 自定义的reward计算函数
        """
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

    def __call__(self, data: DataProto, **kwargs) -> torch.Tensor:
        """
        计算整个Batch的Reward

        Args:
            data: DataProto格式的batch数据

        Returns:
            torch.Tensor: 每个样本的reward分数
        """
        batch_size = len(data)
        rewards = []

        for i in range(batch_size):
            data_item = data[i]

            # 获取必要数据
            response_ids = data_item.batch["response_ids"]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # 解码模型输出
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # 计算reward
            score = self.reward_fn(
                solution_str=response_str,
                ground_truth=ground_truth
            )
            rewards.append(score)

        return torch.tensor(rewards)
```

#### 2.1.2 异步方式 - 继承 `RewardManagerBase`（推荐）

文件路径：`/verl/workers/reward_manager/` 或 `/verl/experimental/reward_loop/reward_manager/`

```python
from verl.experimental.reward_loop.reward_manager import RewardManagerBase
from verl.single_controller.base.decorator import register
from verl.protocol import DataProto

@register("my_reward_manager")  # 注册名称，用于配置文件引用
class MyWebSearchRewardManager(RewardManagerBase):
    """
    自定义异步RewardManager示例
    """

    def __init__(self, tokenizer, reward_fn, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.loop = asyncio.get_event_loop()

    async def run_single(self, data: DataProto) -> dict:
        """
        计算单条样本的Reward（异步方式）

        Args:
            data: 单条DataProto数据

        Returns:
            dict: 包含reward_score和reward_extra_info
        """
        # 1. 数据校验
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # 2. 提取必要字段
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # 3. 获取模型输出的response_ids
        response_ids = data_item.batch["response_ids"]
        # 过滤padding tokens
        valid_response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]

        # 4. 解码模型输出（在executor中运行避免阻塞）
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # 5. 调用自定义Reward计算方法
        score = await self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # 6. 构建返回结果
        reward_extra_info = {
            "acc": score,
            "data_source": data_source,
        }

        return {
            "reward_score": score,
            "reward_extra_info": reward_extra_info
        }

    async def compute_score(self, data_source, solution_str, ground_truth, extra_info):
        """
        调用reward function计算分数
        """
        return self.reward_fn(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info
        )
```

---

### 步骤2：实现自定义Reward计算逻辑

Reward Function是具体的打分逻辑实现，独立于RewardManager。

文件路径：`/verl/utils/reward_score/my_reward_function.py`

```python
"""
自定义Reward Function示例
"""
import re
import random


def extract_solution(solution_str: str) -> str:
    """
    从模型输出中提取答案

    Args:
        solution_str: 模型的完整输出

    Returns:
        str: 提取的答案，如果没找到返回None
    """
    # 示例：提取<answer>标签中的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(answer: str) -> str:
    """
    标准化答案，用于比较
    """
    if answer is None:
        return ""
    # 转小写、去除多余空格
    answer = answer.lower().strip()
    # 去除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer


def em_check(prediction: str, targets: list) -> bool:
    """
    Exact Match检查

    Args:
        prediction: 模型预测的答案
        targets: 标准答案列表

    Returns:
        bool: 是否匹配
    """
    pred_normalized = normalize_answer(prediction)
    for target in targets:
        if normalize_answer(target) == pred_normalized:
            return True
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
    method: str = "strict",
    format_score: float = 0.0,
    correct_score: float = 1.0
) -> float:
    """
    计算Reward分数的主函数

    Args:
        data_source: 数据来源标识
        solution_str: 模型输出的完整response
        ground_truth: 标准答案字典，包含target字段
        extra_info: 额外信息（可选）
        method: 评分方法 ("strict" | "soft")
        format_score: 格式正确但答案错误时的分数
        correct_score: 答案正确时的分数

    Returns:
        float: reward分数
    """
    # 调试日志（随机采样打印）
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"[Reward] data_source: {data_source}")
        print(f"[Reward] solution_str: {solution_str[:200]}...")

    # 1. 从模型输出中提取答案
    answer = extract_solution(solution_str)

    # 2. 检查是否成功提取答案
    if answer is None:
        if do_print:
            print("[Reward] Failed to extract answer, score: 0")
        return 0.0

    # 3. 获取标准答案
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    # 4. 执行Exact Match检查
    if em_check(answer, targets):
        if do_print:
            print(f"[Reward] Correct! answer: {answer}, score: {correct_score}")
        return correct_score
    else:
        if do_print:
            print(f"[Reward] Wrong! answer: {answer}, targets: {targets}, score: {format_score}")
        return format_score


# ============ 高级示例：基于LLM Judge的Reward ============

async def compute_score_with_llm_judge(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
    llm_client = None,
) -> float:
    """
    使用LLM作为Judge来计算Reward（适用于开放式问题）

    Args:
        llm_client: LLM API客户端

    Returns:
        float: reward分数 (0-1)
    """
    question = extra_info.get("question", "")
    answer = extract_solution(solution_str)
    reference = ground_truth.get("target", [""])[0]

    if answer is None:
        return 0.0

    # 构建Judge Prompt
    judge_prompt = f"""请评估以下回答的质量。

问题：{question}
参考答案：{reference}
模型回答：{answer}

请从以下维度评分（0-10分）：
1. 准确性：答案是否正确
2. 完整性：答案是否完整
3. 相关性：答案是否切题

请只输出一个0到1之间的总分，不要输出其他内容。
"""

    try:
        response = await llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))  # 确保在[0,1]范围内
    except Exception as e:
        print(f"[Reward] LLM Judge error: {e}")
        return 0.0
```

---

### 步骤3：配置启动脚本

在总控脚本中配置Reward相关参数。

文件路径：项目根目录下的 `run_train.sh`

```bash
#!/bin/bash

# 项目根目录
ROOT_DIR=$(pwd)

# Reward配置
REWARD_PATH="${ROOT_DIR}/verl/utils/reward_score/my_reward_function.py"

# 启动训练
python3 -m verl.trainer.main_ppo \
    # ========== 基础配置 ==========
    algorithm.adv_estimator=grpo \
    data.train_files=${HOME}/data/train.parquet \
    data.val_files=${HOME}/data/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \

    # ========== Reward配置（核心） ==========
    # 指定RewardManager名称（对应@register注册的名称）
    reward_model.reward_manager=my_reward_manager \

    # 指定自定义reward function的文件路径
    custom_reward_function.path=${REWARD_PATH} \

    # 指定reward function的函数名
    custom_reward_function.name=compute_score \

    # ========== 其他配置 ==========
    actor_rollout_ref.model.path=/path/to/model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    trainer.total_epochs=3 \
    trainer.save_freq=100
```

---

## 3. 数据协议

### 3.1 DataProto结构

VERL使用`DataProto`作为统一的数据协议：

```python
@dataclass
class DataProto:
    # Tensor数据（参与模型计算）
    batch: TensorDict = None
    # 非Tensor数据（每个样本的原始信息）
    non_tensor_batch: dict = field(default_factory=dict)
    # 全局元信息
    meta_info: dict = field(default_factory=dict)
```

### 3.2 样本数据格式要求

```json
{
    "data_source": "nq",
    "prompt": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "Question: who got the first nobel prize in physics?",
            "role": "user"
        }
    ],
    "reward_model": {
        "ground_truth": {
            "target": ["Wilhelm Röntgen", "Rontgen"]
        },
        "style": "rule"
    },
    "extra_info": {
        "index": 0,
        "question": "who got the first nobel prize in physics?",
        "split": "train"
    }
}
```

### 3.3 字段说明

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `data_source` | string | 是 | 数据来源标识 |
| `prompt` | list[dict] | 是 | 对话消息列表 |
| `reward_model.ground_truth` | dict | 是 | 标准答案 |
| `reward_model.ground_truth.target` | list[string] | 是 | 可接受的答案列表 |
| `reward_model.style` | string | 否 | reward计算风格 |
| `extra_info` | dict | 否 | 透传给reward function的额外信息 |

---

## 4. 文件结构汇总

```
verl_project_root/
├── run_train.sh                                    # 总控启动脚本
│
├── verl/
│   ├── experimental/
│   │   ├── reward_loop/
│   │   │   ├── reward_manager/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py                        # RewardManagerBase基类
│   │   │   │   ├── naive.py                       # 官方示例实现
│   │   │   │   └── my_reward_manager.py           # [新增] 自定义RewardManager
│   │
│   ├── workers/
│   │   ├── reward_manager/
│   │   │   ├── __init__.py
│   │   │   └── base.py                            # AbstractRewardManager基类
│   │
│   ├── utils/
│   │   ├── reward_score/
│   │   │   ├── __init__.py
│   │   │   ├── math_reward.py                     # 官方数学题reward示例
│   │   │   └── my_reward_function.py              # [新增] 自定义Reward Function
│   │
│   └── trainer/
│       ├── config/
│       │   └── ppo_trainer.yaml                   # 默认配置文件
│       └── main_ppo.py                            # 训练入口
│
└── data/
    ├── train.parquet                              # 训练数据
    └── test.parquet                               # 测试数据
```

---

## 5. 常见场景示例

### 5.1 精确匹配（Exact Match）

适用于：问答任务、数学题、代码题等有明确答案的场景

```python
def compute_score(data_source, solution_str, ground_truth, **kwargs):
    answer = extract_answer(solution_str)
    if answer and em_check(answer, ground_truth["target"]):
        return 1.0
    return 0.0
```

### 5.2 格式奖励 + 正确性奖励

适用于：需要模型遵循特定格式输出的场景

```python
def compute_score(data_source, solution_str, ground_truth, **kwargs):
    # 检查格式
    has_think = "<think>" in solution_str and "</think>" in solution_str
    has_answer = "<answer>" in solution_str and "</answer>" in solution_str

    format_reward = 0.0
    if has_think and has_answer:
        format_reward = 0.2  # 格式正确给0.2分

    # 检查答案正确性
    answer = extract_answer(solution_str)
    correct_reward = 0.0
    if answer and em_check(answer, ground_truth["target"]):
        correct_reward = 0.8  # 答案正确给0.8分

    return format_reward + correct_reward
```

### 5.3 多维度评分

适用于：需要从多个角度评估的复杂任务

```python
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    scores = {}

    # 1. 答案正确性 (权重0.5)
    answer = extract_answer(solution_str)
    scores["correctness"] = 1.0 if em_check(answer, ground_truth["target"]) else 0.0

    # 2. 推理步骤完整性 (权重0.3)
    think_content = extract_think(solution_str)
    scores["reasoning"] = evaluate_reasoning_steps(think_content)

    # 3. 工具使用合理性 (权重0.2)
    tool_calls = extract_tool_calls(solution_str)
    scores["tool_usage"] = evaluate_tool_usage(tool_calls, extra_info)

    # 加权求和
    final_score = (
        0.5 * scores["correctness"] +
        0.3 * scores["reasoning"] +
        0.2 * scores["tool_usage"]
    )

    return final_score
```

---

