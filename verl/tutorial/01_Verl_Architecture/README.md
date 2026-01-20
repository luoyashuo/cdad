# Verl 框架架构详解

> 本章节深入讲解 Verl (Volcano Engine RL) 框架的核心设计理念和架构，帮助你理解如何利用 Ray + PyTorch FSDP 实现高效的分布式 RLHF 训练。

## 1. Verl 概述

Verl 是字节跳动开源的分布式强化学习训练框架，专为大规模语言模型的 RLHF（Reinforcement Learning from Human Feedback）训练设计。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| **模型无关性** | 支持 HuggingFace Transformers、Megatron-LM 等多种后端 |
| **高效推理** | 集成 vLLM/SGLang 进行快速 Rollout |
| **灵活算法** | 支持 PPO、GRPO、RLOO、REINFORCE++ 等多种 RL 算法 |
| **FSDP 支持** | 完善的 PyTorch FSDP/FSDP2 分布式训练支持 |
| **Ray 分布式** | 基于 Ray 的灵活资源管理和 Worker 调度 |

### 1.2 与其他框架对比

```
┌─────────────────────────────────────────────────────────────────┐
│                      RLHF 训练框架对比                           │
├──────────────┬─────────────┬──────────────┬────────────────────┤
│    特性       │    TRL      │   OpenRLHF   │       Verl        │
├──────────────┼─────────────┼──────────────┼────────────────────┤
│ 分布式后端    │ Accelerate  │ Ray+DeepSpeed│  Ray+FSDP/Megatron│
│ 推理引擎      │ Native      │ vLLM         │  vLLM/SGLang      │
│ 多模型协调    │ 简单        │ 中等         │  灵活（Role-based）│
│ 显存效率      │ 中等        │ 高           │  高               │
│ 代码复杂度    │ 低          │ 中等         │  中等             │
└──────────────┴─────────────┴──────────────┴────────────────────┘
```

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Verl Training Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Driver Process (CPU)                          │   │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │  │                     RayPPOTrainer                               │ │   │
│   │  │  - 协调训练流程                                                  │ │   │
│   │  │  - 管理数据流转                                                  │ │   │
│   │  │  - 调度 Worker Groups                                           │ │   │
│   │  └─────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    ▼               ▼               ▼                        │
│   ┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐       │
│   │  ActorRollout WG    │ │   Critic WG     │ │  RewardModel WG     │       │
│   │  (GPU 0,1,2,3)      │ │  (GPU 0,1,2,3)  │ │  (可选, 独立资源)    │       │
│   │                     │ │                 │ │                     │       │
│   │  ┌───────────────┐  │ │ ┌─────────────┐ │ │ ┌─────────────────┐ │       │
│   │  │ Actor Model   │  │ │ │Critic Model │ │ │ │  Reward Model   │ │       │
│   │  │ (FSDP/LoRA)   │  │ │ │   (FSDP)    │ │ │ │  (冻结/FSDP)    │ │       │
│   │  └───────────────┘  │ │ └─────────────┘ │ │ └─────────────────┘ │       │
│   │  ┌───────────────┐  │ │                 │ │                     │       │
│   │  │ Rollout Engine│  │ │                 │ │                     │       │
│   │  │ (vLLM/SGLang) │  │ │                 │ │                     │       │
│   │  └───────────────┘  │ │                 │ │                     │       │
│   │  ┌───────────────┐  │ │                 │ │                     │       │
│   │  │  Ref Policy   │  │ │                 │ │                     │       │
│   │  │  (冻结/LoRA)  │  │ │                 │ │                     │       │
│   │  └───────────────┘  │ │                 │ │                     │       │
│   └─────────────────────┘ └─────────────────┘ └─────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件说明

#### 2.2.1 Driver Process

Driver 进程运行在单个 CPU 节点上，负责整体协调：

```python
# verl/trainer/ppo/ray_trainer.py
class RayPPOTrainer:
    """
    分布式 PPO 训练协调器

    职责：
    1. 初始化 Ray 资源池和 Worker Groups
    2. 协调训练循环 (rollout -> reward -> advantage -> update)
    3. 管理检查点保存/加载
    4. 收集和记录训练指标
    """

    def __init__(self, config, tokenizer, role_worker_mapping, ...):
        # 资源池管理
        self.resource_pool_manager = resource_pool_manager
        # Worker 角色映射
        self.role_worker_mapping = role_worker_mapping
        # 是否使用混合引擎 (Actor + Rollout 共享显存)
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
```

#### 2.2.2 Worker Groups

Worker Group 是一组运行在 GPU 上的 Ray Actor，负责实际的模型计算：

```python
# 角色定义 (verl/trainer/ppo/utils.py)
class Role(str, Enum):
    Actor = "actor"              # 策略模型 (可训练)
    Rollout = "rollout"          # 推理引擎 (生成序列)
    ActorRollout = "actor_rollout"  # 混合: Actor + Rollout
    RefPolicy = "ref"            # 参考策略 (冻结)
    ActorRolloutRef = "actor_rollout_ref"  # 混合: Actor + Rollout + Ref
    Critic = "critic"            # 价值函数 (PPO需要)
    RewardModel = "reward_model" # 奖励模型
```

#### 2.2.3 Hybrid Engine

Verl 的一大特色是 **Hybrid Engine** 设计，将 Actor 训练和 Rollout 推理合并到同一个 Worker：

```
传统方式:                          Hybrid Engine:
┌────────────┐  ┌────────────┐     ┌────────────────────────┐
│  Actor WG  │  │ Rollout WG │     │  ActorRollout WG       │
│  (GPU 0-3) │  │ (GPU 4-7)  │     │  (GPU 0-3)             │
│            │  │            │     │  ┌──────┐ ┌──────────┐ │
│ Train Mode │  │ Infer Mode │     │  │Actor │ │ Rollout  │ │
└────────────┘  └────────────┘     │  │(FSDP)│ │(vLLM)    │ │
      ↓              ↓              │  └──────┘ └──────────┘ │
   需要8张GPU                        └────────────────────────┘
                                          ↓
                                    只需要4张GPU!
```

**工作流程：**
1. **Rollout 阶段**: 从 FSDP 模型加载权重到 vLLM/SGLang，执行高效推理
2. **Training 阶段**: 在 FSDP 模型上执行梯度更新
3. **权重同步**: 训练完成后将权重同步回推理引擎

## 3. 资源调度与分布式通信

### 3.1 ResourcePoolManager

```python
# verl/trainer/ppo/ray_trainer.py
@dataclass
class ResourcePoolManager:
    """
    Ray 资源池管理器

    - resource_pool_spec: 定义每个资源池在各节点上的 GPU 数量
    - mapping: 将角色 (Role) 映射到资源池
    """
    resource_pool_spec: dict[str, list[int]]  # e.g., {"global": [4, 4]}
    mapping: dict[Role, str]                   # e.g., {Role.Actor: "global"}
```

**配置示例（单机 4 卡）：**
```python
# 所有角色共享同一个资源池
resource_pool_spec = {
    "global_pool": [4]  # 1个节点，4张GPU
}

mapping = {
    Role.ActorRolloutRef: "global_pool",
    Role.Critic: "global_pool",  # 如果使用 PPO
}
```

**配置示例（多机 8 卡 × 2）：**
```python
resource_pool_spec = {
    "actor_pool": [8, 8],   # 2个节点，每节点8张GPU
    "reward_pool": [4, 4],  # 单独的奖励模型资源池
}

mapping = {
    Role.ActorRolloutRef: "actor_pool",
    Role.RewardModel: "reward_pool",
}
```

### 3.2 Worker Colocate

当多个 Worker 需要共享 GPU 时，Verl 使用 `create_colocated_worker_cls` 创建融合 Worker：

```python
# verl/single_controller/ray/base.py
def create_colocated_worker_cls(worker_classes: list, worker_names: list):
    """
    创建共置 Worker 类

    将多个 Worker 类合并为一个，共享 GPU 内存
    例如: Actor + Rollout + RefPolicy -> ActorRolloutRefWorker
    """
    ...
```

## 4. FSDP 分布式训练

### 4.1 FSDPEngine

Verl 封装了 PyTorch FSDP 为 `FSDPEngine`：

```python
# verl/workers/engine/fsdp/transformer_impl.py
class FSDPEngine(BaseEngine):
    """
    FSDP 训练引擎

    特性:
    - 支持 FSDP1 (FULL_SHARD) 和 FSDP2
    - 支持 HSDP (Hybrid Sharded Data Parallel)
    - 支持 LoRA 适配器
    - 支持序列并行 (Ulysses)
    - 支持梯度检查点和激活卸载
    """

    def _build_fsdp_module(self, model, ...):
        """使用 FSDP 包装模型"""
        ...

    def forward_backward_batch(self, batch, ...):
        """执行前向和反向传播"""
        ...

    def optimizer_step(self, ...):
        """梯度裁剪 + 优化器更新"""
        ...
```

### 4.2 FSDP 配置选项

```yaml
# verl/trainer/config/engine/fsdp.yaml
engine:
  fsdp_config:
    # 分片策略
    sharding_strategy: FULL_SHARD  # 或 HYBRID_SHARD, SHARD_GRAD_OP

    # CPU 卸载 (节省显存)
    cpu_offload: false

    # 混合精度
    mixed_precision:
      param_dtype: bfloat16
      reduce_dtype: float32
      buffer_dtype: bfloat16

    # 梯度检查点 (节省显存)
    activation_checkpointing: true

    # 序列并行 (长序列)
    use_ulysses: false
    ulysses_degree: 1
```

### 4.3 LoRA 支持

Verl 原生支持 LoRA 训练，无需单独的 Critic 模型：

```yaml
# 配置 LoRA
actor_rollout_ref:
  model:
    lora:
      rank: 64
      alpha: 128
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
      dropout: 0.05
```

使用 LoRA 时，Reference Policy 直接复用 Actor 的基座权重（禁用 LoRA adapter），无需额外显存。

## 5. RL 训练流程

### 5.1 标准 PPO 训练循环

```python
# verl/trainer/ppo/ray_trainer.py - fit() 方法简化版
def fit(self):
    for epoch in range(total_epochs):
        for batch in train_dataloader:
            # ============ 1. Rollout: 生成响应 ============
            # 使用 vLLM/SGLang 进行高效推理
            gen_batch = self.actor_rollout_wg.generate_sequences(batch)

            # ============ 2. 计算奖励 ============
            # 可以是规则奖励、模型奖励或外部API
            if self.use_rm:
                rm_scores = self.rm_wg.compute_rm_score(gen_batch)
            reward_tensor = compute_reward(gen_batch, self.reward_fn)

            # ============ 3. 计算 Reference Log Prob ============
            # 用于 KL 惩罚
            if self.use_reference_policy:
                ref_log_prob = self.ref_wg.compute_ref_log_prob(gen_batch)

            # ============ 4. 计算 Value (PPO 需要) ============
            if self.use_critic:
                values = self.critic_wg.compute_values(gen_batch)

            # ============ 5. 计算 Old Log Prob ============
            old_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch)

            # ============ 6. 应用 KL 惩罚 ============
            if self.config.algorithm.use_kl_in_reward:
                gen_batch = apply_kl_penalty(gen_batch, self.kl_ctrl)

            # ============ 7. 计算 Advantage ============
            gen_batch = compute_advantage(
                gen_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam
            )

            # ============ 8. 更新 Critic ============
            if self.use_critic:
                self.critic_wg.update_critic(gen_batch)

            # ============ 9. 更新 Actor ============
            self.actor_wg.update_actor(gen_batch)
```

### 5.2 GRPO 训练流程（简化版）

GRPO (Group Relative Policy Optimization) 相比 PPO 更简洁，**不需要 Critic 模型**：

```python
def fit_grpo(self):
    for epoch in range(total_epochs):
        for batch in train_dataloader:
            # 1. Rollout: 每个 prompt 生成 N 个响应
            gen_batch = self.actor_rollout_wg.generate_sequences(
                batch,
                n=self.config.rollout.n  # 例如 n=8
            )

            # 2. 计算奖励 (Outcome Reward)
            reward_tensor = compute_reward(gen_batch, self.reward_fn)

            # 3. 计算 GRPO Advantage
            # advantage = (reward - group_mean) / group_std
            gen_batch = compute_advantage(
                gen_batch,
                adv_estimator=AdvantageEstimator.GRPO,
                norm_adv_by_std_in_grpo=True
            )

            # 4. 更新 Actor (无需 Critic)
            self.actor_wg.update_actor(gen_batch)
```

## 6. 优势估计算法

### 6.1 支持的优势估计器

```python
# verl/trainer/ppo/core_algos.py
class AdvantageEstimator(str, Enum):
    GAE = "gae"                    # Generalized Advantage Estimation (PPO)
    GRPO = "grpo"                  # Group Relative Policy Optimization
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"  # REINFORCE++
    REMAX = "remax"                # ReMax
    RLOO = "rloo"                  # RLOO (Leave-One-Out)
    OPO = "opo"                    # Optimal Policy Optimization
    GPG = "gpg"                    # Generalized Policy Gradient
```

### 6.2 GRPO 优势计算

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,  # prompt ID，用于分组
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    GRPO 优势计算:

    对于每个 prompt 组 g:
        advantage_i = (reward_i - mean(rewards_g)) / std(rewards_g)

    特点:
    - 只使用 Outcome Reward (每个响应一个标量奖励)
    - 在同一 prompt 的多个响应间进行相对比较
    - 不需要 Value Function
    """
    scores = token_level_rewards.sum(dim=-1)  # 汇总为 outcome reward

    # 按 prompt ID 分组计算 mean 和 std
    id2score = defaultdict(list)
    for i in range(len(scores)):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        group_scores = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(group_scores)
        id2std[idx] = torch.std(group_scores)

    # 计算归一化优势
    for i in range(len(scores)):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]  # Dr.GRPO

    return scores.unsqueeze(-1) * response_mask, scores.unsqueeze(-1) * response_mask
```

### 6.3 GAE vs GRPO 对比

| 特性 | GAE (PPO) | GRPO |
|------|-----------|------|
| 奖励类型 | Token-level | Outcome (序列级) |
| 需要 Critic | 是 | 否 |
| 计算复杂度 | 高 | 低 |
| 训练稳定性 | 需要调参 | 更稳定 |
| 适用场景 | 细粒度反馈 | 序列级反馈 (数学、代码) |

## 7. 数据协议 (DataProto)

Verl 使用 `DataProto` 作为统一的数据传输协议：

```python
# verl/protocol.py
class DataProto:
    """
    统一数据传输协议

    Attributes:
        batch: TensorDict - 张量数据 (input_ids, attention_mask, rewards 等)
        non_tensor_batch: dict - 非张量数据 (字符串、元数据等)
        meta_info: dict - 元信息 (batch_size, device 等)
    """

    def union(self, other: DataProto) -> DataProto:
        """合并两个 DataProto"""
        ...

    def repeat(self, n: int) -> DataProto:
        """重复数据 n 次 (用于 rollout.n > 1)"""
        ...

    def reorder(self, indices) -> DataProto:
        """重新排序 (用于序列长度平衡)"""
        ...
```

**常见字段：**
```python
batch.batch = {
    "input_ids": Tensor,           # 输入 token IDs
    "attention_mask": Tensor,      # 注意力掩码
    "prompts": Tensor,             # Prompt 部分
    "responses": Tensor,           # 生成的响应
    "response_mask": Tensor,       # 响应掩码
    "old_log_probs": Tensor,       # Actor 的 log 概率
    "ref_log_prob": Tensor,        # Reference Policy 的 log 概率
    "token_level_scores": Tensor,  # Token 级别奖励
    "token_level_rewards": Tensor, # 包含 KL 惩罚的 token 级别奖励
    "advantages": Tensor,          # 优势函数值
    "returns": Tensor,             # 回报值
    "values": Tensor,              # Critic 的价值估计 (PPO)
}

batch.non_tensor_batch = {
    "uid": np.ndarray,             # Prompt ID (用于 GRPO 分组)
    "reward_model": dict,          # 奖励模型信息 (ground_truth 等)
}
```

## 8. 小结

本章介绍了 Verl 框架的核心架构：

1. **Driver-Worker 架构**: Driver 协调，Worker 执行计算
2. **Hybrid Engine**: Actor + Rollout 共享 GPU，节省显存
3. **ResourcePoolManager**: 灵活的 GPU 资源分配
4. **FSDP 集成**: 高效的分布式训练
5. **多种优势估计器**: 支持 PPO、GRPO 等多种算法
6. **DataProto**: 统一的数据传输协议

下一章我们将介绍环境配置，包括如何使用 uv + CUDA 12.4 搭建 Verl 训练环境。
