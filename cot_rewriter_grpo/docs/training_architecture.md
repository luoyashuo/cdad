# CoT Rewriter GRPO 训练架构文档

本文档详细说明 CoT Rewriter GRPO 训练的完整架构，包括：
- Hybrid Engine 设计（FSDP Actor vs vLLM Engine）
- 三阶段训练流程（Rollout → Reward → Training）
- GPU 显存管理与优化
- Reward 计算逻辑

---

## 1. 系统概述

### 1.1 硬件配置

- **GPU**: 4 x 80GB (A100/H100)
- **Actor Model**: Qwen3-4B (全参数训练)
- **Student Model**: Qwen3-1.7B (Reward 计算)
- **框架**: VERL + vLLM + FSDP + Ray

### 1.2 训练目标

将原始 CoT (Chain-of-Thought) 推理过程改写得更清晰高效，使用 GRPO 算法优化。

**优化目标**: 改写后的 CoT 能让 Student Model 更容易预测正确答案。

---

## 2. Hybrid Engine 架构

VERL 采用 **Hybrid Engine** 设计，同一个模型有两种形态：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Qwen3-4B 模型 (同一份权重，两种形态)                     │
├─────────────────────────────────────┬───────────────────────────────────────┤
│         FSDP Actor                  │           vLLM Engine                 │
│         (PyTorch 训练形态)           │           (推理引擎形态)               │
├─────────────────────────────────────┼───────────────────────────────────────┤
│  用途: 梯度计算、参数更新             │  用途: 高效生成 response              │
│  特点: 支持分布式训练 (FSDP 分片)     │  特点: PagedAttention, KV Cache      │
│        支持梯度检查点                │        Continuous Batching           │
│        支持混合精度训练              │        高效采样                       │
│  显存: ~8-10 GB (4卡分片后)          │  显存: ~56 GB (含预分配 KV cache)     │
└─────────────────────────────────────┴───────────────────────────────────────┘
```

### 2.1 为什么需要两套？

| 需求 | FSDP Actor | vLLM Engine |
|------|------------|-------------|
| **分布式训练** | ✅ FSDP 分片、梯度同步 | ❌ 不支持反向传播 |
| **高效推理** | ❌ 无 KV Cache 优化 | ✅ PagedAttention |
| **动态批处理** | ❌ 固定 batch size | ✅ Continuous Batching |
| **显存效率(推理)** | ❌ 需要存梯度、优化器 | ✅ 只存推理所需 |

**结论**：vLLM 生成速度比 HuggingFace 快 5-10x，但不支持训练，所以需要两套。

### 2.2 各阶段谁在工作

| 阶段 | FSDP Actor | vLLM Engine | 说明 |
|------|------------|-------------|------|
| **Rollout** | 闲置 (可 offload) | **工作** | vLLM 高效生成 response |
| **Reward** | 闲置 (offload 到 CPU) | 闲置 (sleep) | Student vLLM 计算 |
| **Training** | **工作** | 闲置 (sleep) | FSDP 梯度更新 |

### 2.3 关键代码位置

**FSDP Actor 创建** - `verl/workers/fsdp_workers.py:368-410`
```python
actor_module_fsdp = FSDP(
    actor_module,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=cpu_offload,
    mixed_precision=mixed_precision,
)
```

**vLLM Engine 创建** - `verl/workers/fsdp_workers.py:480-520`
```python
rollout = vLLMRollout(actor_module=self.actor_module_fsdp, config=self.config.rollout)
rollout_sharding_manager = FSDPVLLMShardingManager(
    module=self.actor_module_fsdp,
    inference_engine=rollout.inference_engine,
)
```

**权重同步 (FSDP → vLLM)** - `verl/workers/sharding_manager/fsdp_vllm.py:194-236`
```python
def __enter__(self):
    params = self.module.state_dict()           # 收集 FSDP 分片参数
    self.inference_engine.wake_up(tags=["weights"])  # 激活 vLLM
    self.update_params(params)                  # 同步权重
    self.inference_engine.wake_up(tags=["kv_cache"]) # 初始化 KV cache
```

---

## 3. 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GRPO Training Loop                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 1: Rollout     →  Stage 2: Reward     →  Stage 3: Training          │
│  (Actor vLLM 生成)       (Student vLLM 计算)    (FSDP 梯度更新)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Stage 1: Rollout (生成 Response)

**目的**: 用 Actor vLLM 生成改写后的 CoT

**详细流程**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: 进入 sharding_manager 上下文 (__enter__)                            │
│   FSDP Actor ──► 同步权重 ──► vLLM Engine (加载完整权重 + KV Cache)          │
│   代码: fsdp_vllm.py:194-236                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 2: vLLM 推理生成                                                       │
│   vLLM Engine 工作: PagedAttention + Continuous Batching                   │
│   代码: fsdp_workers.py:720-756 generate_sequences()                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 3: 退出 sharding_manager 上下文 (__exit__)                             │
│   vLLM Engine ──► sleep(level=2) ──► 释放 ~55GB                            │
│   FSDP Actor  ──► offload_to_cpu ──► 释放 ~10GB                            │
│   代码: fsdp_vllm.py:244-327                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**显存占用**: ~60 GB
| 组件 | 显存 | 说明 |
|------|------|------|
| Actor vLLM Engine | ~56 GB | `gpu_memory_utilization=0.7` |
| FSDP Actor (Sharded) | ~2 GB | FSDP 分片后的本地参数 |
| 其他 | ~2 GB | CUDA context |

### 3.2 Stage 2: Reward 计算

**目的**: 计算改写后 CoT 的奖励值

**详细流程**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: FSDP Actor Offload (已在 Stage 1 __exit__ 完成)                     │
│   FSDP 参数 + Optimizer ──► offload 到 CPU ──► 释放 ~25-40GB               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 2: Student vLLM 加载                                                   │
│   每个 GPU 启动一个 Ray Actor，运行 Qwen3-1.7B vLLM                         │
│   代码: student_reward.py:291-321                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 3: 计算 Reward                                                        │
│   R_cog = log_prob(rewritten) - log_prob(original)                         │
│   代码: combined_reward.py:129-222                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4: Student vLLM Cleanup                                               │
│   ray.kill(worker) + gc.collect() + torch.cuda.empty_cache()               │
│   代码: combined_reward.py:237-284                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**显存占用**: ~45 GB
| 组件 | 显存 | 说明 |
|------|------|------|
| Student vLLM | ~32 GB | `gpu_memory_utilization=0.4` |
| FSDP 残留 | ~10 GB | metadata, buffers |
| logprobs 计算 | ~3 GB | 临时显存 |

### 3.3 Stage 3: Training (Actor 更新)

**目的**: 用 GRPO 算法更新 Actor 参数

**详细流程**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Reload FSDP 参数和 Optimizer                                        │
│   CPU ──► GPU (reload for training)                                        │
│   代码: fsdp_workers.py:679-682                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 2: 执行 PPO 更新                                                       │
│   • 计算 old_log_probs                                                     │
│   • 计算 advantages (GRPO)                                                 │
│   • 执行梯度更新                                                           │
│   代码: fsdp_workers.py:684-707                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 3: Offload FSDP 参数和 Optimizer                                       │
│   GPU ──► CPU (offload for next rollout)                                   │
│   代码: fsdp_workers.py:709-714                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**显存占用**: ~55 GB
| 组件 | 显存 | 说明 |
|------|------|------|
| FSDP Actor 参数 | ~10 GB | 从 CPU reload |
| Optimizer 状态 | ~20 GB | Adam m + v |
| 梯度 | ~10 GB | FP16 |
| 激活值 | ~15 GB | gradient checkpointing |

---

## 4. Reward 计算逻辑

### 4.1 Reward 公式

```
R_total = R_cog + λ × R_logic
```

当前配置 `use_logic_reward: false`，所以：
```
R_total = R_cog
```

### 4.2 R_cog (认知奖励) 详解

```python
# 计算 answer 的 log probability
answer_log_prob(q, cot, a) = mean(token_log_probs) + α × min(token_log_probs)

# 认知奖励 = 改写后的 log prob - 原始的 log prob
R_cog = answer_log_prob(Q, Rewritten_CoT, A) - answer_log_prob(Q, Original_CoT, A)
```

**参数**:
- `α (alpha_cog) = 0.01`: min log prob 权重，鼓励避免极低置信度 token
- `λ (lambda_logic) = 0.5`: 逻辑奖励权重（当前未启用）

**含义**:
- 使用 Student Model (Qwen3-1.7B) 计算 `P(Answer | Question, CoT)`
- 改写后的 CoT 让 Student 更容易预测正确答案 → **正奖励**
- 改写后的 CoT 让 Student 更难预测正确答案 → **负奖励**

### 4.3 计算流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Reward 计算流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    • Question (问题)                                                        │
│    • Original CoT (原始推理过程)                                             │
│    • Rewritten CoT (Actor 生成的改写)                                        │
│    • Ground Truth Answer (正确答案)                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Student Model (Qwen3-1.7B vLLM)                                     │   │
│  │                                                                     │   │
│  │  构建 prompt: "<Question> <think>CoT</think> <Answer>"              │   │
│  │                                                                     │   │
│  │  P(Answer | Q, Rewritten_CoT) ──► rewrite_log_prob                 │   │
│  │                                                                     │   │
│  │  answer_log_prob = mean + 0.01 × min                               │   │
│  │                                                                     │   │
│  │  R_cog = rewrite_log_prob (直接使用 rewrite 的 log prob)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  特殊情况:                                                                  │
│    • 截断的 response → R = -1.0 (跳过 Student 计算)                         │
│                                                                             │
│  输出: R_total = R_cog                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 代码位置

| 功能 | 文件 | 行号 | 方法 |
|------|------|------|------|
| Reward 入口 | `combined_reward.py` | 287-583 | `compute_cot_rewriter_reward()` |
| 截断检测 | `combined_reward.py` | 374-417 | 检测并跳过截断样本 |
| 认知奖励计算 | `combined_reward.py` | 129-222 | `CombinedRewardComputer.compute_reward()` |
| Student 初始化 | `student_reward.py` | 291-321 | `VLLMStudentRewardClient.initialize()` |
| Log prob 计算 | `student_reward.py` | 90-143 | `VLLMWorkerActor.compute_logprobs()` |
| 奖励差值计算 | `student_reward.py` | 469-491 | `reward = rewrite - original` |

### 4.5 当前配置

```yaml
# examples/qwen3_4b_grpo_ruc_4gpu_80g.yaml
custom_reward_function:
  reward_kwargs:
    student_model_path: Qwen/Qwen3-1.7B
    alpha: 0.01              # min log prob 权重
    lambda_logic: 0.5        # 逻辑奖励权重
    use_logic_reward: false  # 当前禁用逻辑奖励
    truncation_penalty: -1.0 # 截断 response 的惩罚
    vllm_gpu_memory_utilization: 0.4
```

---

## 5. GPU 显存管理

### 5.1 显存流转图

```
时间线 ──────────────────────────────────────────────────────────────────────►

Stage 1: Rollout (~60 GB)
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Actor vLLM (56 GB): Weights + KV Cache                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌───────────┐                                               │
│ │ FSDP (2GB)│                                               │
│ └───────────┘                                               │
└─────────────────────────────────────────────────────────────┘
                │
                │ vLLM sleep(level=2) + FSDP offload
                ▼
Stage 2: Reward (~45 GB)
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Student vLLM (32 GB): Weights + KV Cache                │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────┐                                             │
│ │ FSDP残留(10GB)│                                           │
│ └─────────────┘                                             │
└─────────────────────────────────────────────────────────────┘
                │
                │ Student cleanup + FSDP reload
                ▼
Stage 3: Training (~55 GB)
┌─────────────────────────────────────────────────────────────┐
│ ┌───────────────────┐ ┌───────────────────┐                 │
│ │ FSDP Params (10GB)│ │ Optimizer (20 GB) │                 │
│ └───────────────────┘ └───────────────────┘                 │
│ ┌───────────────────┐ ┌───────────────────┐                 │
│ │ Gradients (10 GB) │ │ Activations (15GB)│                 │
│ └───────────────────┘ └───────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                │
                │ FSDP offload + vLLM wake_up
                ▼
Stage 1: Rollout (下一轮)
```

### 5.2 关键 Offload 代码位置

| 操作 | 文件 | 行号 | 说明 |
|------|------|------|------|
| vLLM sleep | `fsdp_vllm.py` | 293 | `inference_engine.sleep(level=2)` |
| FSDP 参数 offload | `fsdp_workers.py` | 709-710 | `offload_fsdp_model_to_cpu()` |
| FSDP optimizer offload | `fsdp_workers.py` | 712-713 | `offload_fsdp_optimizer()` |
| FSDP 参数 reload | `fsdp_workers.py` | 679-680 | `load_fsdp_model_to_gpu()` |
| Student vLLM cleanup | `combined_reward.py` | 237-284 | `cleanup_reward_computer()` |

### 5.3 OOM 问题案例

**问题**: Student vLLM 计算 logprobs 时 OOM

**错误日志**:
```
GPU 0: 51.9GB / 80.0GB (Student vLLM 加载后)
torch.OutOfMemoryError: Tried to allocate 8.73 GiB.
Process 16013 has 13.05 GiB memory in use.   # FSDP 残留
Process 40457 has 59.57 GiB memory in use.   # Student vLLM
```

**解决方案**:
1. 启用 FSDP offload: `param_offload: true`, `optimizer_offload: true`
2. 降低 Student vLLM 显存: `vllm_gpu_memory_utilization: 0.4`

---

## 6. 配置参考

### 6.1 80GB GPU 推荐配置

```yaml
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 1
    ppo_micro_batch_size_per_gpu: 1
    fsdp_config:
      dtype: float16
      param_offload: true
      optimizer_offload: true
  rollout:
    gpu_memory_utilization: 0.7
    sleep_level: 2
    free_cache_engine: true
  model:
    enable_gradient_checkpointing: true

custom_reward_function:
  reward_kwargs:
    vllm_gpu_memory_utilization: 0.4
    cleanup_after_compute: true
    truncation_penalty: -1.0
    alpha: 0.01
    use_logic_reward: false
```

### 6.2 显存预算参考

| GPU 规格 | Actor vLLM | Student vLLM | FSDP 残留 | 预留 |
|----------|------------|--------------|-----------|------|
| 80GB | 0.7 (56GB) | 0.4 (32GB) | ~10GB | ~38GB |
| 40GB | 0.5 (20GB) | 0.25 (10GB) | ~5GB | ~5GB |

---

## 7. 参考文件

| 文件 | 说明 |
|------|------|
| `verl/workers/fsdp_workers.py` | FSDP Worker，Rollout/Training 入口 |
| `verl/workers/sharding_manager/fsdp_vllm.py` | vLLM 显存管理，sleep/wake_up |
| `verl/experimental/reward/cot_rewriter/combined_reward.py` | Reward 计算入口 |
| `verl/experimental/reward/cot_rewriter/student_reward.py` | Student vLLM 实现 |
| `verl/trainer/ppo/ray_trainer.py` | 训练主循环 |
| `examples/qwen3_4b_grpo_ruc_4gpu_80g.yaml` | 训练配置 |
