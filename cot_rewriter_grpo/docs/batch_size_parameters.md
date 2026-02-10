# VERL/PPO Batch Size 参数详解

本文档详细解释 VERL 框架中与 batch size 相关的参数，帮助理解它们如何影响显存使用和训练效率。

## 参数层级关系

```
train_batch_size (prompts)
    × rollout.n (每个 prompt 生成的响应数)
    = 总 trajectories 数
        ÷ ppo_mini_batch_size
        = mini_batches 数量
            ÷ ppo_micro_batch_size_per_gpu
            = gradient_accumulation 步数
```

## 各参数详解

### 1. `data.train_batch_size`

**含义**: 每个训练步骤使用的 prompt 数量（全局）

**示例**: `train_batch_size: 4` 表示每步从数据集取 4 个 prompt

**影响**:
- 决定了每步训练的数据量
- 总 trajectories = `train_batch_size × rollout.n`

---

### 2. `actor_rollout_ref.rollout.n`

**含义**: 每个 prompt 生成的响应（trajectory）数量

**示例**: `n: 8` 表示每个 prompt 生成 8 个不同的响应

**影响**:
- GRPO 需要多个响应来计算组内相对优势
- 总 trajectories = `train_batch_size × n`

**约束**: `ppo_mini_batch_size × n >= n_gpus_per_node`

---

### 3. `actor_rollout_ref.actor.ppo_mini_batch_size` ⭐

**含义**: PPO 更新时每个 mini-batch 包含的 trajectory 数量（全局，跨所有 worker）

**作用**:
- 将所有 trajectories 分割成多个 mini-batch
- 每个 mini-batch 执行一次完整的 forward + backward + optimizer step
- 较小的值 = 更多的 optimizer steps = 更细粒度的更新

**代码位置**: `verl/verl/workers/actor/dp_actor.py:377`
```python
mini_batches = data.split(self.config.ppo_mini_batch_size)
```

**示例计算**:
```
总 trajectories = 4 prompts × 8 responses = 32
ppo_mini_batch_size = 4
→ mini_batches 数量 = 32 ÷ 4 = 8 个 mini-batch
→ 每个 epoch 执行 8 次 optimizer step
```

**显存影响**:
- 影响 optimizer step 的频率，间接影响训练稳定性
- 不直接影响单次 forward/backward 的显存峰值

---

### 4. `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` ⭐⭐⭐

**含义**: 每个 GPU 上单次 forward/backward 处理的样本数

**这是控制显存的关键参数！**

**作用**:
- 将 mini-batch 进一步分割为 micro-batch
- 使用梯度累积（gradient accumulation）来模拟大 batch
- 较小的值 = 单次 forward/backward 显存更少

**代码位置**: `verl/verl/workers/actor/dp_actor.py:386-389`
```python
self.gradient_accumulation = (
    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
)
micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
```

**示例计算**:
```
ppo_mini_batch_size = 4
ppo_micro_batch_size_per_gpu = 1
→ gradient_accumulation = 4 ÷ 1 = 4 步
→ 每 4 次 micro forward/backward 后执行一次 optimizer step
```

**显存影响**:
- **直接决定 backward 时的显存峰值**
- 值越小 → 显存使用越少（但训练速度可能变慢）
- OOM 时首先降低此参数

---

### 5. `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`

**含义**: 计算 log probability 时每个 GPU 的 micro batch size

**作用**:
- 仅影响 rollout 阶段的 log_prob 计算
- 不涉及梯度计算，显存需求相对较小

---

## 实际配置示例

### 场景: 4 × 80GB GPU，Qwen3-4B 全参数训练

```yaml
data:
  train_batch_size: 4          # 4 个 prompt

actor_rollout_ref:
  rollout:
    n: 8                       # 每个 prompt 生成 8 个响应
    # 总 trajectories = 4 × 8 = 32

  actor:
    ppo_mini_batch_size: 2     # 每个 mini-batch 2 个 trajectory
    # mini_batches 数量 = 32 ÷ 2 = 16

    ppo_micro_batch_size_per_gpu: 1  # 每次 forward/backward 只处理 1 个样本
    # gradient_accumulation = 2 ÷ 1 = 2
```

### OOM 调优顺序

1. **首先降低** `ppo_micro_batch_size_per_gpu`（如 2 → 1）
   - 直接减少 backward 显存峰值

2. **其次降低** `ppo_mini_batch_size`（如 4 → 2）
   - 增加 optimizer step 频率

3. **再考虑** 降低 `gpu_memory_utilization`
   - 为训练留更多显存空间

4. **最后考虑** 启用 LoRA 或 FSDP offload
   - 减少模型参数的显存占用

---

## 数学关系总结

```
总 trajectories = train_batch_size × rollout.n
mini_batches 数量 = 总 trajectories ÷ ppo_mini_batch_size
gradient_accumulation = ppo_mini_batch_size ÷ ppo_micro_batch_size_per_gpu
每个 epoch 的 optimizer steps = mini_batches 数量
```

## 约束条件

1. `ppo_mini_batch_size` 必须能被 `ppo_micro_batch_size_per_gpu` 整除
2. `train_batch_size × rollout.n` 必须能被 `n_gpus_per_node` 整除
3. `ppo_mini_batch_size × rollout.n >= n_gpus_per_node`（GRPO 要求）
