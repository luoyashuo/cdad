# 最优配置方案：7B 全参 GRPO + 1.7B 统一奖励模型

> 本文档详细说明在 **4×A100-80G** 环境下，如何配置 7B 全参 GRPO 训练 + 1.7B 统一奖励模型的最优方案。

## 1. 硬件与场景

### 1.1 硬件配置

| 资源 | 规格 |
|------|------|
| GPU | 4 × NVIDIA A100-80GB |
| CPU 内存 | 256 GB (用于优化器状态 offload) |
| 网络 | NVLink / PCIe 4.0 |

### 1.2 训练场景

| 参数 | 值 |
|------|-----|
| Actor Model | Qwen-2.5-7B (全参训练) |
| 奖励模型 | Qwen-2.5-1.7B (统一模型，冻结) |
| 输入序列长度 | 4000 tokens |
| 输出序列长度 | 4000 tokens |
| GRPO 采样数 | 4 (每 prompt 生成 4 个响应) |

## 2. 显存分析

### 2.1 各组件显存需求

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         显存需求详细分解                                     │
├──────────────────────────────┬─────────────┬───────────────────────────────┤
│           组件                │   显存      │           说明                │
├──────────────────────────────┼─────────────┼───────────────────────────────┤
│                              │             │                               │
│ 【Actor Model - 7B 全参】     │             │                               │
│ ├─ 模型权重 (bf16)           │   14 GB     │ 7B × 2 bytes                  │
│ ├─ 梯度 (bf16)               │   14 GB     │ 7B × 2 bytes                  │
│ ├─ Adam 一阶动量 m (fp32)    │   28 GB     │ 7B × 4 bytes → CPU Offload    │
│ └─ Adam 二阶动量 v (fp32)    │   28 GB     │ 7B × 4 bytes → CPU Offload    │
│                              │             │                               │
│ 【FSDP 分片后 (4卡)】         │             │                               │
│ ├─ 权重/卡                   │   3.5 GB    │ 14 GB ÷ 4                     │
│ ├─ 梯度/卡                   │   3.5 GB    │ 14 GB ÷ 4                     │
│ └─ 优化器状态                │   0 GB      │ 全部 offload 到 CPU           │
│                              │             │                               │
│ 【激活值 (梯度检查点)】        │             │                               │
│ ├─ 8K seq, micro_bs=1       │   ~15 GB    │ 开启 gradient checkpointing   │
│ └─ 不开检查点                │   ~100 GB   │ 不可接受                      │
│                              │             │                               │
│ 【Rollout Engine (SGLang)】  │             │                               │
│ ├─ 模型权重 (共享 Actor)     │   0 GB      │ Hybrid Engine 共享            │
│ └─ KV Cache (n=4, 8K seq)   │   ~20 GB    │ 每卡                          │
│                              │             │                               │
│ 【统一奖励模型 - 1.7B】       │             │                               │
│ ├─ 模型权重 (bf16)           │   3.4 GB    │ 1.7B × 2 bytes               │
│ └─ 推理激活                  │   ~2 GB     │ 短序列，较小                   │
│                              │             │                               │
└──────────────────────────────┴─────────────┴───────────────────────────────┘
```

### 2.2 最终显存分布

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4×A100-80G 显存分布 (最优方案)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GPU 0:                                                                     │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │  Actor 权重分片 (FSDP)                              3.5 GB        │     │
│   │  Actor 梯度分片                                     3.5 GB        │     │
│   │  激活值 (梯度检查点)                                ~15 GB        │     │
│   │  Rollout KV Cache                                  ~20 GB        │     │
│   │  ─────────────────────────────────────────────────────────        │     │
│   │  统一奖励模型 (1.7B)                                3.4 GB        │     │
│   │  奖励推理激活                                       ~2 GB         │     │
│   │  ═══════════════════════════════════════════════════════════      │     │
│   │  GPU 0 总计                                        ~47.4 GB       │     │
│   │  剩余空间                                          ~32.6 GB  ✓    │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   GPU 1-3 (每张):                                                            │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │  Actor 权重分片 (FSDP)                              3.5 GB        │     │
│   │  Actor 梯度分片                                     3.5 GB        │     │
│   │  激活值 (梯度检查点)                                ~15 GB        │     │
│   │  Rollout KV Cache                                  ~20 GB        │     │
│   │  ═══════════════════════════════════════════════════════════      │     │
│   │  GPU 1-3 总计                                      ~42 GB        │     │
│   │  剩余空间                                          ~38 GB  ✓     │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   CPU 内存:                                                                  │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │  Adam 优化器状态 (m + v)                           56 GB         │     │
│   │  数据加载缓冲                                      ~10 GB         │     │
│   │  ═══════════════════════════════════════════════════════════      │     │
│   │  CPU 总计                                         ~66 GB         │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│   ┌───────────────────────────────────────────────────────────────────┐     │
│   │  总 GPU 显存: 47.4 + 42×3 = 173.4 GB / 320 GB (54%)              │     │
│   │  峰值余量充足，可应对通信和临时缓冲                                 │     │
│   └───────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 最优配置文件

### 3.1 训练配置

```yaml
# configs/optimal_7b_fullparam_grpo.yaml
# 最优配置：7B 全参 GRPO + 1.7B 统一奖励模型
# 硬件要求：4×A100-80G, CPU 内存 256GB

hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

# ============================================================
#                      实验配置
# ============================================================
trainer:
  project_name: cot_rewriter_grpo
  experiment_name: 7b_fullparam_unified_reward
  total_epochs: 3
  save_freq: 100
  val_freq: 50
  log_freq: 10
  device: cuda

  # 资源配置
  n_gpus_per_node: 4
  nnodes: 1

# ============================================================
#                      数据配置
# ============================================================
data:
  train_files: data/processed/train.parquet
  val_files: data/processed/val.parquet

  # Batch 配置
  train_batch_size: 8           # 每 step 8 个 prompt
  val_batch_size: 16
  gen_batch_size: 8

  # 序列长度
  max_prompt_length: 4000
  max_response_length: 4000

  # 数据加载
  return_raw_chat: true
  dataloader_num_workers: 4
  reward_fn_key: reward_model

  # 序列长度平衡 (优化显存利用)
  use_seqlen_balancing: true

# ============================================================
#                 Actor/Rollout 混合引擎配置
# ============================================================
actor_rollout_ref:
  hybrid_engine: true           # 必须为 true，共享显存

  # ---------- 模型配置 ----------
  model:
    path: ./models/Qwen2.5-7B-Instruct
    trust_remote_code: true
    dtype: bfloat16
    # 全参训练，不使用 LoRA

  # ---------- Actor 训练配置 ----------
  actor:
    # PPO/GRPO 参数
    ppo_epochs: 1               # GRPO 通常 1 epoch 即可
    clip_ratio: 0.2
    entropy_coeff: 0.01

    # 显存优化 (关键!)
    gradient_checkpointing: true
    grad_clip: 1.0

    # FSDP 配置
    fsdp_config:
      sharding_strategy: FULL_SHARD   # ZeRO-3 风格，完全分片
      cpu_offload: true               # 优化器状态卸载到 CPU
      backward_prefetch: BACKWARD_PRE # 预取优化
      forward_prefetch: true
      limit_all_gathers: true         # 限制 all-gather 显存

      # 混合精度
      mixed_precision:
        param_dtype: bfloat16
        reduce_dtype: float32
        buffer_dtype: bfloat16

    # 优化器配置
    optim:
      lr: 5e-6                  # 全参用更小学习率
      weight_decay: 0.01
      lr_scheduler: cosine
      warmup_ratio: 0.03
      min_lr_ratio: 0.1

  # ---------- Rollout 推理配置 ----------
  rollout:
    name: sglang                # 推荐 SGLang，也可用 vllm
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.25  # 给 rollout 25% 显存
    mem_fraction_static: 0.8

    # GRPO 采样配置
    n: 4                        # 每 prompt 生成 4 个响应
    temperature: 0.7
    top_p: 0.95
    max_new_tokens: 4000

    # 并发控制
    max_num_seqs: 4             # 限制并发序列数

  # ---------- Reference Policy ----------
  ref:
    enable: false               # 全参训练需要单独的 ref policy
    # 如果需要 KL 惩罚，设置为 true 并配置 ref 模型

# ============================================================
#                      算法配置
# ============================================================
algorithm:
  adv_estimator: grpo           # 使用 GRPO 优势估计
  norm_adv_by_std_in_grpo: true # 标准化优势 (原始 GRPO)
  gamma: 1.0
  lam: 1.0
  use_kl_in_reward: false       # GRPO 通常不需要 KL 惩罚

# ============================================================
#                   Critic 配置 (GRPO 不需要)
# ============================================================
critic:
  enable: false

# ============================================================
#                    统一奖励模型配置
# ============================================================
reward_model:
  enable: true
  use_reward_loop: false

  # 使用统一奖励模型 (关键!)
  custom_reward_function:
    path: src/reward/unified_reward_model.py
    name: compute_unified_reward
    reward_kwargs:
      # 单个 1.7B 模型同时计算 R_cog 和 R_logic
      reward_model_path: ./models/Qwen2.5-1.7B-Instruct
      lambda_logic: 0.5         # R_logic 权重
      use_logic_reward: true    # 启用逻辑奖励

reward_manager:
  source: register
  name: naive

# ============================================================
#                      日志配置
# ============================================================
logger:
  - wandb

wandb:
  project: cot_rewriter_grpo
  tags: ["7b-fullparam", "grpo", "unified-reward"]
```

### 3.2 启动脚本

```bash
#!/bin/bash
# scripts/run_optimal_training.sh

set -e

# ============ 环境配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0

# NCCL 配置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# PyTorch 配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============ 启动训练 ============
echo "=========================================="
echo "CoT Rewriter GRPO Training"
echo "Config: 7B 全参 + 1.7B 统一奖励模型"
echo "Hardware: 4×A100-80G"
echo "=========================================="

python scripts/train.py \
    --config-name=optimal_7b_fullparam_grpo \
    trainer.total_epochs=3 \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=4

echo "Training completed!"
```

## 4. 统一奖励模型实现

### 4.1 核心代码

```python
# src/reward/unified_reward_model.py

"""
统一奖励模型

使用单个 1.7B 模型同时计算：
1. 认知奖励 R_cog = log P(Answer | Question, CoT)
2. 逻辑奖励 R_logic = 1 if FSF(rewrite) < FSF(original) else 0

优势：
- 显存节省 3.4 GB (相比分开部署两个 1.7B)
- 无网络延迟 (相比 API 调用)
- 统一管理，简化部署
"""

class UnifiedRewardModel:
    """统一奖励模型"""

    FSF_PROMPT = """分析以下推理过程的失败分支占比。
失败分支：尝试后放弃的方法。
输出 JSON: {{"fsf": 0.0-1.0}}

推理过程:
{reasoning}

JSON:"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",      # 固定在 GPU 0
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        # 加载单个模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device},  # 固定设备
        )
        self.model.eval()

        # 冻结
        for p in self.model.parameters():
            p.requires_grad = False

    def compute_cognitive_reward(self, questions, cots, answers):
        """计算 R_cog = log P(Answer | Question, CoT)"""
        # ... 实现见完整代码 ...

    def compute_fsf(self, reasonings):
        """计算 FSF (使用同一模型)"""
        # ... 实现见完整代码 ...

    def compute_logic_reward(self, original_cots, rewrite_cots):
        """计算 R_logic"""
        orig_fsfs = self.compute_fsf(original_cots)
        rewrite_fsfs = self.compute_fsf(rewrite_cots)
        return [1.0 if r < o else 0.0 for o, r in zip(orig_fsfs, rewrite_fsfs)]

    def __call__(self, questions, original_cots, rewrite_cots, answers, lambda_logic=0.5):
        """计算组合奖励 R = R_cog + λ * R_logic"""
        r_cog = self.compute_cognitive_reward(questions, rewrite_cots, answers)
        r_logic = self.compute_logic_reward(original_cots, rewrite_cots)
        return r_cog + lambda_logic * torch.tensor(r_logic, device=self.device)
```

## 5. 关键优化技术

### 5.1 显存优化清单

| 技术 | 作用 | 配置 |
|-----|------|------|
| **FSDP FULL_SHARD** | 分片权重和梯度到 4 卡 | `sharding_strategy: FULL_SHARD` |
| **CPU Offload** | 优化器状态卸载到 CPU | `cpu_offload: true` |
| **梯度检查点** | 减少激活值显存 | `gradient_checkpointing: true` |
| **Hybrid Engine** | Actor 和 Rollout 共享权重 | `hybrid_engine: true` |
| **统一奖励模型** | 合并 Student 和 FSF 模型 | 单个 1.7B 模型 |
| **序列长度平衡** | 动态调整 batch size | `use_seqlen_balancing: true` |

### 5.2 性能优化清单

| 技术 | 作用 | 配置 |
|-----|------|------|
| **SGLang** | 比 vLLM 更快的推理 | `rollout.name: sglang` |
| **Backward Prefetch** | 预取梯度 | `backward_prefetch: BACKWARD_PRE` |
| **Forward Prefetch** | 预取前向参数 | `forward_prefetch: true` |
| **Mixed Precision** | bf16 训练 | `param_dtype: bfloat16` |

## 6. 监控指标

### 6.1 关键指标

```python
# 训练过程中关注的指标
metrics = {
    # 奖励相关
    "reward/cognitive_mean": "认知奖励均值 (应该逐渐上升)",
    "reward/logic_mean": "逻辑奖励均值 (0-1，越高越好)",
    "reward/total_mean": "总奖励均值",

    # GRPO 相关
    "grpo/advantage_std": "优势标准差 (不应为 0)",

    # 系统相关
    "system/gpu_memory_used": "GPU 显存使用",
    "system/throughput_tokens_per_sec": "吞吐量",

    # 训练相关
    "actor/loss": "策略损失",
    "actor/grad_norm": "梯度范数 (监控是否爆炸)",
}
```

### 6.2 显存监控

```python
# 在训练中监控显存
import torch

def log_memory_usage():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
```

## 7. 常见问题

### 7.1 OOM (显存不足)

**症状**: `CUDA out of memory`

**解决方案**:
```yaml
# 1. 减少 rollout.n
actor_rollout_ref:
  rollout:
    n: 2  # 从 4 减到 2

# 2. 减少 batch size
data:
  train_batch_size: 4  # 从 8 减到 4

# 3. 减少序列长度
data:
  max_prompt_length: 3000
  max_response_length: 3000
```

### 7.2 训练不稳定

**症状**: Loss 震荡或 NaN

**解决方案**:
```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 2e-6        # 降低学习率
    grad_clip: 0.5    # 更激进的梯度裁剪
```

### 7.3 CPU Offload 太慢

**症状**: 每个 step 时间过长

**解决方案**:
```yaml
# 使用 HSDP (混合分片) 替代完全 CPU offload
actor_rollout_ref:
  actor:
    fsdp_config:
      sharding_strategy: HYBRID_SHARD  # 节点内 FSDP，节点间 DDP
      cpu_offload: false
```

## 8. 训练流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRPO 训练单个 Step 流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 数据加载                                                                  │
│     └─→ DataLoader 返回 batch (8 prompts)                                   │
│                                                                              │
│  2. Rollout 生成                                                             │
│     └─→ SGLang 生成 8×4=32 个响应 (KV Cache ~20GB/卡)                        │
│                                                                              │
│  3. 奖励计算 (GPU 0)                                                         │
│     ├─→ R_cog: 1.7B 模型计算 log P(answer|q,cot)                            │
│     └─→ R_logic: 同一模型分析 FSF                                            │
│                                                                              │
│  4. GRPO 优势计算                                                            │
│     └─→ advantage = (R - group_mean) / group_std                            │
│                                                                              │
│  5. Actor 更新 (FSDP)                                                        │
│     ├─→ Forward: all-gather 权重 → 前向                                      │
│     ├─→ Backward: 计算梯度 (梯度检查点重计算激活)                              │
│     └─→ Optimizer: CPU 加载优化器状态 → 更新 → 写回 CPU                       │
│                                                                              │
│  6. 权重同步                                                                  │
│     └─→ 更新后的 Actor 权重同步到 Rollout Engine                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 9. 总结

本方案的核心优化：

1. **FSDP + CPU Offload**: 7B 全参训练，优化器状态放 CPU
2. **统一奖励模型**: 单个 1.7B 同时做 R_cog 和 R_logic
3. **Hybrid Engine**: Actor 和 Rollout 共享显存
4. **梯度检查点**: 用计算换显存

**最终效果**:
- 4×A100-80G 可稳定运行
- 显存利用率 ~54%
- 有足够余量应对峰值
