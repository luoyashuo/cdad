# Student Model 作为 Reward Model 集成指南

## 概述

将 Student Model (Qwen2.5-3B + FSF LoRA) 集成到 VERL 中作为 Reward Model，
同时不影响 Policy Model (Qwen2.5-7B LoRA) 的 Rollout 性能。

## 架构对比

### 方案 A: 时分复用 (推荐用于单卡/显存紧张场景)

```
┌───────────────────────────────────────────────────────────────┐
│                         GPU Memory                            │
├───────────────────────────────────────────────────────────────┤
│ Phase 1: Rollout                                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Policy Model (7B LoRA) + vLLM KV Cache                  │  │
│  │ gpu_memory_utilization: 0.85                             │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│ Phase 2: Reward Computation (Rollout 完成后)                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Policy Model (7B LoRA) [保持]                            │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │ Student Model (3B + FSF LoRA) [临时加载]                 │  │
│  │ vLLM KV Cache [已释放]                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│ Phase 3: Actor Update (Reward 计算完成后)                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Policy Model (7B LoRA) + Optimizer States               │  │
│  │ Student Model [offload 到 CPU 或释放]                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

**文件**: `src/reward/student_reward_worker.py`

**配置**: `configs/cot_rewriter_grpo_fsf_lora.yaml`

**优点**:
- 只需要 1 张 GPU
- 显存利用率高
- 适合资源受限场景

**缺点**:
- Rollout 和 Reward 串行执行
- 频繁加载/卸载有一定开销

### 方案 B: 独立 GPU (推荐用于多卡场景)

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0-2: Policy Model + Rollout                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Policy Model (7B LoRA)                                    │  │
│  │ vLLM/SGLang Rollout Engine (KV Cache)                     │  │
│  │ gpu_memory_utilization: 0.5 per GPU                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ GPU 3: Student Reward Model (独立)                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Student Model (Qwen2.5-3B) + FSF LoRA Adapter             │  │
│  │ 常驻显存，无需加载/卸载                                      │  │
│  │                                                           │  │
│  │ 功能:                                                     │  │
│  │  - 认知奖励: 使用基座模型                                   │  │
│  │  - FSF 奖励: 使用基座 + LoRA                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**文件**: `src/reward/ray_student_reward.py`

**配置**: `configs/cot_rewriter_grpo_ray_reward.yaml`

**优点**:
- Rollout 和 Reward 完全隔离
- 可并行执行
- 更稳定，无 OOM 风险

**缺点**:
- 需要额外 GPU
- Ray Actor 通信有少量开销

## 使用指南

### 步骤 1: 训练 FSF LoRA Adapter

```bash
cd verl/tutorial/03_CoT_Rewrite_GRPO

# 准备训练数据 (JSONL 格式)
# {"reasoning": "推理过程...", "fsf_label": {"total_steps": 10, "failed_steps": 3, "fsf": 0.3}}

# 训练 FSF LoRA
bash scripts/run_train_fsf_lora.sh
```

### 步骤 2: 选择集成方案

#### 方案 A: 时分复用 (1-2 GPU)

```yaml
# configs/cot_rewriter_grpo_fsf_lora.yaml

reward_model:
  enable: true
  custom_reward_function:
    path: src/reward/student_reward_worker.py
    name: compute_student_reward
    reward_kwargs:
      student_model_path: ./models/Qwen2.5-3B-Instruct
      fsf_lora_path: ./outputs/fsf_lora
      offload_to_cpu: true  # 计算完成后释放显存
```

```bash
python scripts/train.py --config-name=cot_rewriter_grpo_fsf_lora
```

#### 方案 B: 独立 GPU (4+ GPU)

```yaml
# configs/cot_rewriter_grpo_ray_reward.yaml

reward_model:
  enable: true
  enable_resource_pool: true  # 启用独立资源池
  n_gpus_per_node: 1          # Reward Model 使用 1 个 GPU

  custom_reward_function:
    path: src/reward/ray_student_reward.py
    name: compute_ray_student_reward
    reward_kwargs:
      student_model_path: ./models/Qwen2.5-3B-Instruct
      fsf_lora_path: ./outputs/fsf_lora
```

```bash
python scripts/train.py --config-name=cot_rewriter_grpo_ray_reward
```

## 奖励计算流程

```
Input: batch from Rollout
  ├── prompts: [Question + Instruction]
  ├── responses: [Rewrite CoT]
  └── non_tensor_batch:
      ├── ground_truth: [Answer]
      └── original_cot: [Original CoT]

                    │
                    ▼
┌───────────────────────────────────────────────────┐
│ Student Reward Worker                             │
│                                                   │
│ 1. 认知奖励 (使用原始 Student Model)              │
│    R_cog = log P(Answer | Question, Rewrite_CoT)  │
│                                                   │
│ 2. FSF 奖励 (使用 Student Model + FSF LoRA)       │
│    FSF_orig = analyze(Original_CoT)               │
│    FSF_rewrite = analyze(Rewrite_CoT)             │
│    R_logic = 1 if FSF_rewrite < FSF_orig else 0   │
│                                                   │
│ 3. 组合奖励                                        │
│    R_total = R_cog + λ * R_logic                  │
└───────────────────────────────────────────────────┘
                    │
                    ▼
Output: token_level_rewards
  shape: [batch_size, response_length]
  value: R_total at last token position
```

## 显存估算

| 组件 | 显存 (FP16/BF16) | 备注 |
|------|-----------------|------|
| Policy Model (7B) | ~14 GB | 基座参数 |
| LoRA Adapter | ~0.2 GB | rank=64 |
| vLLM KV Cache | ~8-20 GB | 取决于 batch size |
| Student Model (3B) | ~6 GB | 基座参数 |
| FSF LoRA | ~0.1 GB | rank=64 |
| **方案 A 峰值** | ~30 GB | 单卡，时分复用 |
| **方案 B 分布** | ~24 GB + 7 GB | 3卡 + 1卡 |

## 最佳实践

1. **显存紧张**: 使用方案 A，开启 `offload_to_cpu=True`

2. **追求性能**: 使用方案 B，独立 GPU

3. **FSF LoRA 训练数据**: 建议使用 1000+ 样本

4. **lambda_logic 调参**:
   - 0.5: 平衡认知和逻辑奖励
   - 0.3: 偏重认知奖励（让 CoT 更易理解）
   - 0.7: 偏重逻辑奖励（减少失败分支）

5. **调试建议**:
   ```python
   # 在 reward_kwargs 中添加
   reward_kwargs:
     ...
     debug: true  # 打印详细信息
   ```
