# KL Reward 使用说明

## 概述

KL reward 是一种基于 Group-wise 自适应归一化的奖励函数，用于检测 CoT (Chain-of-Thought) 改写中的信息泄漏或跳步现象。

## 核心公式

### 1. 逐 token KL 计算

```
kl_t = max(0, log π_actor(cot_t | Q, original_cot, cot_<t) - log π_student(cot_t | Q, cot_<t))
```

- `log π_actor(cot_t)`：Actor 对第 t 个 CoT token 的 log prob（prompt 包含 original_cot → "知情"）
- `log π_student(cot_t)`：Student 对同一 token 的 log prob（prompt 只含 question → "不知情"）

### 2. Group 聚合

```
r_KL = avg(kl_t) + α_kl · max(kl_t)
```

- α_kl = 0.01（默认值）
- r_KL ≥ 0，越大表示推理越不自洽

### 3. Group-wise 自适应归一化

对于第 g 个 group（对应一个 question，包含 G 个 samples）：

```
r_min = min(r_KL[0], ..., r_KL[G-1])
r_max = max(r_KL[0], ..., r_KL[G-1])
r_mean = mean(r_KL[0], ..., r_KL[G-1])

τ = max(r_max - r_min, ε) + β · r_mean

R_cog = exp(-r_KL / τ)
```

- β：平衡因子（默认 0.5）
- ε：防止除零（默认 1e-6）

## 配置方法

在 YAML 配置文件中添加以下参数：

```yaml
custom_reward_function:
  reward_kwargs:
    # 启用 KL reward 模式
    reward_mode: "kl"
    
    # KL reward 参数
    alpha_kl: 0.01              # max(kl_t) 的权重系数
    kl_adaptive_beta: 0.5       # τ 计算中 range 与 mean 的平衡因子
    kl_eps: 1e-6                # 防止除零的小量
    kl_fallback_tau: 3.0        # group_size=1 时的 fallback tau 值
    
    # Student Model 配置（用于计算 log probs）
    student_model_path: /path/to/student/model
    student_tensor_parallel_size: 2
    student_max_model_len: 12000
    
    # 其他奖励配置（可选）
    use_logic_reward: true      # 是否启用 FSF 逻辑奖励
    use_correct_judge: true     # 是否启用正确性判断
```

## 执行流程调整

启用 KL reward 后，训练流程调整为：

1. **Rollout** → 产生 responses
2. **LogProb** → 计算 Actor 的 old_log_probs
3. **Reward** → 计算奖励（读取 old_log_probs，计算 Student log probs，计算 KL reward）
4. **Group-wise Normalization** → 在 ray_trainer.py 中进行 group-wise 归一化

## 监控指标

启用 KL reward 后，可以在日志中看到以下指标：

```
[KL-v2] r_kl: min=0.12, mean=2.34, max=8.56 → r_cog_01: min=0.09, mean=0.42, max=0.89
```

SwanLab 中会记录以下指标：
- `reward/r_kl_min`：原始 KL reward 最小值
- `reward/r_kl_max`：原始 KL reward 最大值
- `reward/r_kl_mean`：原始 KL reward 平均值
- `reward/r_cog_01_min`：归一化后 reward 最小值
- `reward/r_cog_01_max`：归一化后 reward 最大值
- `reward/r_cog_01_mean`：归一化后 reward 平均值

## 参数调优

### β (kl_adaptive_beta) 的选择

| β 值 | 效果 |
|------|------|
| 0.0 | 纯动态范围，严格按相对差异归一化 |
| 0.5 | **推荐默认值**，平衡 range 和 mean |
| 1.0 | 更强调绝对值，难题整体分数更高 |

### α_kl 的选择

| α_kl | 效果 |
|------|------|
| 0.0 | 只使用 avg(kl_t)，忽略 max(kl_t) |
| 0.01 | **推荐默认值**，平衡平均和最大 KL |
| 0.1 | 更强调最大 KL，对异常值更敏感 |

## 调试检查清单

- [ ] `reward/r_kl_mean` 应随训练逐渐下降
- [ ] `reward/r_cog_01_mean` 应逐步上升（接近 1）
- [ ] Group-wise 归一化后的 reward 应在 [0, 1] 区间

## 与其他机制的整合

### 与 Tiered Reward 整合

```python
if R_cog >= threshold_high:
    return 1.0
elif R_cog >= threshold_low:
    return R_cog
else:
    return 0.0
```

### 与 Penalties 整合

```python
final_reward = R_cog - format_penalty - similarity_penalty
```

### 与 Correct Judge 整合

```python
if not is_correct:
    final_reward = correct_judge_penalty_value
```

## 注意事项

1. **Student Model 配置**：KL reward 需要 Student Model 计算 log probs，请确保 Student Model 路径正确且可加载。

2. **显存需求**：计算 KL reward 需要同时加载 Actor 和 Student Model，请确保 GPU 显存充足。

3. **Group 大小**：Group-wise 归一化依赖于 group_size（通常是 `actor_rollout_ref.rollout.n`），请确保 batch_size 可被 group_size 整除。

4. **Fallback 处理**：当 group_size=1 时，使用 `kl_fallback_tau` 作为 tau 值。
