# reward_mode: "kl" — Group-wise 自适应归一化 KL Reward

## 1. 设计目标

实现一个物理意义清晰、GRPO 友好的 KL reward：
1. **物理意义清晰**：$r_{KL}$ 表示 Student 面对 CoT 的"惊讶程度"（信息泄漏/跳步的量化）
2. **自适应归一化**：每个 group 独立归一化，适配不同难度的问题
3. **输出有界**：最终 reward 严格在 $[0,1]$ 区间，便于 GRPO 优化

---

## 2. 核心公式

### 2.1 逐 token KL 计算

```
kl_t = max(0, log π_actor(cot_t | Q, original_cot, cot_<t) - log π_student(cot_t | Q, cot_<t))
```

- `log π_actor(cot_t)`：Actor 对第 t 个 CoT token 的 log prob（prompt 包含 original_cot → "知情"）
- `log π_student(cot_t)`：Student 对同一 token 的 log prob（prompt 只含 question → "不知情"）

### 2.2 Group 聚合

```
r_KL = avg(kl_t) + α_kl · max(kl_t)
```

- α_kl = 0.01（默认值）
- $r_{KL} \geq 0$，越大表示推理越不自洽

### 2.3 Group-wise 自适应归一化

对于第 $g$ 个 group（对应一个 question，包含 $G$ 个 samples）：

```
r_min = min(r_KL[0], ..., r_KL[G-1])
r_max = max(r_KL[0], ..., r_KL[G-1])
r_mean = mean(r_KL[0], ..., r_KL[G-1])

τ = max(r_max - r_min, ε) + β · r_mean

R_cog = exp(-r_KL / τ)
```

- $\beta$：平衡因子（默认 0.5）
- $\varepsilon$：防止除零（默认 1e-6）

---

## 3. 物理意义

| 场景 | Group 特征 | τ 行为 | 映射效果 |
|------|-----------|--------|---------|
| **简单问题** | $r_{KL}$ 整体较低且集中 | τ 较小 | 细微差异被放大 |
| **困难问题** | $r_{KL}$ 整体较高且分散 | τ 较大 | 整体压缩，避免过度惩罚 |
| **一致输出** | $r_{max} \approx r_{min}$ | τ ≈ β·r_mean | 以 mean 为尺度归一化 |

**核心思想**：每个 question 自成一个"竞赛场"，reward 反映在该问题上的相对表现。

---

## 4. 算法实现

```python
import torch

def compute_group_wise_kl_reward(r_kl_matrix, beta=0.5, eps=1e-6):
    """
    Args:
        r_kl_matrix: [num_groups, group_size] 原始 r_KL 值
        beta: τ 计算中 range 与 mean 的平衡因子
        eps: 防止除零的小量
        
    Returns:
        r_cog_01: [num_groups, group_size] 归一化后的 [0,1] reward
    """
    num_groups, group_size = r_kl_matrix.shape
    r_cog_01 = torch.zeros_like(r_kl_matrix)
    
    for g in range(num_groups):
        r_group = r_kl_matrix[g]
        
        # Group 统计量
        r_min = r_group.min()
        r_max = r_group.max()
        r_mean = r_group.mean()
        
        # 自适应 tau
        tau = max(r_max - r_min, eps) + beta * r_mean
        
        # 归一化
        r_cog_01[g] = torch.exp(-r_group / tau)
    
    return r_cog_01
```

---

## 5. 与 GRPO 整合

```python
# verl/trainer/ppo/ray_trainer.py

# 1. 计算原始 KL reward [batch_size * group_size]
r_kl_raw = compute_kl_reward(batch, alpha_kl=0.01)

# 2. Reshape 为 [num_questions, group_size]
r_kl_grouped = r_kl_raw.view(-1, config.group_size)

# 3. Group-wise 归一化到 [0,1]
r_cog_01 = compute_group_wise_kl_reward(r_kl_grouped, beta=0.5)

# 4. 后续处理（penalties 等）
final_rewards = apply_penalties(r_cog_01.view(-1), ...)

# 5. 计算 advantage（在 group 内部）
advantages = compute_group_wise_advantage(final_rewards, group_size)
```

### 配置参数

```yaml
custom_reward_function:
  reward_kwargs:
    reward_mode: kl
    alpha_kl: 0.01
    kl_adaptive_beta: 0.5       # τ = range + β·mean 中的 β
```

### 边界情况

| 情况 | 处理 |
|------|------|
| `group_size = 1` | 使用 fallback_tau（如 3.0） |
| `r_max == r_min` | τ = ε + β·r_mean |

---

## 6. 监控指标

### 6.1 原始 r_KL 统计

```python
metrics = {
    "reward/r_kl_min": r_kl_raw.min().item(),
    "reward/r_kl_max": r_kl_raw.max().item(),
    "reward/r_kl_mean": r_kl_raw.mean().item(),
}
```

### 6.2 归一化后 R_cog 统计

```python
metrics = {
    "reward/r_cog_01_min": r_cog_01.min().item(),
    "reward/r_cog_01_max": r_cog_01.max().item(),
    "reward/r_cog_01_mean": r_cog_01.mean().item(),
}
```

### 6.3 调试日志

```
[KL-v2] r_kl: min=0.12, mean=2.34, max=8.56 → r_cog_01: min=0.09, mean=0.42, max=0.89
```

---

## 7. 参数调优

### β 的选择

| β 值 | 效果 |
|------|------|
| 0.0 | 纯动态范围，严格按相对差异归一化 |
| 0.5 | **推荐默认值**，平衡 range 和 mean |
| 1.0 | 更强调绝对值，难题整体分数更高 |

### 调试检查清单

- [ ] `reward/r_kl_mean` 应随训练逐渐下降
- [ ] `reward/r_cog_01_mean` 应逐步上升（接近 1）

---

## 8. 文件修改

| 文件 | 修改内容 |
|------|---------|
| `verl/verl/experimental/reward/cot_rewriter/combined_reward.py` | 新增 `compute_group_wise_kl_reward()` |
| `verl/verl/trainer/ppo/ray_trainer.py` | 调用 group-wise 归一化，传递 group_size |

---

## 9. 与其他机制整合

### 9.1 Tiered Reward

```python
if R_cog >= threshold_high:
    return 1.0
elif R_cog >= threshold_low:
    return R_cog
else:
    return 0.0
```

### 9.2 Penalties

```python
final_reward = R_cog - format_penalty - similarity_penalty
```

### 9.3 Correct Judge

```python
if not is_correct:
    final_reward = correct_judge_penalty_value
```

---

## 10. 数据来源与执行顺序

### 10.1 Actor CoT log probs

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
# shape: [batch_size, response_length]
```

### 10.2 Student CoT log probs

```python
messages = [
    {'role': 'user', 'content': question},
    {'role': 'assistant', 'content': '', 'reasoning_content': cot}
]
```

### 10.3 执行顺序调整

将 `old_log_prob` 计算移到 `reward` 之前：

```
1. Rollout → 产生 responses
2. LogProb → 计算 old_log_probs
3. Reward  → 计算奖励（读取 old_log_probs）
```

修改位置：`verl/verl/trainer/ppo/ray_trainer.py` L1366-L1453 附近。
