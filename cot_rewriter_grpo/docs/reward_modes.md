# Reward Modes 详解

本文档详细说明了 CoT Rewriter GRPO 项目中所有可用的 `reward_mode` 及其计算公式。

---

## 基础定义

| 符号 | 定义 |
|------|------|
| `Q` | Question (问题) |
| `CoT` | Chain-of-Thought (推理过程) |
| `A` | Answer (答案) |
| `mean_log_prob` | Student Model 对 answer 部分的平均 log probability |
| `avg_prob` | `exp(mean_log_prob)` |
| `PPL` | `exp(-mean_log_prob) = 1 / avg_prob` |
| `answer_log_prob` | `mean(token_log_probs) + alpha * min(token_log_probs)` |

---

## 所有 Reward Modes

### 1. `ppl` 模式

**公式**:
```
R_cog = avg_prob(rewrite) = 1 / PPL(rewrite)
```

**特点**:
- 绝对质量度量，不依赖 original
- 值域: (0, 1]，avg_prob 越高越好
- 同时计算相对改进率作为观测值: `avg_prob(rewrite)/avg_prob(original) - 1`

**适用场景**: 没有 original CoT 的 baseline 场景

---

### 2. `ppl_delta` 模式

**公式**:
```
delta = avg_prob(rewrite) - avg_prob(original)
R_cog = max(0, delta)  # 负值截断
```

**特点**:
- 值域: [0, +∞)
- 只做负值截断，不做归一化
- 绝对奖励值，跨组可比
- delta <= 0 时 reward = 0

**适用场景**: 需要绝对奖励值的训练

---

### 3. `ppl_delta_norm_sigmoid` 模式

**公式**:
```
步骤 1: delta = avg_prob(rewrite) - avg_prob(original)

步骤 2: clamped_delta = max(0, delta)  # 负值截断

步骤 3: normalized = (clamped_delta - mean(clamped_deltas)) / std(clamped_deltas)

步骤 4: R_cog = sigmoid(normalized / temperature)
              = 1 / (1 + exp(-normalized / temperature))
```

**参数**:
- `ppl_delta_norm_sigmoid_temperature`: sigmoid 温度，默认 1.0

**特点**:
- 值域: [0, 1]，0.5 表示平均表现
- 组内归一化：只衡量相对排名，不跨组可比
- 负值截断：rewrite 比 original 差时 reward 接近 0

**适用场景**: GRPO 训练，需要组内相对排序

---

### 4. `ppl_ratio` 模式

**公式**:
```
ppl_rewrite = exp(-rewrite_mean_log_prob)
ppl_original = exp(-original_mean_log_prob)

R_cog = min(ppl_original / ppl_rewrite, ceiling)
      = min(avg_prob(rewrite) / avg_prob(original), ceiling)
```

**参数**:
- `ppl_ratio_ceiling`: 天花板值，默认 0.95

**特点**:
- 值域: [0, ceiling]
- 相对改进比例，有上界保护
- ratio > 1 表示 rewrite 比 original 好

**适用场景**: 需要限制最大奖励值的情况

---

### 5. `cot_effect_ratio` 模式

**公式**:
```
# 定义 CoT Effect: CoT 相对于没有 CoT 的帮助程度
cot_effect(CoT) = avg_prob(Q, CoT, A) - avg_prob(Q, A)

cot_effect_original = avg_prob(Q, original_CoT, A) - avg_prob(Q, A)
cot_effect_rewrite  = avg_prob(Q, rewrite_CoT, A)  - avg_prob(Q, A)

if cot_effect_original < 1e-10:
    R_cog = ceiling  # original CoT 没有帮助，给最大奖励
else:
    R_cog = min(cot_effect_rewrite / cot_effect_original, ceiling)
```

**参数**:
- `cot_effect_ratio_ceiling`: 天花板值，默认 0.95

**特点**:
- 值域: [0, ceiling]
- 衡量 rewrite 相对于 original 的有用程度
- 需要 Student Model 计算额外的 baseline (Q, A)

**适用场景**: 关注 CoT 对答案预测的实际贡献

---

### 6. `cot_effect_delta_sigmoid` 模式

**公式**:
```
# CoT Effect 定义
cot_effect_original = avg_prob(Q, original_CoT, A) - avg_prob(Q, A)

# 改进量
delta = avg_prob(Q, rewrite_CoT, A) - avg_prob(Q, original_CoT, A)
delta_clamped = max(0, delta)  # 负值截断

# 归一化
normalized = delta_clamped / (|cot_effect_original| + epsilon)

# Sigmoid
R_cog = sigmoid(normalized / temperature)
      = 1 / (1 + exp(-normalized / temperature))
```

**参数**:
- `cot_effect_epsilon`: 避免除零的小常数，默认 0.01
- `cot_effect_temperature`: sigmoid 温度，默认 2.0

**特点**:
- 值域: [0.5, 1)（delta_clamped >= 0 时）
- 相对于 original CoT 的贡献进行归一化
- 没有 original 时默认 0.5

**适用场景**: 需要 sigmoid 映射且关注 CoT effect 的场景

---

### 7. `logprob` 模式

**公式**:
```
answer_log_prob = mean(token_log_probs) + alpha * min(token_log_probs)

if use_reward_delta = True:
    R_cog = answer_log_prob(Q, rewrite_CoT, A) - answer_log_prob(Q, original_CoT, A)
else:
    R_cog = answer_log_prob(Q, rewrite_CoT, A)  # 绝对值
```

**参数**:
- `alpha`: min-log-prob 权重，默认 0.01
- `use_reward_delta`: 是否计算 delta，默认 True

**特点**:
- 考虑最差 token 的惩罚（alpha * min 项）
- 值域: (-∞, +∞) 或 (-∞, 0)
- 更敏感到异常糟糕的 token

**适用场景**: 需要惩罚极端差的情况

---

## 对比总结

| 模式 | 值域 | 相对/绝对 | 需要 baseline | 归一化 |
|------|------|----------|--------------|--------|
| `ppl` | (0, 1] | 绝对 | 否 | 否 |
| `ppl_delta` | [0, +∞) | 相对 | 否 | 否 |
| `ppl_delta_norm_sigmoid` | [0, 1] | 相对 (组内) | 否 | 是 |
| `ppl_ratio` | [0, ceiling] | 相对 | 否 | 否 |
| `cot_effect_ratio` | [0, ceiling] | 相对 | 是 | 否 |
| `cot_effect_delta_sigmoid` | [0.5, 1) | 相对 | 是 | 是 |
| `logprob` | (-∞, +∞) | 相对/绝对 | 否 | 否 |

---

## 配置示例

```yaml
custom_reward_function:
  reward_kwargs:
    reward_mode: ppl_delta  # 选择模式

    # ppl_delta 参数 (无额外参数)

    # ppl_delta_norm_sigmoid 参数
    ppl_delta_temperature: 1.0

    # ppl_ratio 参数
    ppl_ratio_ceiling: 0.95

    # cot_effect_ratio 参数
    cot_effect_ratio_ceiling: 0.95

    # cot_effect_delta_sigmoid 参数
    cot_effect_epsilon: 0.01
    cot_effect_temperature: 2.0

    # logprob 参数
    alpha: 0.01
    use_reward_delta: true
```

---

## 代码位置

- 主要实现: `verl/verl/experimental/reward/cot_rewriter/student_reward.py`
- vLLM 版本: 行 699-949
- SGLang 版本: `verl/verl/experimental/reward/cot_rewriter/sglang_student_reward.py`
- 入口: `verl/verl/experimental/reward/cot_rewriter/combined_reward.py`
