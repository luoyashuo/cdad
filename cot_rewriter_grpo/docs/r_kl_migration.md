# RLT r_KL 迁移方案（方案 B）：完整 Actor-Student KL + Sigmoid 归一化

## 1. 问题背景

当前框架中 R_cog 几乎不上升。本文给出将 RLT 的 r_KL 迁移到我们框架的完整方案，采用 RLT 的 α 机制（avg + α·extreme），并通过 sigmoid 将最终 reward 归一化到 [0, 1]。

---

## 2. 完整数学公式

### 2.1 r_SS：Student Success Reward（answer 级，对标 RLT 的 r_SS）

衡量改写后的 CoT 是否帮助 Student 更好地预测 answer。

**单条 r_SS 计算**（带 α 机制）：

```
r_SS(CoT) = avg{log π_s(answer_t | Q, CoT, answer_<t)}
          + α · min{log π_s(answer_t | Q, CoT, answer_<t)}
```

- `avg` 项：answer token 平均 log prob，衡量整体预测质量
- `α · min` 项：最差 token 的 log prob，惩罚"有一步完全预测不出来"的情况
- α = 0.01（RLT 默认值）

**Delta 版本**（用于 GRPO，衡量改写 vs 原始的提升）：

```
Δr_SS = r_SS(rewrite_CoT) - r_SS(original_CoT)
```

- Δr_SS > 0：改写让 Student 更好地理解 answer → 正奖励
- Δr_SS < 0：改写反而更差 → 负奖励
- 典型取值范围：约 [-0.5, +0.5]

### 2.2 r_KL：Logical Continuation KL Penalty（CoT token 级，对标 RLT 的 r_KL）

衡量改写 CoT 的每个推理步骤是否"自洽"——不需要看过 original_cot 也能理解。

**对应关系**（RLT → 我们）：

| RLT | 我们的框架 |
|-----|-----------|
| π_θ(CoT_token \| Q, **answer**) | π_actor(CoT_token \| Q, **original_cot**, CoT_<t) |
| π_s(CoT_token \| Q) | π_student(CoT_token \| Q, CoT_<t) |
| "内幕信息" = answer | "内幕信息" = original_cot |

**逐 token KL 近似**（PPO/GRPO 标准做法）：

```
kl_t = max(0, log π_actor(cot_t) - log π_student(cot_t))
```

- `log π_actor(cot_t)`：Actor 对第 t 个 CoT token 的 log prob（条件包含 original_cot → "知情"）
- `log π_student(cot_t)`：Student 对同一 token 的 log prob（条件只含 question → "不知情"）
- `max(0, ·)`：KL 散度非负，只惩罚 Actor 比 Student 更自信的 token

**聚合**（带 α 机制）：

```
r_KL = avg(kl_t) + α · max(kl_t)
```

- `avg` 项：整体逻辑连贯性（平均每个 token 的不连贯程度）
- `α · max` 项：最不连贯的那个 token（抓最严重的跳步/信息泄漏）
- α = 0.01（与 r_SS 相同）
- r_KL ≥ 0，越大说明推理越不自洽
- 典型取值范围：约 [0, 10]

**直觉举例**：

| 改写步骤 | Actor log prob | Student log prob | kl_t | 含义 |
|----------|---------------|-----------------|------|------|
| "设 f(x) = x²" | -2.0 | -2.5 | 0.5 | 双方都觉得自然，轻微 KL |
| "由泰勒展开得 sin(x)≈x" | -3.0 | -5.0 | 2.0 | Actor 见过原始推理知道要用泰勒展开，Student 觉得突然 |
| "所以答案是 42" | -0.5 | -8.0 | **7.5** | Actor 从 original_cot 直接搬答案，Student 完全无法预测 → **重罚** |

### 2.3 组合公式

**Raw reward**：

```
R_raw = Δr_SS - λ · r_KL
```

- λ = 可调参数（RLT 用 3，我们的场景需要校准，建议从 0.1 开始）
- Δr_SS 鼓励有效改写（answer 更好预测）
- λ · r_KL 惩罚不自洽改写（跳步/信息泄漏）

**如需保留 FSF 和其他 penalty**：

```
R_raw = Δr_SS - λ_KL · r_KL - λ_FSF · r_FSF - penalties
```

### 2.4 Sigmoid 归一化到 [0, 1]

```
R_final = sigmoid(R_raw / τ) = 1 / (1 + exp(-R_raw / τ))
```

- τ（temperature）控制 sigmoid 的"陡峭度"
- 小 τ → reward 集中在 0 或 1（近似二值化）
- 大 τ → reward 集中在 0.5 附近（差异被压缩）
- **合适的 τ** → reward 均匀分布在 [0.1, 0.9] 范围

**τ 的设定方法**：

```
τ ≈ std(R_raw)
```

即 τ 等于 R_raw 的标准差。这样 ±1σ 范围内的 R_raw 映射到 sigmoid 的 [0.27, 0.73]，提供良好区分度。

**标定流程**：

1. 先跑 5-10 个 step，λ_KL=0，只 log R_raw 的统计量
2. 观察 `mean(R_raw)` 和 `std(R_raw)`
3. 设 τ = std(R_raw)
4. 加入 r_KL 后重新标定

**R_final 的含义**：

| R_final | 含义 |
|---------|------|
| → 1 | 改写有效且逻辑自洽 |
| ≈ 0.5 | 改写效果中性 |
| → 0 | 改写无效或逻辑不连贯 |

---

## 3. 架构改动：执行顺序调整

### 3.1 问题

当前 `ray_trainer.py` 执行顺序：

```
1. Rollout: generate_sequences()      → 产生 responses
2. Reward:  compute_reward()           → 计算奖励 ← old_log_probs 此时不存在！
3. LogProb: compute_log_prob()         → 计算 old_log_probs
```

r_KL 需要 Actor 的 per-token log probs（`old_log_probs`），但它在 reward 之后才计算。

### 3.2 解决方案

交换步骤 2 和 3：

```
1. Rollout: generate_sequences()      → 产生 responses
2. LogProb: compute_log_prob()         → 计算 old_log_probs，合并到 batch
3. Reward:  compute_reward()           → 计算奖励（现在可以访问 old_log_probs）
```

**为什么可以安全交换**：

- 两个步骤完全独立：compute_reward 不依赖 old_log_probs（当前），compute_log_prob 不依赖 reward
- 显存方面更优：FSDP Actor 计算完 log probs 后 offload，释放 GPU 给 Student Model
- 代码修改量小：只需调换两个 `with marked_timer(...)` 代码块的顺序

### 3.3 修改 `ray_trainer.py`（L1366-L1450 附近）

**修改前**：

```python
                    # === reward block (L1366-L1417) ===
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        ...

                    # === old_log_prob block (L1419-L1450) ===
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        ...
                        batch = batch.union(old_log_prob)
```

**修改后**：

```python
                    # === old_log_prob block (原 L1419-L1450，移到前面) ===
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        # 现在 batch.batch["old_log_probs"] 可用

                    # === reward block (原 L1366-L1417，移到后面) ===
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # batch 现在包含 old_log_probs，reward 函数可以访问
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        ...
```

---

## 4. Actor CoT Log Probs 提取

### 4.1 数据来源

交换执行顺序后，reward 函数中：

```python
def compute_cot_rewriter_reward(data: DataProto, tokenizer, ...):
    # old_log_probs 现在可用
    old_log_probs = data.batch['old_log_probs']    # shape: [batch_size, response_length]
    response_ids = data.batch['responses']           # shape: [batch_size, response_length]
```

### 4.2 从 response 中定位 CoT token 位置

response 格式为 `<rewrite_reasoning>CoT 内容</rewrite_reasoning>`，需要找到 CoT 内容对应的 token 位置。

```python
def extract_cot_token_range(response_ids_i, tokenizer):
    """
    从 response token IDs 中找到 CoT 内容的 token 起止位置。

    Args:
        response_ids_i: 单条 response 的 token IDs (1D tensor)
        tokenizer: tokenizer

    Returns:
        (cot_start, cot_end): CoT 内容 token 的起止索引（不含标签 token）
    """
    response_text = tokenizer.decode(response_ids_i, skip_special_tokens=False)

    open_tag = '<rewrite_reasoning>'
    close_tag = '</rewrite_reasoning>'

    # 字符级定位
    char_start = response_text.find(open_tag)
    char_end = response_text.find(close_tag)
    if char_start == -1 or char_end == -1:
        return None, None

    cot_char_start = char_start + len(open_tag)
    cot_char_end = char_end

    # 字符位置 → token 位置（使用 offset_mapping）
    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.offset_mapping

    cot_token_start = None
    cot_token_end = None
    for idx, (cs, ce) in enumerate(offsets):
        if cs >= cot_char_start and cot_token_start is None:
            cot_token_start = idx
        if cs >= cot_char_end:
            cot_token_end = idx
            break

    if cot_token_end is None:
        cot_token_end = len(offsets)

    return cot_token_start, cot_token_end
```

### 4.3 提取 Actor CoT log probs

```python
# 对每个样本提取 Actor 在 CoT 部分的 log probs
actor_cot_log_probs_list = []
for i in range(batch_size):
    response_ids_i = response_ids[i]
    cot_start, cot_end = extract_cot_token_range(response_ids_i, tokenizer)

    if cot_start is not None and cot_end is not None:
        actor_cot_lp = old_log_probs[i, cot_start:cot_end].tolist()
    else:
        actor_cot_lp = []

    actor_cot_log_probs_list.append(actor_cot_lp)
```

---

## 5. Student CoT Log Probs 计算

### 5.1 新增 `_build_prompt_cot_only` 方法

在 `sglang_student_reward.py` 的 `SGLangStudentRewardClient` 类中新增：

```python
def _build_prompt_cot_only(self, question: str, cot: str) -> tuple[str, int]:
    """
    构造 Student 评估 CoT 可理解性的 prompt。

    Student 只看 question，对 CoT 每个 token 给出 log prob。
    用于计算 r_KL 中的 π_student(cot_t | Q, cot_<t)。

    Returns:
        (full_prompt_text, cot_start_token_idx)
    """
    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': '', 'reasoning_content': cot}
    ]
    full_text = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # 用 dummy 技巧定位 CoT 起始 token 位置
    dummy_cot = "X"
    messages_with_dummy = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': '', 'reasoning_content': dummy_cot}
    ]
    text_with_dummy = self.tokenizer.apply_chat_template(
        messages_with_dummy, tokenize=False, add_generation_prompt=False
    )
    tokens_with_dummy = self.tokenizer.encode(text_with_dummy, add_special_tokens=False)

    dummy_token_id = self.tokenizer.encode(dummy_cot, add_special_tokens=False)[0]
    cot_start_idx = 0
    for i in range(len(tokens_with_dummy) - 1, -1, -1):
        if tokens_with_dummy[i] == dummy_token_id:
            cot_start_idx = i
            break

    return full_text, cot_start_idx
```

### 5.2 在 `compute_cognitive_reward_batch` 中新增 CoT-only 计算

和现有的 rewrite/original/no_cot 并行，新增第 4 组 prompt：

```python
# Phase 1: 构建 CoT-only prompts
cot_only_prompts = []
cot_only_starts = []
for i in range(batch_size):
    cot_only_prompt, cot_only_start = self._build_prompt_cot_only(
        questions[i], rewrite_cots[i]
    )
    cot_only_prompts.append(cot_only_prompt)
    cot_only_starts.append(cot_only_start)

# Phase 2: 分发到 Ray workers（与 rewrite/original/no_cot 并行）
cot_only_chunks = self._split_into_chunks(cot_only_prompts, self.num_actors)
cot_only_start_chunks = self._split_into_chunks(cot_only_starts, self.num_actors)
cot_only_futures = []
for worker_idx, (pc, sc) in enumerate(zip(cot_only_chunks, cot_only_start_chunks)):
    if pc:
        future = self.workers[worker_idx].compute_logprobs.remote(pc, sc)
        cot_only_futures.append((worker_idx, len(pc), future))

# Phase 3: 收集结果
cot_only_results = [None] * batch_size
for worker_idx, chunk_size, future in cot_only_futures:
    chunk_results = ray.get(future)
    start_idx = sum(len(cot_only_chunks[j]) for j in range(worker_idx))
    for j, result in enumerate(chunk_results):
        cot_only_results[start_idx + j] = result
```

### 5.3 返回 per-token log probs

当前 `compute_logprobs` 已经返回 `token_log_probs`（per-token list），直接使用：

```python
# 提取 Student 的 CoT per-token log probs
student_cot_log_probs_list = []
student_cot_mean_log_probs = []
student_cot_min_log_probs = []

for i in range(batch_size):
    result = cot_only_results[i]
    student_cot_log_probs_list.append(result["token_log_probs"])
    student_cot_mean_log_probs.append(result["mean_log_prob"])
    student_cot_min_log_probs.append(result["min_log_prob"])
```

在返回 dict 中新增：

```python
return {
    # ... 现有字段 ...
    "student_cot_token_log_probs": student_cot_log_probs_list,
    "student_cot_mean_log_probs": student_cot_mean_log_probs,
    "student_cot_min_log_probs": student_cot_min_log_probs,
}
```

---

## 6. r_KL 计算与 Sigmoid 归一化

### 6.1 `combined_reward.py` — `__init__` 新增参数

```python
# ========== RLT-style r_KL 配置 ==========
use_kl_reward: bool = False,        # 是否启用 r_KL（逻辑连贯性惩罚）
lambda_kl: float = 0.1,             # r_KL 权重（RLT 用 3，我们先从 0.1 开始）
alpha_kl: float = 0.01,             # r_KL 中 max 项的权重（RLT 默认 0.01）
# ========== Sigmoid 归一化配置 ==========
use_sigmoid_norm: bool = True,      # 是否启用 sigmoid 归一化到 [0, 1]
sigmoid_tau: float = 1.0,           # sigmoid 温度参数 τ
```

### 6.2 `compute_reward` 方法中的 r_KL 计算

在获取 cog_result 之后、计算 base_rewards 之前插入：

```python
# ========== 计算 r_KL（RLT-style Logical Continuation KL Penalty） ==========
r_kl_values = [0.0] * batch_size
r_kl_avg_values = [0.0] * batch_size    # avg(kl_t) 部分，用于 log
r_kl_max_values = [0.0] * batch_size    # max(kl_t) 部分，用于 log

if self.use_kl_reward:
    # actor_cot_log_probs_list: 从 DataProto 提取（见 §4）
    # student_cot_token_log_probs: 从 cog_result 获取（见 §5）
    student_cot_token_lps = cog_result.get("student_cot_token_log_probs", [[] for _ in range(batch_size)])

    for i in range(batch_size):
        actor_lp = actor_cot_log_probs_list[i]
        student_lp = student_cot_token_lps[i]

        if not actor_lp or not student_lp:
            r_kl_values[i] = 0.0
            continue

        # Token 对齐：取较短长度（边界 token 可能不同）
        min_len = min(len(actor_lp), len(student_lp))
        if min_len == 0:
            r_kl_values[i] = 0.0
            continue

        # 逐 token KL 近似
        kl_per_token = []
        for t in range(min_len):
            kl_t = max(0.0, actor_lp[t] - student_lp[t])
            kl_per_token.append(kl_t)

        # RLT α 聚合: avg + α · max
        kl_avg = sum(kl_per_token) / len(kl_per_token)
        kl_max = max(kl_per_token)
        r_kl = kl_avg + self.alpha_kl * kl_max

        r_kl_values[i] = r_kl
        r_kl_avg_values[i] = kl_avg
        r_kl_max_values[i] = kl_max

        if i < 3:
            logger.info(
                f"[R_KL] Sample {i}: kl_avg={kl_avg:.4f}, kl_max={kl_max:.4f}, "
                f"r_kl={r_kl:.4f}, aligned_tokens={min_len}/{len(actor_lp)}"
            )
```

### 6.3 组合并 Sigmoid 归一化

```python
# ========== 组合 raw reward ==========
# R_raw = Δr_SS - λ_KL · r_KL - λ_FSF · r_FSF - penalties
raw_rewards = []
for i in range(batch_size):
    r_raw = r_cog[i]  # Δr_SS (已经是 delta 模式)
    r_raw -= self.lambda_kl * r_kl_values[i]         # r_KL 惩罚
    r_raw += fsf_scale * r_logic[i]                    # FSF（如果启用）
    r_raw += too_short_penalties_actual[i]              # 过短惩罚（负值）
    r_raw += too_long_penalties_actual[i]               # 过长惩罚（负值）
    r_raw += repeat_penalties_actual[i]                 # 重复惩罚（负值）
    raw_rewards.append(r_raw)

# ========== Sigmoid 归一化到 [0, 1] ==========
import math

if self.use_sigmoid_norm:
    total_rewards = []
    for r_raw in raw_rewards:
        r_norm = 1.0 / (1.0 + math.exp(-r_raw / self.sigmoid_tau))
        total_rewards.append(r_norm)
else:
    total_rewards = raw_rewards
```

---

## 7. YAML 配置

### 7.1 新增配置项

```yaml
custom_reward_function:
  reward_kwargs:
    # ==================== R_cog 基础配置（修复） ====================
    reward_mode: ppl
    ppl_on_full_response: false     # 改为 false！只算 answer 的 PPL
    use_reward_delta: true           # 改为 true！Δr_SS = rewrite - original
    alpha: 0.01                      # r_SS 的 α（avg + α·min）

    # ==================== r_KL 配置（RLT-style，核心新增） ====================
    use_kl_reward: true              # 启用 r_KL
    lambda_kl: 0.1                   # r_KL 权重（建议从 0.1 开始，逐步调到 0.5-3.0）
    alpha_kl: 0.01                   # r_KL 的 α（avg + α·max）

    # ==================== Sigmoid 归一化配置 ====================
    use_sigmoid_norm: true           # 启用 sigmoid 归一化到 [0, 1]
    sigmoid_tau: 1.0                 # τ 温度（≈ R_raw 的标准差，先用 1.0，后续根据 log 校准）
```

### 7.2 完整示例

```yaml
custom_reward_function:
  path: ${oc.env:PROJECT_BASE_DIR}/verl/verl/experimental/reward/cot_rewriter/combined_reward.py
  name: compute_cot_rewriter_reward
  reward_kwargs:
    # Student Model
    student_model_path: /fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B
    student_tensor_parallel_size: 2
    student_max_model_len: 12000
    student_micro_batch_size: 1
    reward_backend: sglang
    sglang_mem_fraction_static: 0.5

    # r_SS (answer-level, 带 α)
    reward_mode: ppl
    ppl_on_full_response: false
    use_reward_delta: true
    alpha: 0.01

    # r_KL (CoT token-level, RLT-style)
    use_kl_reward: true
    lambda_kl: 0.1
    alpha_kl: 0.01

    # Sigmoid 归一化
    use_sigmoid_norm: true
    sigmoid_tau: 1.0

    # FSF (可选，保留或关闭)
    use_logic_reward: false           # 先关掉 FSF，用 r_KL 替代

    # Correct Judge (保留)
    use_correct_judge: true
    correct_judge_penalty_value: 0.0

    # 其他 penalty
    format_penalty: 0.2
    similarity_threshold: 0.5
    similarity_penalty: 0.0
    too_short_max_penalty: 0.0
    too_long_max_penalty: 0.0
    penalty_mode: absolute
```

---

## 8. 标定 & 调参指南

### 8.1 第一阶段：标定 τ

```yaml
# 关闭 r_KL，只跑 r_SS delta，观察 R_raw 的统计量
use_kl_reward: false
use_sigmoid_norm: false   # 先不归一化，看 raw 值
```

跑 10 个 step 后，查看 log：
- `mean(R_raw)` = ?
- `std(R_raw)` = ?
- 设 `sigmoid_tau ≈ std(R_raw)`

### 8.2 第二阶段：加入 r_KL 并标定 λ_KL

```yaml
use_kl_reward: true
lambda_kl: 0.0     # 先设 0，只 log r_KL 不参与奖励
```

跑 10 个 step 后，查看 log：
- `mean(r_KL)` = ?
- `std(r_KL)` = ?
- 设 λ_KL 使 `λ_KL · mean(r_KL) ≈ 0.3 · |mean(Δr_SS)|`

### 8.3 第三阶段：联合调优

```yaml
use_kl_reward: true
lambda_kl: <标定值>
use_sigmoid_norm: true
sigmoid_tau: <标定值>
```

观察指标：
- `R_final` 分布应在 [0.1, 0.9] 范围内均匀分布
- GRPO group 内 advantage 方差 > 0（有区分度）
- R_cog（Δr_SS）应该逐步上升

### 8.4 参数灵敏度

| 参数 | 过小的表现 | 过大的表现 | 推荐范围 |
|------|-----------|-----------|---------|
| λ_KL | r_KL 无效，Actor 仍然跳步 | Actor 只生成简单短句，不敢复杂推理 | 0.05 - 1.0 |
| α | 忽视最差 token，允许个别极端跳步 | 被少数难 token 主导，奖励波动大 | 0.005 - 0.05 |
| τ | reward 二值化（0 或 1），GRPO 丢失细粒度 | reward 全在 0.5 附近，无区分度 | std(R_raw) * 0.5 ~ 2.0 |

---

## 9. Token 对齐说明

### 9.1 为什么可以对齐

- Actor: Llama-3.2-3B-Instruct
- Student: Llama-3.2-1B
- **共享同一个 tokenizer**（Llama-3.2 系列）
- 同一段 CoT 文本经过相同 tokenizer 编码后，token 序列完全一致

### 9.2 边界处理

Actor 的 response 中包含 `<rewrite_reasoning>` 标签，Student 的 prompt 不包含。在标签边界处，BPE 分词可能略有不同。

**处理策略**：取两者的较短长度对齐，丢弃边界处不匹配的 1-2 个 token。对于数百个 CoT token，这 1-2 个 token 的影响可忽略。

### 9.3 对齐失败的兜底

如果某个样本无法对齐（标签缺失、格式错误等），r_KL 设为 0：

```python
if not actor_lp or not student_lp or min_len < 5:
    r_kl_values[i] = 0.0  # 无法对齐，不施加 KL 惩罚
```

---

## 10. 涉及修改的文件清单

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `verl/verl/trainer/ppo/ray_trainer.py` (L1366-L1450) | 交换 reward 和 old_log_prob 的执行顺序 | P0 |
| `verl/verl/experimental/reward/cot_rewriter/sglang_student_reward.py` | 新增 `_build_prompt_cot_only`，`compute_cognitive_reward_batch` 中新增 CoT-only 计算并返回 per-token log probs | P0 |
| `verl/verl/experimental/reward/cot_rewriter/combined_reward.py` | 新增 r_KL 计算、sigmoid 归一化、相关参数 | P0 |
| `examples/llama3_3b_grpo_2x80g.yaml` | 新增 r_KL/sigmoid 配置，修改 `ppl_on_full_response` 和 `use_reward_delta` | P0 |

---

## 11. 完整公式汇总

```
# ===== r_SS（answer 级，带 α）=====
r_SS(CoT) = avg{log π_s(a_t | Q, CoT, a_<t)} + α · min{log π_s(a_t | Q, CoT, a_<t)}
Δr_SS = r_SS(rewrite) - r_SS(original)

# ===== r_KL（CoT token 级，带 α）=====
kl_t = max(0, log π_actor(c_t | Q, orig, c_<t) - log π_student(c_t | Q, c_<t))
r_KL = avg(kl_t) + α · max(kl_t)

# ===== 组合 =====
R_raw = Δr_SS - λ_KL · r_KL                    # 核心
      - λ_FSF · r_FSF                           # 可选: FSF 逻辑惩罚
      + penalty_short + penalty_long + penalty_repeat  # 可选: 长度/重复惩罚

# ===== Sigmoid 归一化 =====
R_final = sigmoid(R_raw / τ) ∈ (0, 1)

# α = 0.01, λ_KL ∈ [0.05, 1.0], τ ≈ std(R_raw)
```
