# reward_mode: "kl" — Actor-Student KL 散度作为 R_cog

## 1. 设计目标

将 r_KL 作为 `reward_mode` 的一种选项（与 `"ppl"`、`"logprob"` 等并列），**不包含 r_SS**。

当 `reward_mode: "kl"` 时：
```
R_cog = -r_KL
```

即 R_cog 完全由 Actor-Student 在 CoT token 上的 KL 散度决定，不计算 answer 级别的 Student Success Reward。

---

## 2. 数学公式

### 2.1 逐 token KL 近似

```
kl_t = max(0, log π_actor(cot_t | Q, original_cot, cot_<t) - log π_student(cot_t | Q, cot_<t))
```

- `log π_actor(cot_t)`：Actor 对第 t 个 CoT token 的 log prob（prompt 包含 original_cot → "知情"）
- `log π_student(cot_t)`：Student 对同一 token 的 log prob（prompt 只含 question → "不知情"）
- `max(0, ·)`：KL 散度非负，只惩罚 Actor 比 Student 更自信的 token

### 2.2 聚合（α 机制）

```
r_KL = avg(kl_t) + α_kl · max(kl_t)
```

- `avg` 项：整体逻辑连贯性（平均每个 token 的不连贯程度）
- `α_kl · max` 项：最不连贯的那个 token（抓最严重的跳步/信息泄漏）
- α_kl = 0.01（默认值）

### 2.3 R_cog

```
R_cog = -r_KL
```

- r_KL ≥ 0，越大说明推理越不自洽
- R_cog ≤ 0，越接近 0 越好
- 典型取值范围：约 [-10, 0]

### 2.4 直觉举例

| 改写步骤 | Actor log prob | Student log prob | kl_t | 含义 |
|----------|---------------|-----------------|------|------|
| "设 f(x) = x²" | -2.0 | -2.5 | 0.5 | 双方都觉得自然 |
| "由泰勒展开得 sin(x)≈x" | -3.0 | -5.0 | 2.0 | Actor 见过原始推理知道要用泰勒展开，Student 觉得突然 |
| "所以答案是 42" | -0.5 | -8.0 | **7.5** | Actor 从 original_cot 直接搬答案，Student 完全无法预测 → **重罚** |

---

## 3. 两个 log prob 的数据来源

### 3.1 Actor CoT log probs — 来自 `old_log_probs`

Actor 的 per-token log probs 由 `ray_trainer.py` 中的 `compute_log_prob` 计算：

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
# old_log_prob.batch["old_log_probs"]: shape [batch_size, response_length]
```

- 覆盖整个 response（包括 `<rewrite_reasoning>` 标签和 CoT 内容）
- 需要从 response 中定位 CoT token 的起止位置，截取对应的 log probs

### 3.2 Student CoT log probs — 新增计算

Student Model 对改写后的 CoT 逐 token 给出 log prob。Student 的 prompt 只包含 question，不包含 original_cot。

**prompt 构造**：

```python
messages = [
    {'role': 'user', 'content': question},
    {'role': 'assistant', 'content': '', 'reasoning_content': cot}
]
```

Student 从 question 出发，评估 CoT 每个 token 的可预测性。

---

## 4. 执行顺序调整（前置条件）

### 4.1 问题

当前 `ray_trainer.py` 的执行顺序：

```
1. Rollout → 产生 responses
2. Reward  → 计算奖励          ← 此时 old_log_probs 不存在！
3. LogProb → 计算 old_log_probs
```

reward_mode="kl" 需要 `old_log_probs`，但它在 reward 之后才计算。

### 4.2 解决方案

交换步骤 2 和 3：

```
1. Rollout → 产生 responses
2. LogProb → 计算 old_log_probs，合并到 batch
3. Reward  → 计算奖励（现在可以从 batch 中读取 old_log_probs）
```

**安全性**：
- 两个步骤原本完全独立，不存在数据依赖
- 显存更优：FSDP Actor 计算完 log probs 后 offload，释放 GPU 给 Student Model
- 代码改动：只需调换两个 `with marked_timer(...)` 代码块的顺序

### 4.3 修改位置

`verl/verl/trainer/ppo/ray_trainer.py` L1366-L1453 附近：

**修改前**：
```python
                    # reward block (L1366-L1417)
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        ...

                    # old_log_prob block (L1420-L1453)
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        ...
                        batch = batch.union(old_log_prob)
```

**修改后**：
```python
                    # old_log_prob block（移到前面）
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
                        # batch.batch["old_log_probs"] 现在可用

                    # reward block（移到后面）
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        ...
```

---

## 5. 实现细节

### 5.1 涉及修改的文件

| 文件 | 修改内容 |
|------|---------|
| `verl/verl/trainer/ppo/ray_trainer.py` | 交换 reward 和 old_log_prob 的执行顺序 |
| `verl/verl/experimental/reward/cot_rewriter/sglang_student_reward.py` | 新增 `_build_prompt_cot_only`；`compute_cognitive_reward_batch` 新增 `"kl"` 分支 |
| `verl/verl/experimental/reward/cot_rewriter/combined_reward.py` | 新增 Actor CoT log probs 提取逻辑；新增 `"kl"` 模式的 R_cog 计算 |
| YAML 配置文件 | 新增 `alpha_kl` 参数 |

### 5.2 sglang_student_reward.py

#### 5.2.1 新增 `_build_prompt_cot_only` 方法

```python
def _build_prompt_cot_only(self, question: str, cot: str) -> tuple[str, int]:
    """
    构造 Student 评估 CoT 可理解性的 prompt。
    Student 只看 question，对 CoT 每个 token 给出 log prob。

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

与现有的 `_build_prompt(question, cot, answer)` 区别：
- `_build_prompt`：prompt 包含 Q + CoT + Answer，log prob 起始于 answer 的第一个 token
- `_build_prompt_cot_only`：prompt 包含 Q + CoT（无 answer），log prob 起始于 CoT 的第一个 token

#### 5.2.2 compute_cognitive_reward_batch 新增 "kl" 分支

当 `reward_mode == "kl"` 时：

```python
# Phase 1: 只构建 CoT-only prompts（不构建 rewrite/original/no_cot 的 answer prompts）
cot_only_prompts = []
cot_only_starts = []
for i in range(batch_size):
    cot_only_prompt, cot_only_start = self._build_prompt_cot_only(
        questions[i], rewrite_cots[i]
    )
    cot_only_prompts.append(cot_only_prompt)
    cot_only_starts.append(cot_only_start)

# Phase 2: 分发到 Ray workers 计算 log probs
# 复用现有的 worker.compute_logprobs.remote() 机制

# Phase 3: 收集结果，返回 per-token log probs
```

返回值新增：
```python
return {
    "rewards": [0.0] * batch_size,  # kl 模式下 reward 在 combined_reward.py 中计算
    "rewrite_mean_log_probs": [...],
    "original_mean_log_probs": [...],
    "rewrite_token_log_probs": [...],
    "rewrite_token_ids": [...],
    # === kl 模式新增 ===
    "student_cot_token_log_probs": student_cot_token_log_probs_list,  # List[List[float]]
    "student_cot_mean_log_probs": student_cot_mean_log_probs,
    "student_cot_min_log_probs": student_cot_min_log_probs,
}
```

> **注意**：kl 模式下，rewrite/original/no_cot 的 answer prompts **仍然计算**，用于 log 观测但不参与 R_cog。
> 这样可以同时观察 Δr_SS 的趋势，方便后续对比分析。
> 如果确定不需要，可以跳过这些计算以节省推理时间。

### 5.3 combined_reward.py

#### 5.3.1 新增参数

```python
def __init__(
    self,
    ...
    alpha_kl: float = 0.01,  # r_KL 中 max 项的权重（avg + α·max）
    ...
):
```

仅新增一个参数 `alpha_kl`。不需要 `use_kl_reward` / `lambda_kl` / `use_sigmoid_norm` / `sigmoid_tau`，因为 r_KL 就是 R_cog 本身，后续的 tiered reward、penalties 等由现有机制处理。

#### 5.3.2 从 response 中定位 CoT token 位置

```python
def extract_cot_token_range(response_ids_i, tokenizer):
    """
    从 response token IDs 中找到 CoT 内容的 token 起止位置。
    response 格式: <rewrite_reasoning>CoT 内容</rewrite_reasoning>

    Returns:
        (cot_start, cot_end): CoT 内容 token 的起止索引（不含标签 token）
    """
    response_text = tokenizer.decode(response_ids_i, skip_special_tokens=False)

    open_tag = '<rewrite_reasoning>'
    close_tag = '</rewrite_reasoning>'

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

#### 5.3.3 compute_reward 中 "kl" 模式的 R_cog 计算

在 `compute_reward()` 方法中，获取 `cog_result` 之后：

```python
if self.reward_mode == "kl":
    # ===== 1. 提取 Actor CoT log probs =====
    old_log_probs = data.batch['old_log_probs']    # [batch_size, response_length]
    response_ids = data.batch['responses']           # [batch_size, response_length]

    actor_cot_log_probs_list = []
    for i in range(batch_size):
        response_ids_i = response_ids[i]
        cot_start, cot_end = extract_cot_token_range(response_ids_i, tokenizer)

        if cot_start is not None and cot_end is not None:
            actor_cot_lp = old_log_probs[i, cot_start:cot_end].tolist()
        else:
            actor_cot_lp = []
        actor_cot_log_probs_list.append(actor_cot_lp)

    # ===== 2. 获取 Student CoT log probs =====
    student_cot_token_lps = cog_result["student_cot_token_log_probs"]

    # ===== 3. 逐 token KL 计算 =====
    r_kl_values = []
    for i in range(batch_size):
        actor_lp = actor_cot_log_probs_list[i]
        student_lp = student_cot_token_lps[i]

        if not actor_lp or not student_lp:
            r_kl_values.append(0.0)
            continue

        min_len = min(len(actor_lp), len(student_lp))
        if min_len < 5:
            r_kl_values.append(0.0)
            continue

        kl_per_token = [max(0.0, actor_lp[t] - student_lp[t]) for t in range(min_len)]

        kl_avg = sum(kl_per_token) / len(kl_per_token)
        kl_max = max(kl_per_token)
        r_kl = kl_avg + self.alpha_kl * kl_max

        r_kl_values.append(r_kl)

    # ===== 4. R_cog = -r_KL =====
    r_cog_raw = [-v for v in r_kl_values]
```

后续流程不变：r_cog_raw 进入 tiered reward → penalties → total_rewards。

### 5.4 Token 对齐说明

- Actor (Llama-3.2-3B-Instruct) 和 Student (Llama-3.2-1B) **共享同一个 tokenizer**
- 同一段 CoT 文本经过相同 tokenizer 编码后，token 序列完全一致
- 边界处理：Actor response 包含 `<rewrite_reasoning>` 标签，Student prompt 不包含。在标签边界处 BPE 分词可能略有不同，取两者较短长度对齐即可
- 对齐失败兜底：如果 `min_len < 5`，r_KL 设为 0（不施加惩罚）

---

## 6. YAML 配置

### 6.1 新增参数

仅新增 `alpha_kl`（可选，默认 0.01）：

```yaml
custom_reward_function:
  reward_kwargs:
    reward_mode: kl             # 使用 KL 散度作为 R_cog
    alpha_kl: 0.01              # r_KL 的 α 聚合参数（avg + α·max）
```

### 6.2 完整示例

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

    # R_cog = -r_KL（KL 散度模式）
    reward_mode: kl
    alpha_kl: 0.01              # avg + α·max 聚合
    use_reward_delta: false      # kl 模式下无需 delta（本身就是绝对值）

    # Tiered reward（可选，对 R_cog 做阶梯化）
    use_tiered_reward: false     # kl 模式的 R_cog 是负值，tiered reward 需要适配

    # FSF（可选）
    use_logic_reward: false

    # Correct Judge（可选）
    use_correct_judge: true
    correct_judge_penalty_value: 0.0

    # Penalties
    format_penalty: 0.2
    similarity_threshold: 0.5
    similarity_penalty: 0.0
    too_short_max_penalty: 0.0
    too_long_max_penalty: 0.0
    penalty_mode: absolute
```

---

## 7. 与现有 reward_mode 的对比

| 属性 | `ppl` / `logprob` | `kl`（新增） |
|------|-------------------|-------------|
| 评估目标 | Answer 质量（Student 预测 answer 的能力） | CoT 自洽性（推理步骤的逻辑连贯性） |
| Student 评估内容 | answer tokens 的 log prob | CoT tokens 的 log prob |
| 是否需要 Actor log probs | 否 | **是**（需要 old_log_probs） |
| 是否需要调整执行顺序 | 否 | **是**（old_log_prob 必须在 reward 前） |
| R_cog 含义 | 改写让 Student 更好地预测 answer | 改写的推理过程不依赖"内幕信息" |
| R_cog 符号 | 越高越好（正值为佳） | 越接近 0 越好（负值，绝对值越小越好） |
| 典型取值范围 | 因 mode 而异 | [-10, 0] |

---

## 8. 日志与监控

### 8.1 新增 metrics

```python
# 每个 step 记录的指标
"reward/r_kl_mean"           # r_KL 均值
"reward/r_kl_max"            # r_KL 最大值
"reward/kl_avg_mean"         # kl_avg（avg(kl_t)）的均值
"reward/kl_max_mean"         # kl_max（max(kl_t)）的均值
"reward/cot_token_aligned"   # 平均对齐的 CoT token 数
"reward/cot_token_align_fail"  # 对齐失败的样本数
```

### 8.2 per-sample 日志

```
[R_KL] Sample 0: kl_avg=0.3421, kl_max=2.5601, r_kl=0.3677, aligned_tokens=156/160
[R_KL] Sample 1: kl_avg=0.1234, kl_max=1.0023, r_kl=0.1334, aligned_tokens=203/203
[R_KL] Sample 2: cot_align_failed (missing close tag), r_kl=0.0
```

---

## 9. 注意事项

### 9.1 tiered reward 的适配

现有的 `apply_tiered_reward` 假设 R_cog ≥ 0（负值返回 0）。kl 模式下 R_cog ≤ 0，因此：
- 如果使用 `use_tiered_reward: true`，需要修改 tiered reward 逻辑以支持负值
- 建议 kl 模式下设置 `use_tiered_reward: false`

### 9.2 Correct Judge 的交互

kl 模式下 Correct Judge 仍然可用：
- 如果 Correct Judge 判定不正确，R_cog 被强制设为 `correct_judge_penalty_value`
- 这与 kl 模式不冲突

### 9.3 与其他 reward_mode 的切换

kl 模式是互斥的 reward_mode 选项。如果想同时使用 r_SS 和 r_KL，应该使用原有 migration doc 中描述的组合方案（`R_raw = Δr_SS - λ_KL · r_KL`），而不是 kl 作为 reward_mode。

### 9.4 执行顺序对其他 reward_mode 的影响

交换 old_log_prob 和 reward 的执行顺序对所有 reward_mode 都安全（因为原本两者不存在依赖关系）。但建议仅在 `reward_mode == "kl"` 时才执行交换，以最小化对其他模式的影响。或者，也可以无条件交换（更简单），因为逻辑上完全等价。
