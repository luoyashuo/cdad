# MCC-KD: Multi-CoT Consistent Knowledge Distillation (DeepMath 版本)

本项目在 DeepMath 数据集上使用 Qwen2.5-7B-Instruct 模型复现 **MCC-KD**（Multi-CoT Consistent Knowledge Distillation）方法。

## 与 MCC-KD 官方实现的对齐

本实现严格遵循 MCC-KD 官方实现（`/home/luoys/qzq/MCC-KD`）的核心逻辑：
- **数据格式对齐**: 使用 `instruction`, `output_a`, `output_b`, `indices_a`, `indices_b` 格式
- **损失计算对齐**: 完全复现 `ParallelSolverMccDistillTrainer` 的损失计算逻辑
- **KL 计算对齐**: 使用与 `src/criterion.py` 中 `KLDivLoss` 相同的实现

---

## 项目概述

MCC-KD 是一种基于思维链（Chain-of-Thought, CoT）的知识蒸馏方法，通过以下两个核心机制提升学生模型的推理能力：

1. **Rationale Filtering（推理过滤）**：基于 N-gram Jaccard 相似度筛选多样化的推理路径
2. **Multi-CoT Consistent Distillation（多 CoT 一致蒸馏）**：使用双向 KL 散度确保多个推理路径预测结果的一致性

---

## 数据集预处理逻辑

### 输入数据格式 (DeepMath 原始格式)

```json
{
  "question": "数学问题文本...",
  "reasoning": [
    {"thought": "思考过程1...", "solution": "解答1...\\boxed{答案1}"},
    {"thought": "思考过程2...", "solution": "解答2...\\boxed{答案2}"},
    {"thought": "思考过程3...", "solution": "解答3...\\boxed{答案3}"}
  ]
}
```

### 3.2 Rationale Filtering（推理过滤）

**实现文件**: `data_preprocess.py`

**核心逻辑**（与 MCC-KD 官方实现完全一致）:

```python
def rationale_filtering(rationales, target_count=2, ngram_size=3):
    """
    1. 将所有 rationale 转换为 N-gram 集合
    2. 计算所有 rationale 对之间的 Jaccard 相似度
    3. 找出相似度最高的一对，随机保留其中一个
    4. 重复直到保留目标数量的 rationale
    """
    while len(working_list) > target_count:
        # 找出 Jaccard 相似度最高的一对
        (i, j) = argmax_{m,n} |S_m ∩ S_n| / |S_m ∪ S_n|
        # 随机保留一个
        working_list.pop(random.choice([i, j]))
```

**Jaccard 相似度计算**:
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### 输出数据格式 (MCC-KD 对齐格式)

预处理后的数据格式与 MCC-KD 官方实现完全对齐：

```json
{
  "instruction": "数学问题文本...",
  "output_a": "<think>\n思考过程1\n</think>\n\n解答1...\\boxed{答案1}",
  "output_b": "<think>\n思考过程2\n</think>\n\n解答2...\\boxed{答案2}",
  "indices_a": [[start_char_1, end_char_1]],
  "indices_b": [[start_char_2, end_char_2]],
  "_answer_text_a": "答案1",
  "_answer_text_b": "答案2"
}
```

**关键字段说明**:
- `output_a` / `output_b`: 两个筛选后的 rationale 的格式化输出
- `indices_a` / `indices_b`: **答案（\\boxed{} 中的内容）**在 output 中的**字符级**位置 `[[start, end]]`
- 格式遵循 Qwen2.5-Instruct 的 chat template: `<think>\n{thought}\n</think>\n\n{solution}`

### 预处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 读取 deepmath_sft_30k/train.jsonl                            │
│ 2. 随机采样 15,000 个 question                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 对每个问题的 reasoning 列表应用 Rationale Filtering          │
│    - 提取 thought + solution 的 N-gram                          │
│    - 计算所有对的 Jaccard 相似度                                │
│    - 迭代移除高相似度的 rationale，保留 2 个                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 格式化输出                                                   │
│    - 拼接 thought 和 solution: `<think>\n{t}\n</think>\n\n{s}`  │
│    - 提取 \\boxed{} 中的答案                                    │
│    - 计算答案在 output 中的字符位置 [[start, end]]              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 保存为 train_mcckd.jsonl                                     │
│    格式: {instruction, output_a, output_b, indices_a, indices_b}│
└─────────────────────────────────────────────────────────────────┘
```

---

## 损失计算逻辑

### 总体目标函数 (3.4 Overall Objective)

与 MCC-KD 官方实现 `ParallelSolverMccDistillTrainer` 完全对齐：

```
L_total = L_ce + α * L_kl
```

其中：
- **L_ce**: 交叉熵损失（对 output_a 和 output_b 分别计算后取平均）
- **L_kl**: 双向 KL 散度损失（只在 answer 区域计算）
- **α**: KL 损失的权重系数（默认 0.1）

### 3.3 Multi-CoT Consistent Distillation

**实现文件**: `train_mcckd.py` 中的 `MCCKDTrainer`

#### L_ce（交叉熵损失）计算

```python
# 第一次前向传播：计算 logits_a 和 ce_loss_a
logits_a = model(input_ids_a, attention_mask_a).logits
ce_loss_a = CrossEntropyLoss(ignore_index=-100)(
    logits_a[..., :-1, :].view(-1, vocab_size),
    labels_a[..., 1:].view(-1)
)

# 第二次前向传播：计算 logits_b 和 ce_loss_b
logits_b = model(input_ids_b, attention_mask_b).logits
ce_loss_b = CrossEntropyLoss(ignore_index=-100)(
    logits_b[..., :-1, :].view(-1, vocab_size),
    labels_b[..., 1:].view(-1)
)

# 平均 CE Loss
L_ce = (ce_loss_a + ce_loss_b) * 0.5
```

**注意**: 
- 使用 `ignore_index=-100` 忽略 prompt 部分的损失
- 对 output_a 和 output_b **分别进行前向传播**（与官方实现一致）

#### L_kl（KL 散度损失）计算

**核心方法**: `compute_kl_for_mcc()` - 与 MCC-KD 官方实现完全对齐

```python
def compute_kl_for_mcc(indices_a, indices_b, logits_a, logits_b, T):
    """
    1. 根据 indices 提取 answer 区域的 logits
    2. 对齐两个 rationale 的 answer 长度
    3. 计算双向 KL 散度
    """
    # 提取 answer 区域的 logits
    start_a, end_a = indices_a[0]
    start_b, end_b = indices_b[0]
    
    logits_a_answer = logits_a[:, start_a:end_a, :]  # [batch, ans_len_a, vocab]
    logits_b_answer = logits_b[:, start_b:end_b, :]  # [batch, ans_len_b, vocab]
    
    # 对齐序列长度（取较短的长度）
    min_len = min(logits_a_answer.size(1), logits_b_answer.size(1))
    logits_a_answer = logits_a_answer[:, :min_len, :]
    logits_b_answer = logits_b_answer[:, :min_len, :]
    
    # 计算双向 KL 散度
    p_loss = KLDivLoss(logits_a_answer, logits_b_answer, masks, T)
    q_loss = KLDivLoss(logits_b_answer, logits_a_answer, masks, T)
    
    return (p_loss + q_loss) * 0.5
```

**KL 散度计算**（`KLDivLoss` 类，与 `src/criterion.py` 对齐）：

```python
class KLDivLoss(nn.Module):
    def forward(self, logits, targets, masks=None, T=1.0):
        # 展平为 [batch*seq_len, vocab_size]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1))
        
        # 计算概率分布（使用温度 T）
        estimates = softmax(logits)
        targets = softmax(targets / T)
        
        # 使用 powmax 平滑（与官方实现一致）
        estimates = powmax(estimates + eps)
        targets = powmax(targets + eps)
        
        # KL(target || estimate) = sum(target * (log(target) - log(estimate)))
        loss = targets * (log(targets) - log(estimates))
        loss = sum(loss, dim=-1)
        
        # 应用 mask
        if masks is not None:
            loss = masked_select(loss, masks)
        
        return mean(loss)
```

**双向 KL 散度**:
```
L_kl = 0.5 * (KL(p||q) + KL(q||p))

其中:
KL(p||q) = Σ p(x) * (log p(x) - log q(x))
KL(q||p) = Σ q(x) * (log q(x) - log p(x))
```

#### 完整训练流程

```python
def compute_loss(model, inputs):
    # 准备数据
    input_ids_a = inputs["input_ids_a"]
    input_ids_b = inputs["input_ids_b"]
    labels_a = inputs["labels_a"]
    labels_b = inputs["labels_b"]
    answer_indices_a = inputs["answer_indices_a"]
    answer_indices_b = inputs["answer_indices_b"]
    
    # ========== L_ce 计算 ==========
    # 第一次前向传播
    logits_a = model(input_ids_a).logits
    ce_loss_a = CrossEntropyLoss(logits_a, labels_a)
    
    # 第二次前向传播
    logits_b = model(input_ids_b).logits
    ce_loss_b = CrossEntropyLoss(logits_b, labels_b)
    
    L_ce = (ce_loss_a + ce_loss_b) * 0.5
    
    # ========== L_kl 计算 ==========
    # 只在 answer 区域计算 KL
    L_kl = compute_kl_for_mcc(
        indices_a=answer_indices_a,
        indices_b=answer_indices_b,
        logits_a=logits_a,
        logits_b=logits_b,
        T=temperature
    )
    
    # ========== 总损失 ==========
    L_total = L_ce + alpha * L_kl
    
    return L_total
```

---

## 项目结构

```
mcc-deepmath/
├── data_preprocess.py          # 数据预处理：Rationale Filtering
├── train_mcckd.py              # 训练脚本：MCC-KD 损失计算
├── configs/
│   └── training_config.yaml    # 训练参数配置
├── run_preprocess.sh           # 数据预处理启动脚本
├── run_train.sh                # 训练启动脚本
└── README.md                   # 本说明文档
```

---

## 环境要求

```
transformers==4.55.0
accelerate==1.11.0
deepspeed==0.16.9
peft==0.15.1
torch==2.6.0+cu124
flash_attn==2.7.3
```

---

## 使用说明

### 1. 数据预处理

运行 Rationale Filtering，从原始 deepmath_sft_30k 数据集中：
- 随机采样 15,000 个 question
- 每个 question 筛选出 2 个多样化的 rationale
- 输出 MCC-KD 格式（instruction, output_a, output_b, indices_a, indices_b）

```bash
python data_preprocess.py \
    --input_path /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl \
    --output_path /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl \
    --sample_size 15000 \
    --rationales_per_question 2 \
    --seed 42
```

### 2. 模型训练

使用 8 张 GPU 进行分布式训练：

```bash
deepspeed --num_gpus=8 train_mcckd.py \
    --config configs/training_config.yaml
```

**训练配置** (`configs/training_config.yaml`):
- **模型**: Qwen2.5-7B-Instruct
- **微调方法**: LoRA (r=128, alpha=16)
- **分布式**: DeepSpeed ZeRO-2
- **训练参数**:
  - Epochs: 2
  - Batch size per GPU: 1
  - Gradient accumulation: 8
  - Global batch size: 64 (1 × 8 × 8)
  - Learning rate: 1e-5
  - Warmup ratio: 0.1
  - LR scheduler: cosine
- **MCC-KD 参数**:
  - alpha: 0.1 (KL 损失权重)
  - kl_temperature: 1.0
  - kl_eps: 7e-5

---

## 关键技术细节

### 与 MCC-KD 官方实现的对齐点

| 组件 | MCC-KD 官方实现 | 本实现 |
|------|----------------|--------|
| **数据格式** | `instruction, output_a, output_b, indices_a, indices_b` | 完全对齐 |
| **Rationale Filtering** | `src/dataset.py` 中的 `deduplicate_texts` | 相同逻辑，使用 N-gram Jaccard |
| **KL 计算** | `src/criterion.py` 中的 `KLDivLoss` | 完全复现，使用 `powmax` 平滑 |
| **损失计算** | `src/trainer.py` 中的 `ParallelSolverMccDistillTrainer.distill` | 完全复现，两次前向传播 |
| **双向 KL** | `(KL(p\|\|q) + KL(q\|\|p)) * 0.5` | 完全对齐 |

### 与原版 DeepMath 数据的适配

| 特性 | 原版 DeepMath | MCC-KD 适配 |
|------|--------------|-------------|
| **输入格式** | `question, reasoning[{thought, solution}]` | 转换为 `instruction, output_a/b, indices_a/b` |
| **输出格式** | thought + solution | `<think>\n{thought}\n</think>\n\n{solution}` |
| **答案标识** | `\boxed{answer}` | 提取位置信息存入 `indices` |
| **损失区域** | 整个 sequence | 只对 `\boxed{}` 内容计算 KL |

---

## 监控指标

训练过程中会记录以下指标：
- `train/ce_loss`: 交叉熵损失（L_ce）
- `train/kl_loss`: KL 散度损失（L_kl）
- `train/kl_loss_weighted`: 加权后的 KL 损失（α * L_kl）
- `train/total_loss`: 总损失（L_total = L_ce + α * L_kl）
- `train/learning_rate`: 学习率变化

---

## 参考文献

```
MCC-KD: Multi-CoT Consistent Knowledge Distillation
Chen et al., 2023
https://arxiv.org/abs/2310.14747
```

---

## 实现说明

本实现严格遵循 MCC-KD 官方实现的核心逻辑：

1. **数据预处理**: 
   - 复现 `deduplicate_texts` 函数的逻辑，使用 N-gram Jaccard 相似度进行 rationale 筛选
   - 输出格式与 `MultiOutputsConsistentDataset` 对齐

2. **损失计算**:
   - 完全复现 `ParallelSolverMccDistillTrainer` 的 `distill` 方法
   - 两次独立的前向传播计算 logits_a 和 logits_b
   - 使用 `compute_kl_for_mcc` 方法提取 answer 区域的 logits
   - KL 计算使用 `powmax` 平滑，与 `KLDivLoss` 完全一致

3. **训练策略**:
   - 使用 LoRA 进行高效微调
   - 支持 DeepSpeed ZeRO-2 分布式训练
   - 适配 Flash Attention 2 加速
