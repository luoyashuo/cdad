# CoT Rewrite 推理流水线任务文档

## 概述
构建端到端的 CoT（思维链）改写推理流水线，包括模型加载、批量推理生成、质量评估和筛选，最终构建高质量微调数据集。

---

## 脚本 1：推理与评估流水线（步骤 1-4）

**目标：** 一键运行模型加载、批量改写生成、正确性判断和 FSF 评分，支持自动断点续传。

### 功能模块

#### 1.1 模型加载与 HF 格式转换
**输入：**
- 模型检查点路径（如：`/data/cdad/checkpoints/.../global_step_30`）

**处理逻辑：**
- 自动检测检查点格式（标准 HF 格式 / LoRA / FSDP / Megatron）
- 如果是 LoRA 格式：使用 `verl.model_merger` 合并权重
- 如果是 FSDP/Megatron 分布式格式：调用对应 merger 转为 HF 格式
- 最终加载为标准的 HuggingFace 模型用于推理

**参考实现：**
- `verl.model_merger.__main__` 命令行工具
- `FSDPModelMerger` / `MegatronModelMerger` 类

**输出：**
- 加载到 GPU 的 HF 格式模型（vLLM 或 SGLang Engine）

#### 1.2 批量改写生成（Rollout）
**输入：**
- ShareGPT 格式 JSONL 数据文件（如：`/data/hf_home/datasets/.../train.jsonl`）

**数据解析规则：**
```python
question = conversations[0]['content'].strip()
original_cot = conversations[1]['content'].split("<think>")[-1].split("</think>")[0].strip()
solution = conversations[1]['content'].split("</think>")[-1].strip()
```

**提示词构建：**
参考 `/home/luoys/cdad/reasoning_graph/build_cot_rewrite_from_deepmath.py` 中的 `build_prompt()` 函数：
- System Prompt：CoT 改写专家指令
- User Prompt：包含问题、原始推理过程，要求生成改写后的推理

**推理要求：**
- 每个样本生成 `N` 个改写（rollout_num 可配置）
- 支持多卡数据并行（DP）推理
- 使用 tqdm 显示总体进度

**断点续传机制：**
- 维护一个状态文件（`checkpoint.json`），记录：
  - 已处理样本索引
  - 当前步骤（rollout / correct_tag / fsf_tag）
  - 中间结果文件路径
- 启动时自动检测状态文件，从断点继续

**输出格式（JSONL）：**
```json
{
    "question": "问题文本",
    "original_cot": "原始思维链",
    "solution": "最终答案",
    "rewrite_cots": [
        {"rewrite_cot": "改写后的思维链1"},
        {"rewrite_cot": "改写后的思维链2"}
    ]
}
```

#### 1.3 正确性标签标注（Correct Tag）
**输入：**
- 步骤 1.2 的输出 JSONL 文件

**处理逻辑：**
参照 `sglang_correct_judge.py`：
- 使用 SGLang Correct Judge 模型
- 对每个 `rewrite_cot` 判断是否能逻辑推导出 `solution`
- 输出 `<judge>Yes</judge>` 或 `<judge>No</judge>`
- 解析结果为 `is_correct: true/false`

**断点续传：**
- 检测状态文件，从上次中断的样本继续

**输出格式（JSONL）：**
```json
{
    "question": "问题文本",
    "original_cot": "原始思维链",
    "solution": "最终答案",
    "rewrite_cots": [
        {"rewrite_cot": "改写1", "is_correct": true},
        {"rewrite_cot": "改写2", "is_correct": false}
    ]
}
```

#### 1.4 FSF 评分标签（FSF Tag）
**输入：**
- 步骤 1.3 的输出 JSONL 文件

**处理逻辑：**
参照 `sglang_fsf_reward.py`：
- 使用 SGLang FSF Judge 模型
- 对每个 `rewrite_cot` 生成 DOT 格式推理图
- 解析 DOT 图计算 FSF（Failure Step Fraction）= 失败节点数 / 总节点数
- 失败节点标识：`fillcolor=lightpink`
- 同时记录原始模型输出 `fsf_model_output`

**断点续传：**
- 检测状态文件，从上次中断的样本继续

**输出格式（JSONL）：**
```json
{
    "question": "问题文本",
    "original_cot": "原始思维链",
    "solution": "最终答案",
    "rewrite_cots": [
        {
            "rewrite_cot": "改写1",
            "is_correct": true,
            "fsf": 0.15,
            "fsf_model_output": "DOT图内容..."
        }
    ],
    "fsf_tag": true
}
```

### 启动方式（YAML 配置 + Bash 脚本）

#### 1. 创建配置文件
参考 `config_example.yaml`：

```yaml
# 基本配置
experiment_name: "cot_rewrite_exp1"
output_root: "/data/output/cot_rewrite"
resume: true
seed: 42

# 模型配置
model:
  checkpoint_path: "/data/cdad/checkpoints/.../global_step_30"
  format: "auto"  # auto/hf/lora/fsdp/megatron
  merge_lora: true
  dtype: "bfloat16"
  max_model_len: 8192

# Rollout 配置
rollout:
  input_data: "/data/hf_home/datasets/.../train.jsonl"
  num_rollouts: 4
  batch_size: 32
  generation:
    temperature: 0.7
    top_p: 0.9
    max_new_tokens: 4096
  parallel:
    tensor_parallel_size: 1

# Correct Judge 配置
correct_judge:
  model_path: "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B"
  enabled: true
  batch_size: 8
  generation:
    temperature: 0.0
    max_new_tokens: 50

# FSF Judge 配置
fsf_judge:
  model_path: "/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B"
  enabled: true
  batch_size: 4
  generation:
    temperature: 0.6
    max_new_tokens: 8000
```

#### 2. 使用 Bash 启动

**基本用法：**
```bash
bash inference/run_pipeline.sh inference/config_example.yaml
```

**从指定步骤开始：**
```bash
bash inference/run_pipeline.sh inference/config_example.yaml --step correct
```

**强制断点续传：**
```bash
bash inference/run_pipeline.sh inference/config_example.yaml --resume
```

**显示帮助：**
```bash
bash inference/run_pipeline.sh --help
```

#### 3. Python 模块直接调用

```bash
python -m inference.pipeline --config inference/config_example.yaml
```

**可选参数：**
| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径（必需） |
| `--step` | 指定起始步骤: model/rollout/correct/fsf |
| `--resume` | 强制启用断点续传 |
| `--dry-run` | 仅检查配置，不实际运行 |

### 状态文件结构（checkpoint.json）
```json
{
    "version": "1.0",
    "created_at": "2026-03-14T10:00:00",
    "updated_at": "2026-03-14T12:30:00",
    "config": {
        "model_path": "...",
        "input_data": "...",
        "rollout_num": 4
    },
    "progress": {
        "total_samples": 30000,
        "current_step": "fsf_tag",
        "completed_steps": ["rollout", "correct_tag"],
        "step_progress": {
            "rollout": {"completed": 30000, "output_file": ".../rollout_output.jsonl"},
            "correct_tag": {"completed": 30000, "output_file": ".../correct_tag_output.jsonl"},
            "fsf_tag": {"completed": 15000, "output_file": ".../fsf_tag_output.jsonl"}
        }
    }
}
```

---

## 脚本 2：ShareGPT 数据集重建（步骤 5）

**目标：** 从步骤 1-4 的结果中筛选最优改写，重建 ShareGPT 格式数据集。

### 功能模块

#### 2.1 筛选策略
支持以下筛选条件的灵活组合：

| 筛选条件 | 说明 |
|----------|------|
| `--filter_fsf_positive` | 只保留 FSF > 0 的样本 |
| `--filter_correct_only` | 只保留 is_correct=True 的样本 |
| `--selection_strategy` | 选择策略：random / max_fsf / priority_correct |

**选择策略说明：**
- `random`：从满足条件的改写中随机选择 1 个
- `max_fsf`：优先选择 FSF 分数最高的改写（即失败最少）
- `priority_correct`：优先选择 is_correct=True 的，其次按 FSF 排序

#### 2.2 输出格式转换
将筛选后的数据转为 ShareGPT 格式：
```json
{
    "conversations": [
        {"role": "user", "value": "问题文本"},
        {"role": "assistant", "value": "<think>\n改写后的思维链\n</think>\n\n最终答案"}
    ]
}
```

### 启动方式（YAML 配置 + Bash 脚本）

#### 1. 配置文件扩展（数据集重建部分）

在原有配置文件中添加 `dataset_build` 部分：

```yaml
# 数据集重建配置
dataset_build:
  # 筛选条件
  filters:
    fsf_positive: true      # 只保留 FSF > 0
    correct_only: true      # 只保留 is_correct=True
  
  # 选择策略: random/max_fsf/priority_correct
  selection_strategy: "priority_correct"
  
  # 输出文件名
  output_filename: "train_sharegpt.jsonl"
```

#### 2. 使用 Bash 启动

**使用配置文件中的筛选条件：**
```bash
bash inference/build_dataset.sh inference/config_example.yaml
```

**命令行覆盖筛选条件：**
```bash
bash inference/build_dataset.sh inference/config_example.yaml \
    --filter-fsf-positive \
    --filter-correct-only \
    --strategy priority_correct
```

**指定输入输出文件：**
```bash
bash inference/build_dataset.sh inference/config_example.yaml \
    --input-file /path/to/fsf_tag_output.jsonl \
    --output-file /path/to/output.jsonl
```

**显示帮助：**
```bash
bash inference/build_dataset.sh --help
```

#### 3. Python 模块直接调用

```bash
python -m inference.build_sharegpt --config inference/config_example.yaml
```

**可选参数：**
| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径（必需） |
| `--filter-fsf-positive` | 过滤 FSF > 0 |
| `--filter-correct-only` | 过滤 is_correct=True |
| `--strategy` | 选择策略: random/max_fsf/priority_correct |
| `--input-file` | 覆盖输入文件路径 |
| `--output-file` | 覆盖输出文件路径 |

---

## 流水线执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    脚本 1: pipeline_step1_4.py                   │
│                   (一键运行，支持断点续传)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  步骤 1      │───▶│  步骤 2      │───▶│  步骤 3      │      │
│  │ 模型加载     │    │ Rollout      │    │ Correct Tag  │      │
│  │ &格式转换    │    │ 改写生成     │    │ 正确性判断   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                           ┌──────▼───────┐      │
│                                           │  步骤 4      │      │
│                                           │ FSF Tag      │      │
│                                           │ 评分计算     │      │
│                                           └──────────────┘      │
│                                                                 │
│  checkpoint.json 自动记录进度，支持任意步骤中断后恢复             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 脚本 2: build_sharegpt_dataset.py                │
│                    (数据集重建与筛选)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 读取步骤 4   │───▶│ 应用筛选条件 │───▶│ 选择最优改写 │      │
│  │ 输出结果     │    │ (FSF/Correct)│    │ (策略选择)   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                           ┌──────▼───────┐      │
│                                           │ 输出 ShareGPT│      │
│                                           │ 格式数据集   │      │
│                                           └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 输出目录结构

```
/data/output/cot_rewrite_exp1/
├── checkpoint.json              # 断点状态文件
├── config.yaml                  # 运行配置（自动复制）
├── logs/
│   ├── pipeline.log             # 主日志
│   ├── rollout.log              # 步骤 2 详细日志
│   ├── correct_tag.log          # 步骤 3 详细日志
│   └── fsf_tag.log              # 步骤 4 详细日志
├── intermediate/
│   ├── rollout_output.jsonl     # 步骤 2 输出
│   ├── correct_tag_output.jsonl # 步骤 3 输出
│   └── fsf_tag_output.jsonl     # 步骤 4 输出
└── final/
    └── train_sharegpt.jsonl     # 步骤 5 最终输出
```

## 项目文件清单

```
inference/
├── README.md                    # 原始需求文档
├── task.md                      # 本任务文档
├── config_example.yaml          # 完整配置文件示例
├── config_minimal.yaml          # 最小化配置文件示例
├── run_pipeline.sh              # 步骤 1-4 启动脚本（YAML模式）
├── build_dataset.sh             # 步骤 5 启动脚本（YAML模式）
├── pipeline.py                  # 步骤 1-4 Python 实现
├── build_sharegpt.py            # 步骤 5 Python 实现
└── utils/
    ├── config_loader.py         # YAML 配置加载器
    ├── checkpoint.py            # 断点续传管理
    └── __init__.py
```

---

**注意：** 脚本自动使用项目自带的虚拟环境 `.venv`，无需手动激活。

## 快速开始

### 1. 准备配置文件

```bash
cp inference/config_example.yaml inference/my_config.yaml
# 编辑 my_config.yaml，修改路径和参数
```

### 2. 运行完整流水线

```bash
# 步骤 1-4: 推理与评估
bash inference/run_pipeline.sh inference/my_config.yaml

# 步骤 5: 数据集重建
bash inference/build_dataset.sh inference/my_config.yaml
```

### 3. 中断后恢复

```bash
# 自动从上次中断处继续
bash inference/run_pipeline.sh inference/my_config.yaml --resume
```

### 4. 从指定步骤开始

```bash
# 跳过 rollout，从 correct judge 开始
bash inference/run_pipeline.sh inference/my_config.yaml --step correct
```

---

## 关键实现细节

### 断点续传检测逻辑
```python
def detect_checkpoint(output_dir):
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        return ckpt["progress"]["current_step"], ckpt
    return None, None
```

### 多卡数据并行示例
```python
# 使用 Ray Actor 实现多卡并行
@ray.remote
class InferenceWorker:
    def __init__(self, model_path, gpu_id):
        # 每个 worker 负责一部分数据
        pass

# 数据分片
data_chunks = split_data(data, num_gpus)
futures = [worker.inference.remote(chunk) for worker, chunk in zip(workers, data_chunks)]
results = ray.get(futures)
```

### 步骤间数据传递
- 每个步骤完成后，将中间结果保存到 `intermediate/` 目录
- 下一步骤从上一中间文件读取，避免重复计算
- 状态文件记录每个步骤的完成状态，支持从任意步骤恢复
