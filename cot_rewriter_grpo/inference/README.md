# 数据集改写工具

使用改写模型对原有数据集进行改造。

## 功能概述

该脚本实现了完整的数据集改写流程：

1. **模型格式转换**: 将训练后的改写模型(checkpoint)转换为Huggingface格式
2. **Prompt构建**: 为改写模型构建system_prompt和user_prompt
3. **改写生成**: 调用改写模型，每个输入生成n次改写结果（支持断点续传，带tqdm进度条）
4. **相似度过滤**: 计算similarity_penalty，过滤掉过于相似的结果（多进程加速，带tqdm进度条）
5. **PPL过滤**: 使用student model计算ppl_relative_improvement，过滤掉改进不足的结果（支持断点续传，带tqdm进度条）
6. **FSF过滤**: 使用FSF judge model计算FSF，过滤掉逻辑错误过多的结果（支持断点续传，带tqdm进度条）
7. **结果选择**: 选择最优改写结果，生成新的数据集（带tqdm进度条）

### 进度显示

所有步骤都支持tqdm进度条显示：
- 显示当前进度、速度、预计剩余时间
- 显示每个样本的处理状态（如成功数、错误信息等）
- 支持动态更新，适配不同终端宽度

## 使用方法

### 1. 准备配置文件

复制示例配置并修改：

```bash
cp rewrite_config_example.yaml my_config.yaml
```

编辑配置文件，设置模型路径、数据路径、阈值等参数。

### 2. 运行完整流程

```bash
python rewrite_dataset.py --config my_config.yaml
```

### 3. 分步骤运行（可选）

可以单独运行某个步骤：

```bash
# 只运行改写生成步骤
python rewrite_dataset.py --config my_config.yaml --step rewrite

# 只运行similarity过滤
python rewrite_dataset.py --config my_config.yaml --step similarity

# 只运行PPL过滤
python rewrite_dataset.py --config my_config.yaml --step ppl

# 只运行FSF过滤
python rewrite_dataset.py --config my_config.yaml --step fsf

# 只运行结果选择和生成
python rewrite_dataset.py --config my_config.yaml --step select
```

### 4. 断点续传

脚本支持自动断点续传：

- **改写生成步骤**: 如果中断，会从中断的样本继续
- **PPL过滤步骤**: 如果中断，会从中断的样本继续
- **FSF过滤步骤**: 如果中断，会从中断的样本继续

检查点文件保存在 `{output_dir}/ckpt/` 目录下。

## 配置文件说明

```yaml
# 模型路径
rewrite_model_path: /path/to/rewrite/checkpoint      # 改写模型checkpoint路径
student_model_path: /path/to/student/model           # Student model路径（用于PPL计算）
fsf_judge_model_path: /path/to/fsf/model             # FSF judge model路径

# 数据路径
input_data_path: /path/to/input/data.jsonl           # 输入数据路径
output_dir: /path/to/output                          # 输出目录

# 改写参数
n_rewrites: 4                                        # 每个样本生成n次改写

# 阈值参数
similarity_penalty_threshold: 0.9                    # 相似度阈值，高于此值丢弃
ppl_relative_improvement_threshold: 0.0              # PPL改进阈值，低于此值丢弃
fsf_threshold: 0.0                                   # FSF阈值，高于此值丢弃

# 模型生成参数
max_new_tokens: 15000
temperature: 0.6
top_p: 0.95
presence_penalty: 1.2
frequency_penalty: 1.2

# 多进程参数
num_workers: 8                                       # similarity过滤使用的进程数
batch_size: 32

# SGLang参数
sglang_tensor_parallel_size: 1                       # SGLang tensor并行大小
sgland_micro_batch_size: 4                           # SGLang micro batch大小
```

## 输入数据格式

输入数据为JSONL格式，每条记录包含conversations字段：

```json
{
  "conversations": [
    {"role": "user", "content": "问题文本"},
    {"role": "assistant", "content": "<think>推理过程</think>\n\n答案"}
  ]
}
```

## 输出数据格式

输出数据为JSONL格式，保存在 `{output_dir}/rewritten_dataset.jsonl`：

```json
{
  "conversations": [
    {"role": "user", "content": "问题文本"},
    {"role": "assistant", "content": "<think>改写后的推理过程</think>\n\n答案"}
  ],
  "metadata": {
    "using_original": false,
    "original_thought": "原始推理过程",
    "answer": "答案",
    "rewrite_metrics": {
      "ppl_relative_improvement": 0.5,
      "fsf": 0.0,
      "similarity_penalty": 0.3
    }
  }
}
```

## 检查点文件

中间结果保存在 `{output_dir}/ckpt/` 目录：

- `ckpt_1.jsonl`: 改写生成结果
- `ckpt_1_5.jsonl`: similarity过滤结果
- `ckpt_2.jsonl`: PPL过滤结果
- `ckpt_3.jsonl`: FSF过滤结果
- `progress_*.json`: 进度文件，用于断点续传

## 输出统计

统计信息保存在 `{output_dir}/stats.json`，包含：

- 总样本数
- 使用改写的样本数
- 使用原始的样本数
- 各项过滤的通过情况

## 依赖

确保已安装以下依赖：

```bash
# 基础依赖
pip install transformers torch numpy pyyaml

# SGLang (用于改写模型推理和reward计算)
pip install sglang

# 其他依赖
pip install pydot  # 用于FSF解析
```

## 注意事项

1. **显存管理**: PPL和FSF步骤使用SGLang，会占用较多显存。如果显存不足，可以减小`sgland_micro_batch_size`

2. **模型路径**: `rewrite_model_path`指向的是训练后的checkpoint目录，脚本会自动转换为HF格式并保存到`{rewrite_model_path}/hf`

3. **阈值调整**: 
   - `similarity_penalty_threshold`: 值越大，过滤越严格（建议0.8-0.95）
   - `ppl_relative_improvement_threshold`: 建议保持0.0，只保留有改进的结果
   - `fsf_threshold`: 建议保持0.0，过滤掉有任何逻辑错误的结果

4. **多进程**: similarity过滤步骤使用多进程加速，可以根据CPU核心数调整`num_workers`

5. **Tensor Parallelism (TP)**: 
   - 对于7B/8B模型，TP=1（单卡）通常足够快
   - 对于更大模型（如70B），可能需要TP=4或TP=8
   - TP越大，单卡显存占用越少，但多卡间通信开销会增加
   - 建议根据模型大小和可用GPU数量调整`sglang_tensor_parallel_size`

6. **进度显示**: 所有步骤都支持tqdm进度条，可以实时查看处理进度、速度和预计剩余时间
