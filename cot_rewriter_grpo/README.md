# CoT Rewriter GRPO

基于 VERL 框架的 Chain-of-Thought (CoT) 改写模型训练系统，使用 GRPO (Group Relative Policy Optimization) 算法。

## 目标

训练一个 **CoT Rewriter** 模型，能够将原始的 Chain-of-Thought 推理过程改写得更加清晰、高效，减少失败分支，帮助下游模型更好地理解和利用推理过程。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training System                               │
├─────────────────────────────────────────────────────────────────────┤
│  GPU 0-2: Actor Model (Qwen3-8B)                                    │
│           + Rollout Engine (vLLM/SGLang)                            │
│           + GRPO Training                                            │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP API (Reward Computation)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 3: vLLM Services                                                │
│    ├── Student Model (Qwen2.5-3B) → R_cog (Cognitive Reward)        │
│    └── FSF Judge (Qwen2.5-7B)     → R_logic (Logic Reward)          │
└─────────────────────────────────────────────────────────────────────┘
```

## 奖励函数设计

总奖励: `R = R_cog + λ * R_logic`

### 1. 认知奖励 (Cognitive Reward, R_cog)

```
<!-- R_cog = 1 if avg(log P_student(Answer | Question, Rewrite_CoT)) > avg(log P_student(Answer | Question, Original_CoT)) else 0 -->
我希望修改奖励为
R_cog = avg(log P_student(Answer | Question, Rewrite_CoT)) + alpha * min(avg(log P_student(Answer | Question, Rewrite_CoT))) ,
```

- 使用冻结的 Student Model 计算改写后 CoT 对答案预测的概率
- 目的：激励改写的 CoT 能让 Student Model 更容易预测出正确答案

### 2. 逻辑奖励 (Logic Reward, R_logic)

```
R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0
```

- FSF (Failed Step Fraction): 失败分支占比
- 使用 LLM-as-a-Judge 分析推理过程中的失败分支
- 目的：鼓励减少无效的推理分支

## 环境配置

### 硬件要求
| 配置模式 | GPU 要求 | 显存需求 | 说明 |
|---------|---------|---------|------|
| 全参数微调 | 4 × A100-80G | ~64GB/卡 | 推荐配置，训练速度快 |

### 软件要求
- **Python**: >= 3.10 
- **PyTorch**: >= 2.6.0 
- **CUDA**: >= 12.4 
- **python**: == 3.12
- **CUDA Toolkit**: 需与 PyTorch 版本匹配

### 使用 uv 配置环境

本项目使用 [uv](https://docs.astral.sh/uv/) 作为 Python 包管理工具。uv 是一个极速的 Python 包管理器和解析器，用 Rust 编写。

#### 1. 安装 uv

```bash
# 使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

#### 2. 克隆仓库并设置项目结构

```bash
# 克隆主项目
git clone <repository-url> cot_rewriter_grpo
cd cot_rewriter_grpo

# VERL 框架需要和本项目在同一目录下
# 克隆 verl 框架（父目录）
cd ..
git clone https://github.com/volcengine/verl.git

# 最终目录结构
cdad/
├── cot_rewriter_grpo/     # 本项目
│   ├── src/
│   ├── examples/
│   ├── scripts/
│   └── README.md
└── verl/                   # VERL 框架
    └── verl/
        └── trainer/
```

#### 3. 创建虚拟环境并安装依赖

```bash
cd cot_rewriter_grpo

# 使用 uv 创建虚拟环境（Python 3.12 推荐）
uv venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate


# 安装 VERL 框架（在父目录的 verl 中）
cd verl
uv pip install -e .
uv pip install transformers==4.55.0
uv pip install zmq
uv pip install swanlab

uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install vllm==0.8.5.post1
uv pip install opentelemetry-api==1.34.1 opentelemetry-sdk==1.34.1 opentelemetry-exporter-otlp==1.34.1
```

**主要依赖说明：**

| 依赖包 | 用途 |
|--------|------|
| `torch>=2.6.0` | PyTorch 深度学习框架 |
| `transformers` | HuggingFace 模型库 |
| `vllm` | 高效推理引擎 |
| `ray` | 分布式训练框架 |
| `hydra-core` | 配置管理 |
| `omegaconf` | YAML 配置文件解析 |
| `datasets` | 数据集处理 |
| `peft` | LoRA 等参数高效微调 |
| `accelerate` | 训练加速 |

#### 4. 下载模型

本项目需要以下模型：

| 模型 | 用途 | 路径配置 |
|------|------|----------|
| Qwen3-8B | Actor Model（主训练模型） | `actor_rollout_ref.model.path` |
| Qwen3-1.7B | Student Model（认知奖励） | `reward_model.custom_reward_function.reward_kwargs.student_model_path` |
| FSF Judge | 逻辑奖励判断模型 | `reward_model.custom_reward_function.reward_kwargs.fsf_model_path` |

```bash
# 使用 HuggingFace CLI 下载模型（示例）
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/Qwen3-8B
huggingface-cli download Qwen/Qwen3-1.7B --local-dir /path/to/Qwen3-1.7B

# 或使用 Python
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-8B')"
```

#### 5. 准备训练数据

```bash
# 方式1：使用数据转换脚本（推荐）
bash scripts/grpo/convert_data.sh data/raw/train.jsonl data/processed 0.9 jsonl

# 方式2：直接使用原始数据（会自动进行字段映射）
# 在配置文件中设置：
# data.train_files: /path/to/original.jsonl
```

**数据格式说明：**

原始数据格式（支持自动映射）：
```json
{
  "problem": "问题描述",
  "question": "具体问题",
  "thought": "原始CoT推理过程",
  "solution": "正确答案",
  "fsf": 0.2
}
```

转换后格式：
```json
{
  "prompt": "改写提示",
  "reward_model": {
    "question": "具体问题",
    "original_cot": "原始CoT",
    "ground_truth": "正确答案"
  }
}
```

#### 6. 配置环境变量

```bash
# 可选：添加到 ~/.bashrc 或 ~/.zshrc
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### 7. 验证安装

```bash
# 检查 GPU 可用性
nvidia-smi

# 验证 Python 环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"

# 验证 VERL 安装
python -c "from verl.trainer.main_ppo import run_ppo; print('VERL installed successfully')"
```

#### 8. 启动训练

```bash
# 使用默认配置启动训练
bash scripts/grpo/run_train.sh

# 指定配置文件和 GPU
bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu 0,1,2,3

# 或直接运行 Python
python src/trainer/train.py --config-name=qwen3_8b_grpo_4gpu
```

### 常见问题

**Q: 遇到 `ModuleNotFoundError: No module named 'verl'`**

A: 确保 VERL 框架已克隆到父目录，并执行了 `uv pip install -e .` 安装

**Q: 遇到 CUDA 版本不匹配**

A: 确保 PyTorch CUDA 版本与系统 CUDA 版本匹配：
```bash
python -c "import torch; print(torch.version.cuda)"  # 查看 PyTorch CUDA 版本
nvcc --version  # 查看系统 CUDA 版本
```

**Q: 训练时显存不足**

A: 可以尝试以下方法：
- 启用 LoRA（在配置中设置 `actor_rollout_ref.model.lora`）
- 减小 `train_batch_size` 和 `gen_batch_size`
- 降低 `max_prompt_length` 和 `max_response_length`
- 增加 `gradient_checkpointing`

**Q: Ray 初始化失败**

A: 检查是否有其他 Ray 进程占用资源：
```bash
ray status
ray stop --force  # 强制停止所有 Ray 进程
```

