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
R_cog = log P_student(Answer | Question, Rewrite_CoT)
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
| LoRA 微调 | 2 × A100-80G | ~40GB/卡 | 资源受限情况下的选择 |
| 轻量级训练 | 1 × A100-80G | ~70GB/卡 | 需要进一步降低 batch size 和 rollout 数 |

### 软件要求
- **Python**: >= 3.10 (推荐 3.11)
- **PyTorch**: >= 2.6.0 (推荐 2.6.1)
- **CUDA**: >= 12.4 (推荐 12.4 或 12.5)
- **CUDA Toolkit**: 需与 PyTorch 版本匹配

### 安装步骤

本项目使用 `uv` 进行依赖管理，提供一致、快速的包安装体验。

```bash
# 1. 安装 uv 工具（如果尚未安装）
pip install uv

# 2. 进入项目目录
cd cot_rewriter_grpo

# 3. 创建并激活虚拟环境
uv venv
source .venv/bin/activate

# 4. 安装项目依赖（包括 torch、transformers、vLLM 等核心库）
uv pip install -e .

# 5. 安装 VERL 框架 (editable mode，本地开发模式)
uv pip install -e ../verl
```

### 依赖验证
安装完成后，验证关键依赖是否正常工作：
```bash
# 检查 Python 版本
python --version  # 应 >= 3.10

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 VERL 框架
python -c "import verl; print(f'Verl: {verl.__version__}')"

# 检查 vLLM
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

### uv 使用技巧
```bash
# 查看已安装的包
uv pip list

# 更新所有依赖
uv pip upgrade

# 冻结依赖到 requirements.txt
uv pip freeze > requirements.txt

# 安装特定版本的包
uv pip install torch==2.6.1

# 运行 Python 脚本（使用虚拟环境）
uv run python src/data/dataset.py
```


## 数据格式

### 输入数据格式

数据应为 JSON/JSONL/Parquet 格式，每条记录包含:

```json
{
  "question": "数学问题或推理问题",
  "original_cot": "原始的 Chain-of-Thought 推理过程，可能包含错误尝试和回溯",
  "answer": "正确答案"
}
```

### 数据示例

```json
{
  "question": "What is 15 + 27?",
  "original_cot": "Let me think... 15 plus 27. First, I'll try 15 + 20 = 35. Wait, that's not right because I need to add 27, not 20. Let me start over. 15 + 27: I can break it down as 15 + 25 + 2 = 40 + 2 = 42. Yes, that's correct!",
  "answer": "42"
}
```

### 准备数据

```bash
# 从示例数据创建训练集
python src/data/dataset.py

# 或从自定义数据创建
python -c "
from src.data.dataset import prepare_dataset
prepare_dataset(
    input_path='your_data.json',
    output_dir='data',
    train_ratio=0.9,
)
"
```

## 模型准备

下载所需模型到 `models/` 目录:

```bash
mkdir -p models

# Actor Model (参与 GRPO 训练)
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B

# Student Model (计算认知奖励)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct

# FSF Judge Model (计算逻辑奖励)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

## 训练

### 4 x A100-80G 配置 (全参数微调)

```bash
bash scripts/run_train.sh examples/qwen3_8b_grpo_4gpu.yaml
```

### 2 x A100-80G 配置 (LoRA 微调)

```bash
bash scripts/run_train.sh examples/qwen3_8b_grpo_2gpu_lora.yaml
```

### 手动启动 (分步执行)

```bash
# 1. 启动 vLLM 服务
bash scripts/start_vllm_services.sh both

# 2. 等待服务就绪后启动训练
SKIP_VLLM=true bash scripts/run_train.sh examples/qwen3_8b_grpo_4gpu.yaml
```

### 停止 vLLM 服务

```bash
bash scripts/stop_vllm_services.sh
```

## 配置说明

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `actor_rollout_ref.model.path` | Actor 模型路径 | `./models/Qwen3-8B` |
| `actor_rollout_ref.model.lora` | LoRA 配置 (null 表示全参数) | `null` |
| `data.train_batch_size` | 训练批次大小 | 32 |
| `actor_rollout_ref.rollout.n` | GRPO 每个 prompt 采样数 | 8 |
| `reward_model.custom_reward_function.reward_kwargs.lambda_logic` | 逻辑奖励权重 | 0.5 |
| `trainer.total_epochs` | 总训练轮数 | 3 |

### LoRA 配置示例

```yaml
actor_rollout_ref:
  model:
    lora:
      rank: 32
      alpha: 64
      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
      dropout: 0.05
```

### 禁用逻辑奖励

如果只想使用认知奖励:

```yaml
reward_model:
  custom_reward_function:
    reward_kwargs:
      use_logic_reward: false
```

## 输出目录结构

训练输出保存在带时间戳的目录中，避免覆盖:

```
outputs/
└── cot_rewriter_grpo/
    └── qwen3_8b_4xA100/
        └── version_20240115_143022/
            ├── checkpoints/
            │   ├── step_100/
            │   ├── step_200/
            │   └── ...
            ├── logs/
            └── config.yaml
```

## 项目结构

```
cot_rewriter_grpo/
├── pyproject.toml          # uv/pip 依赖配置
├── README.md               # 项目文档
├── examples/               # 训练配置文件
│   ├── qwen3_8b_grpo_4gpu.yaml      # 4卡全参数配置
│   └── qwen3_8b_grpo_2gpu_lora.yaml # 2卡 LoRA 配置
├── scripts/                # 启动脚本
│   ├── run_train.sh               # 训练启动脚本
│   ├── start_vllm_services.sh     # vLLM 服务启动
│   └── stop_vllm_services.sh      # vLLM 服务停止
├── src/                    # 源代码
│   ├── data/              # 数据处理
│   │   ├── dataset.py     # 数据集类
│   │   └── __init__.py
│   ├── reward/            # 奖励函数
│   │   ├── student_reward.py   # 认知奖励 (R_cog)
│   │   ├── fsf_reward.py       # 逻辑奖励 (R_logic)
│   │   ├── combined_reward.py  # 组合奖励 (VERL 接口)
│   │   └── __init__.py
│   └── trainer/           # 训练入口
│       ├── train.py       # 训练脚本
│       └── __init__.py
└── data/                   # 数据目录
    ├── examples/          # 示例数据
    │   └── sample_data.json
    ├── train.parquet      # 训练数据
    └── val.parquet        # 验证数据
```

## 常见问题

### Q: 显存不足怎么办?

A: 尝试以下方法:
1. 使用 LoRA 配置 (`examples/qwen3_8b_grpo_2gpu_lora.yaml`)
2. 减少 `data.train_batch_size`
3. 减少 `actor_rollout_ref.rollout.n` (GRPO 采样数)
4. 降低 `actor_rollout_ref.rollout.gpu_memory_utilization`

### Q: vLLM 服务启动失败?

A: 检查:
1. 模型路径是否正确
2. GPU 是否可用 (`nvidia-smi`)
3. 端口是否被占用 (`lsof -i:8100`)

### Q: 训练过程中奖励值异常?

A: 可能原因:
1. vLLM 服务未正确启动
2. Student/FSF 模型路径错误
3. 数据格式不符合要求

## License

Apache-2.0
