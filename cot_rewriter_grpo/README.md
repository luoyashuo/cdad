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
uv pip install vllm==0.8.3
uv pip install sglang[all]==0.4.6.post5
uv pip install transformers==4.55.0
uv pip install zmq
uv pip install swanlab

uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

uv pip install opentelemetry-api==1.34.1 opentelemetry-sdk==1.34.1 opentelemetry-exporter-otlp==1.34.1
uv pip install loguru

```


#### 环境问题
1. RUC服务器有：NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.4
  NVIDIA驱动跟不上CUDA驱动的问题，建议升级大535+
2. RUC服务器上VLLM运行必须开：
  - VLLM_USE_V1=0 - 强制使用 vLLM V0 引擎                                                                                                                    
  - VLLM_ATTENTION_BACKEND=FLASH_ATTN - 使用 Flash Attention 后端 
  - FSDP必须用fp16，不能用bf16:
  ```
    fsdp_config:
      dtype: float16
      mixed_precision:
        param_dtype: fp16
        reduce_dtype: fp16
        buffer_dtype: fp16
    ```
3. VLLM SLEEP失效：
  - https://github.com/vllm-project/vllm-ascend/issues/977
  - https://github.com/vllm-project/vllm/issues/15254