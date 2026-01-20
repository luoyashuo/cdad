# Verl 环境配置指南

> 本章节介绍如何使用 uv + CUDA 12.4 搭建 Verl 训练环境，包括依赖安装、推理引擎配置等。

## 1. 硬件要求

### 1.1 最低配置

| 组件 | 规格 |
|------|------|
| GPU | 4 × A100 (40G) 或更高 |
| CPU 内存 | 128 GB |
| 磁盘 | 500 GB SSD (用于模型和数据) |
| CUDA | 12.1+ |

### 1.2 推荐配置 (本教程使用)

| 组件 | 规格 |
|------|------|
| GPU | 4 × A100 (80G) |
| CPU 内存 | 256 GB |
| 磁盘 | 1 TB NVMe SSD |
| CUDA | 12.4 |

## 2. 使用 uv 管理 Python 环境

[uv](https://github.com/astral-sh/uv) 是 Rust 编写的极速 Python 包管理器，比 pip 快 10-100 倍。

### 2.1 安装 uv

```bash
# 使用 pip 安装
pip install uv

# 或使用官方脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 创建虚拟环境

```bash
# 创建项目目录
mkdir -p ~/projects/verl_grpo_practice
cd ~/projects/verl_grpo_practice

# 使用 uv 创建虚拟环境 (Python 3.10)
uv venv --python 3.10

# 激活虚拟环境
source .venv/bin/activate
```

### 2.3 配置 pyproject.toml

创建 `pyproject.toml` 来管理依赖：

```toml
[project]
name = "verl_grpo_practice"
version = "0.1.0"
description = "CoT Rewriter Training with GRPO"
requires-python = ">=3.10,<3.12"

dependencies = [
    # Verl 核心
    "verl>=0.3.0",

    # PyTorch (CUDA 12.4)
    "torch>=2.4.0",
    "torchvision>=0.19.0",

    # Transformers 生态
    "transformers>=4.45.0",
    "accelerate>=0.34.0",
    "peft>=0.13.0",        # LoRA 支持
    "datasets>=3.0.0",
    "tokenizers>=0.20.0",

    # 推理引擎 (选择其一)
    "vllm>=0.6.0",
    # "sglang>=0.3.0",     # 或者使用 SGLang

    # 分布式训练
    "ray[default]>=2.35.0",

    # 配置管理
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.0",

    # 日志和监控
    "wandb>=0.18.0",
    "tensorboard>=2.18.0",

    # 工具库
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.6.0",
]

[tool.uv]
# 使用 PyTorch 官方源
extra-index-url = [
    "https://download.pytorch.org/whl/cu124"
]
```

### 2.4 安装依赖

```bash
# 使用 uv 安装所有依赖
uv pip install -e .

# 安装开发依赖
uv pip install -e ".[dev]"
```

## 3. 安装 Verl

### 3.1 从 PyPI 安装 (稳定版)

```bash
uv pip install verl
```

### 3.2 从源码安装 (最新版)

```bash
# 克隆仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 安装
uv pip install -e .

# 安装 Megatron-LM 支持 (可选)
uv pip install -e ".[megatron]"
```

### 3.3 验证安装

```python
# 验证 Verl 安装
python -c "import verl; print(verl.__version__)"

# 验证 CUDA 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 验证 Ray
python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"
```

## 4. 推理引擎配置

Verl 支持 vLLM 和 SGLang 两种推理引擎，用于高效的 Rollout。

### 4.1 vLLM 配置

```bash
# 安装 vLLM (CUDA 12.4)
uv pip install vllm>=0.6.0

# 验证
python -c "from vllm import LLM; print('vLLM OK')"
```

**vLLM 在 Verl 中的配置：**

```yaml
# config.yaml
actor_rollout_ref:
  rollout:
    name: vllm  # 使用 vLLM 作为推理引擎
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
    enforce_eager: false  # 是否禁用 CUDA Graphs
    max_model_len: 4096
```

### 4.2 SGLang 配置 (推荐)

SGLang 通常比 vLLM 更快，特别是在 multi-turn 场景：

```bash
# 安装 SGLang
uv pip install "sglang[all]>=0.3.0"

# 验证
python -c "import sglang; print('SGLang OK')"
```

**SGLang 在 Verl 中的配置：**

```yaml
# config.yaml
actor_rollout_ref:
  rollout:
    name: sglang  # 使用 SGLang 作为推理引擎
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
    mem_fraction_static: 0.8
```

### 4.3 推理引擎选择建议

| 场景 | 推荐引擎 | 原因 |
|------|----------|------|
| 单轮对话 | vLLM 或 SGLang | 两者性能相近 |
| 多轮对话 | SGLang | 更好的 prefix caching |
| Agent/Tool | SGLang | 支持 RadixAttention |
| 长上下文 (>32K) | SGLang | 更好的内存管理 |

## 5. 模型下载

### 5.1 使用 HuggingFace Hub

```bash
# 安装 huggingface_hub
uv pip install huggingface_hub

# 登录 (获取 gated model 权限)
huggingface-cli login

# 下载 Qwen-2.5-7B-Instruct
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    local_dir='./models/Qwen2.5-7B-Instruct',
    local_dir_use_symlinks=False
)
"
```

### 5.2 使用 ModelScope (国内加速)

```bash
# 安装 modelscope
uv pip install modelscope

# 下载模型
python -c "
from modelscope import snapshot_download
snapshot_download(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    cache_dir='./models'
)
"
```

### 5.3 模型目录结构

```
models/
├── Qwen2.5-7B-Instruct/     # Actor Model (可训练)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── Qwen2.5-3B-Instruct/     # Student Model (冻结，用于认知奖励)
    ├── config.json
    ├── model.safetensors
    └── ...
```

## 6. 数据准备

### 6.1 数据格式

Verl 使用 Parquet 格式存储训练数据：

```python
# 数据示例
{
    "prompt": "请分析以下数学问题的解题思路：...",
    "original_cot": "首先我们需要...",  # 原始 CoT
    "answer": "42",                      # 正确答案
    "uid": "math_001",                   # 唯一标识 (用于 GRPO 分组)
}
```

### 6.2 数据预处理脚本

```python
# scripts/prepare_data.py
import pandas as pd
from datasets import Dataset
import pyarrow.parquet as pq

def prepare_cot_rewrite_dataset(input_file: str, output_file: str):
    """
    准备 CoT 改写数据集

    输入: JSON Lines 文件
    输出: Parquet 文件
    """
    # 读取原始数据
    df = pd.read_json(input_file, lines=True)

    # 构造 prompt
    def create_prompt(row):
        return f"""你是一个 Chain-of-Thought 改写专家。
请将以下推理过程改写得更清晰、更有逻辑性。

原问题: {row['question']}
原始推理: {row['original_cot']}

请输出改写后的推理过程:"""

    df['prompt'] = df.apply(create_prompt, axis=1)

    # 添加 uid (用于 GRPO 分组)
    df['uid'] = df.index.astype(str)

    # 保留必要字段
    df = df[['prompt', 'original_cot', 'answer', 'uid', 'question']]

    # 保存为 Parquet
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")

if __name__ == "__main__":
    prepare_cot_rewrite_dataset(
        "data/raw/math_cot.jsonl",
        "data/processed/train.parquet"
    )
```

### 6.3 数据目录结构

```
data/
├── raw/
│   └── math_cot.jsonl           # 原始数据
└── processed/
    ├── train.parquet            # 训练集
    └── val.parquet              # 验证集
```

## 7. 环境变量配置

创建 `.env` 文件或在 shell 配置中设置：

```bash
# .env 或 ~/.bashrc

# CUDA 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.4

# PyTorch 配置
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A100/A6000/H100

# Ray 配置
export RAY_DEDUP_LOGS=1                     # 去重日志
export RAY_memory_monitor_refresh_ms=0      # 禁用内存监控 (加速)

# NCCL 配置 (分布式训练)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0                    # 启用 InfiniBand
export NCCL_NET_GDR_LEVEL=2

# Hugging Face 配置
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers

# 日志配置
export WANDB_PROJECT="verl_grpo_practice"
```

## 8. 验证完整环境

运行以下脚本验证环境：

```python
# scripts/verify_env.py
import sys

def check_python():
    version = sys.version_info
    assert version.major == 3 and version.minor >= 10, f"需要 Python 3.10+, 当前: {version}"
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")

def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA 不可用"
    print(f"[OK] CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"     GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

def check_verl():
    import verl
    print(f"[OK] Verl {verl.__version__}")

def check_transformers():
    import transformers
    print(f"[OK] Transformers {transformers.__version__}")

def check_vllm():
    try:
        import vllm
        print(f"[OK] vLLM {vllm.__version__}")
    except ImportError:
        print("[WARN] vLLM 未安装")

def check_sglang():
    try:
        import sglang
        print(f"[OK] SGLang {sglang.__version__}")
    except ImportError:
        print("[WARN] SGLang 未安装")

def check_ray():
    import ray
    ray.init(ignore_reinit_error=True)
    resources = ray.cluster_resources()
    print(f"[OK] Ray, Resources: {resources}")
    ray.shutdown()

def check_model_loading():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # 使用小模型测试
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"[INFO] 测试加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 测试推理
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    print(f"[OK] 模型推理测试通过: {tokenizer.decode(outputs[0], skip_special_tokens=True)[:50]}...")

    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=" * 50)
    print("Verl 环境验证")
    print("=" * 50)

    check_python()
    check_cuda()
    check_verl()
    check_transformers()
    check_vllm()
    check_sglang()
    check_ray()
    check_model_loading()

    print("=" * 50)
    print("环境验证完成!")
    print("=" * 50)
```

运行验证：

```bash
python scripts/verify_env.py
```

预期输出：

```
==================================================
Verl 环境验证
==================================================
[OK] Python 3.10.14
[OK] CUDA 12.4, 4 GPU(s)
     GPU 0: NVIDIA A100-SXM4-80GB (80.0 GB)
     GPU 1: NVIDIA A100-SXM4-80GB (80.0 GB)
     GPU 2: NVIDIA A100-SXM4-80GB (80.0 GB)
     GPU 3: NVIDIA A100-SXM4-80GB (80.0 GB)
[OK] Verl 0.3.0
[OK] Transformers 4.45.0
[OK] vLLM 0.6.0
[OK] SGLang 0.3.0
[OK] Ray, Resources: {'CPU': 64.0, 'GPU': 4.0, ...}
[OK] 模型推理测试通过: Hello, my name is John. I am a software engineer...
==================================================
环境验证完成!
==================================================
```

## 9. 常见问题

### 9.1 CUDA 版本不匹配

```bash
# 检查系统 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 如果不匹配，重新安装 PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 9.2 Ray 内存不足

```bash
# 增加 Ray 对象存储大小
export RAY_OBJECT_STORE_MEMORY=50000000000  # 50GB

# 或在代码中设置
ray.init(object_store_memory=50 * 1024**3)
```

### 9.3 vLLM/SGLang 安装失败

```bash
# 方法1: 使用预编译 wheel
uv pip install vllm --find-links https://github.com/vllm-project/vllm/releases

# 方法2: 从源码编译 (需要 CUDA 开发环境)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 9.4 模型加载 OOM

```python
# 使用量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config
)
```

## 10. 小结

本章介绍了 Verl GRPO 训练环境的完整配置：

1. **uv 包管理**: 使用 uv 快速管理 Python 环境和依赖
2. **Verl 安装**: 从 PyPI 或源码安装
3. **推理引擎**: 配置 vLLM 或 SGLang
4. **模型下载**: 从 HuggingFace 或 ModelScope 下载
5. **数据准备**: 准备 Parquet 格式的训练数据
6. **环境验证**: 验证完整环境可用

下一章我们将深入实现 CoT Rewriter 的 GRPO 训练逻辑，包括自定义奖励函数的设计。
