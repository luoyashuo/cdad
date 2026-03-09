# cdad


# Installation

## 1. 创建虚拟环境
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 安装基础依赖
```bash
uv pip install -r requirements.txt
```

## 3. 安装难编译的依赖（需要特殊处理）

### PyTorch（CUDA 12.4 版本）
```bash
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
### FlashAttention（GPU 加速库）
```bash
# 方法1：直接安装预编译的 wheel（推荐）
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 方法2：从源代码编译（较慢，约10-30分钟）
# uv pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### FlashInfer（快速推理库）
```bash
uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6
```

### vLLM（高性能推理引擎）
```bash
uv pip install vllm==0.8.5.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 其他特殊依赖
```bash
uv pip install git+https://github.com/NICTA/pyairports.git
uv pip install transformers==4.55.0 accelerate==1.11.0 deepspeed==0.16.9
uv pip install swanlab
```

## 4. 安装子模块依赖

### Evalscope
```bash
uv pip install -e ./eval/evalscope
```

# Evaluation 临时文件配置

评测时的沙箱临时文件存储在 `/mnt/moonfs/luoyashuo-ksyun/tmp/evalscope_sandbox`，
通过 `eval_configs/eval_cg.yaml` 中的 `sandbox.work_dir` 配置。


# 更正关键package
uv pip install trl==0.24.0
uv pip install transformers==4.55.0

# 已知问题修复

## `pkg_resources` 缺失错误

如果遇到 `ModuleNotFoundError: No module named 'pkg_resources'`，是因为 `setuptools>=70` 已移除 `pkg_resources`。

**解决方案**：降级 setuptools 到兼容版本
```bash
uv pip install 'setuptools<70'
```

已在 `requirements.txt` 中锁定 `setuptools<70` 以避免此问题。