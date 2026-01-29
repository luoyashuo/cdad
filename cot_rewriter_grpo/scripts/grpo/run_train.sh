#!/bin/bash

# CoT Rewriter GRPO 训练启动脚本
# 基于配置文件: examples/qwen3_8b_grpo_4gpu.yaml
#
# 硬件配置:
#   - GPU 1-3: Actor Model (Qwen3-8B) + Rollout Engine (训练过程)
#   - GPU 5:   Reward Models (Student + FSF Judge, 独立资源池)
#
# 使用方法:
#   bash scripts/grpo/run_train.sh                    # 使用默认配置
#   bash scripts/grpo/run_train.sh <config_name>      # 指定配置文件
#   bash scripts/grpo/run_train.sh <config_name> <gpus> # 指定配置和GPU
#
# 示例:
#   bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu 1,2,3,5

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认配置
DEFAULT_CONFIG="qwen3_8b_grpo_4gpu"
DEFAULT_GPUS="1,2,3,5"

# 解析参数
CONFIG_NAME="${1:-$DEFAULT_CONFIG}"
CUDA_VISIBLE_DEVICES="${2:-$DEFAULT_GPUS}"

# 配置文件路径
CONFIG_FILE="examples/${CONFIG_NAME}.yaml"

# 日志配置
LOG_DIR="${PROJECT_ROOT}/logs/grpo"
LOG_FILE="${LOG_DIR}/train_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "CoT Rewriter GRPO Training"
echo "============================================================"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

# 检查配置文件是否存在
if [ ! -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${PROJECT_ROOT}/${CONFIG_FILE}"
    echo "Available configs:"
    ls -1 "${PROJECT_ROOT}/examples/"*.yaml 2>/dev/null || echo "  No yaml files found in examples/"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# 强制 Ray 创建新的本地集群，不连接现有集群（避免权限问题）
export RAY_ADDRESS=local

# 可选: 设置 Python 路径
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo ""

# 激活虚拟环境 (如果存在)
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/../.venv/bin/activate" ]; then
    echo "Activating parent directory virtual environment..."
    source "${PROJECT_ROOT}/../.venv/bin/activate"
fi

# 检查必要的 Python 包
echo "Checking dependencies..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  Warning: PyTorch not found"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || echo "  Warning: Transformers not found"
python -c "import ray; print(f'  Ray: {ray.__version__}')" 2>/dev/null || echo "  Warning: Ray not found"

echo ""
echo "Starting training..."
echo "  Config: ${CONFIG_FILE}"
echo "  Output log: ${LOG_FILE}"
echo ""

# 进入项目目录
cd "${PROJECT_ROOT}"

# 启动训练
# 方式1: 直接运行 Python (使用 Hydra 配置)
python src/trainer/train.py --config-name="${CONFIG_NAME}" 2>&1 | tee "${LOG_FILE}"

# 方式2: 使用 torchrun (如果需要多进程)
# torchrun --standalone --nnodes=1 --nproc_per_node=4 \
#     src/trainer/train.py --config-name="${CONFIG_NAME}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "Training completed!"
echo "Log saved to: ${LOG_FILE}"
echo "============================================================"
