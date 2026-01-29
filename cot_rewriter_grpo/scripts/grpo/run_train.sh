#!/bin/bash

# ============================================================================
# CoT Rewriter GRPO Training Launch Script
# ============================================================================
# 
# Usage:
#   bash scripts/grpo/run_train.sh                    # Use default config
#   bash scripts/grpo/run_train.sh <config_name>      # Use specific config
#   bash scripts/grpo/run_train.sh <config_name> <gpus>  # Specify GPUs
#
# Examples:
#   bash scripts/grpo/run_train.sh
#   bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu
#   bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu 1,2,3,5
#
# GPU Allocation (default: 1,2,3,5):
#   GPU 1,2,3: Actor Model + Rollout Engine (training)
#   GPU 5:     Reward Models (Student + FSF Judge)
#
# ============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 配置参数
CONFIG_NAME="${1:-qwen3_8b_grpo_4gpu}"
CUDA_DEVICES="${2:-1,2,3,5}"

# 路径配置
VENV_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate"
CONFIG_PATH="${PROJECT_ROOT}/examples/${CONFIG_NAME}.yaml"
LOG_DIR="${PROJECT_ROOT}/logs/grpo"
LOG_FILE="${LOG_DIR}/train_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 检查配置文件
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found: ${CONFIG_PATH}"
    echo "Available configs:"
    ls -1 "${PROJECT_ROOT}/examples/"*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

echo "========================================================================"
echo "CoT Rewriter GRPO Training"
echo "========================================================================"
echo "Config:    ${CONFIG_NAME}"
echo "GPUs:      ${CUDA_DEVICES}"
echo "Log:       ${LOG_FILE}"
echo "========================================================================"

# 激活虚拟环境
if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
else
    echo "Warning: Virtual environment not found at ${VENV_PATH}"
    echo "Trying to use system Python..."
fi

# 进入项目目录
cd "${PROJECT_ROOT}"

# 设置环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/../verl:${PYTHONPATH}"

# Ray 配置
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0

# NCCL 配置
export NCCL_DEBUG=WARN

# Tokenizer 配置
export TOKENIZERS_PARALLELISM=false

# 显示配置信息
echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo ""

# 检查 GPU 可用性
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | while read line; do
    echo "  GPU: ${line}"
done
echo ""

# 启动训练
echo "Starting training..."
echo "Command: python src/trainer/train.py --config-name=${CONFIG_NAME}"
echo ""

# 运行训练
python src/trainer/train.py --config-name="${CONFIG_NAME}" 2>&1 | tee "${LOG_FILE}"

TRAIN_STATUS=${PIPESTATUS[0]}

if [ ${TRAIN_STATUS} -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Training completed successfully!"
    echo "Log saved to: ${LOG_FILE}"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "Training failed with exit code: ${TRAIN_STATUS}"
    echo "Log saved to: ${LOG_FILE}"
    echo "========================================================================"
    exit ${TRAIN_STATUS}
fi
