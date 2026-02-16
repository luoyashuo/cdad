#!/bin/bash
# QC 环境 GRPO 训练启动脚本 (2x 40GB GPU)
#
# 使用方法:
#   bash scripts/grpo/start_qc.sh              # 默认使用 GPU 4,5
#   bash scripts/grpo/start_qc.sh 6,7          # 指定 GPU
#   bash scripts/grpo/start_qc.sh 4,5 qwen3_4b_grpo_2x40g_lora_sglang  # 指定 GPU 和配置

set -e

# ========================================
# QC 环境路径配置 - 根据实际情况修改
# ========================================
export MODEL_BASE_DIR="/data/luoys/modelscope_cache/models"
export DATA_BASE_DIR="/data/luoys/data/cdad"
export CHECKPOINT_BASE_DIR="/data/luoys/checkpoints/cdad"
# ========================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PROJECT_BASE_DIR="${PROJECT_ROOT}"

# 默认配置
# DEFAULT_GPUS="0,1"
# DEFAULT_CONFIG="qwen3_4b_grpo_2x80g_vllm_rollout_sglang_reward"


DEFAULT_GPUS="0,1,2,3,4,5,6,7"
# DEFAULT_GPUS="0,1,2,3"
DEFAULT_CONFIG="llama3_8b_grpo_8x80g"
# 解析参数
GPUS="${1:-$DEFAULT_GPUS}"
CONFIG_NAME="${2:-$DEFAULT_CONFIG}"

echo "============================================================"
echo "CoT Rewriter GRPO Training (QC 环境)"
echo "============================================================"
echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${GPUS}"
echo ""
echo "Path Configuration:"
echo "  MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "  DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "  CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "  PROJECT_BASE_DIR=${PROJECT_BASE_DIR}"
echo "============================================================"
echo ""

# 检查路径是否存在
if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo "Error: MODEL_BASE_DIR 不存在: $MODEL_BASE_DIR"
    exit 1
fi

if [ ! -d "$DATA_BASE_DIR" ]; then
    echo "Error: DATA_BASE_DIR 不存在: $DATA_BASE_DIR"
    exit 1
fi

# 创建检查点目录
mkdir -p "$CHECKPOINT_BASE_DIR"

# 启动训练
exec bash "${SCRIPT_DIR}/run_train.sh" "${CONFIG_NAME}" "${GPUS}"
