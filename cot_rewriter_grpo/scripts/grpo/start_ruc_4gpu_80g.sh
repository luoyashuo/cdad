#!/bin/bash
# RUC 环境 GRPO 训练启动脚本 (4x 80GB GPU)
#
# 使用方法:
#   bash scripts/grpo/start_ruc_4gpu_80g.sh              # 默认使用 GPU 0,1,2,3
#   bash scripts/grpo/start_ruc_4gpu_80g.sh 4,5,6,7      # 指定 GPU

set -e

# ========================================
# RUC 环境路径配置 - 根据实际情况修改
# ========================================
export MODEL_BASE_DIR="/fs/fast/u20240075/models"
export DATA_BASE_DIR="/fs/fast/u20240075/data"
export CHECKPOINT_BASE_DIR="/fs/fast/u20240075/checkpoints"
# ========================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PROJECT_BASE_DIR="${PROJECT_ROOT}"

# 默认配置
DEFAULT_GPUS="0,1,2,3"
CONFIG_NAME="qwen3_4b_grpo_ruc_4x80g_full_sglang"

# 解析参数
GPUS="${1:-$DEFAULT_GPUS}"

echo "============================================================"
echo "CoT Rewriter GRPO Training (RUC 4x80G 环境)"
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
    echo "Warning: MODEL_BASE_DIR 不存在: $MODEL_BASE_DIR"
    echo "请确认模型路径是否正确"
fi

if [ ! -d "$DATA_BASE_DIR" ]; then
    echo "Warning: DATA_BASE_DIR 不存在: $DATA_BASE_DIR"
    echo "请确认数据路径是否正确"
fi

# 创建检查点目录
mkdir -p "$CHECKPOINT_BASE_DIR"

# 后台启动训练并记录 PID
LOG_FILE="${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

nohup bash "${SCRIPT_DIR}/run_train.sh" "${CONFIG_NAME}" "${GPUS}" > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "============================================================"
echo "训练已后台启动"
echo "  Config: ${CONFIG_NAME}"
echo "  GPUs: ${GPUS}"
echo "  PID: ${TRAIN_PID}"
echo ""
echo "日志文件: ${LOG_FILE}"
echo ""
echo "查看日志: tail -f ${LOG_FILE}"
echo "停止训练: kill ${TRAIN_PID}"
echo "============================================================"
