#!/bin/bash
# 2x A100-80G GPU GRPO 训练启动脚本
# FSDP1 配置：不卸载 optimizer（保留 param_offload）
#
# 使用方法:
#   bash scripts/grpo/start.sh              # 默认使用 GPU 0,1
#   bash scripts/grpo/start.sh 4,5          # 指定 GPU

set -e

# ========================================
# 路径配置 - 根据实际情况修改
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
DEFAULT_GPUS="0,1"
CONFIG_NAME="qwen3_4b_grpo_2x80g_vllm_rollout_sglang_reward"

# 解析参数
GPUS="${1:-$DEFAULT_GPUS}"

echo "============================================================"
echo "CoT Rewriter GRPO Training (2x A100-80G GPU)"
echo "============================================================"
echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${GPUS}"
echo ""
echo "FSDP1 配置: 不卸载 optimizer"
echo "  - param_offload: true"
echo "  - optimizer_offload: false"
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
