#!/bin/bash
#
# 使用方法:
#   bash scripts/grpo/start_a100.sh              # 默认使用 GPU 0-7
#   bash scripts/grpo/start_a100.sh 6,7          # 指定 GPU
#   bash scripts/grpo/start_a100.sh 0,1,2,3 qwen2_5_7b_grpo_8x40g  # 指定 GPU 和配置

set -e

# ========================================
# QC 环境路径配置 - 根据实际情况修改
# ========================================
export MODEL_BASE_DIR="/data/modelscope_cache"
export DATA_BASE_DIR="/data/hf_home/datasets/Yaseal/CDAD_EXP"
export CHECKPOINT_BASE_DIR="/data/cdad/checkpoints/cot_rewriter"
# ========================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PROJECT_BASE_DIR="${PROJECT_ROOT}"

# 默认配置

DEFAULT_GPUS="0,1,2,3,4,5,6,7"
DEFAULT_CONFIG="qwen2_5_7b_grpo_8x40g"
# 解析参数
GPUS="${1:-$DEFAULT_GPUS}"
CONFIG_NAME="${2:-$DEFAULT_CONFIG}"

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
