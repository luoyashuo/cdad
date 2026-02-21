#!/bin/bash
# Llama-3.2-3B GRPO 训练启动脚本 (GSM8K + MATH)
#
# 使用方法:
#   bash scripts/grpo/start_llama32_math.sh                    # 默认: 8x GPU, 中等配置
#   bash scripts/grpo/start_llama32_math.sh 0,1,2,3            # 指定 GPU
#   bash scripts/grpo/start_llama32_math.sh 0,1 safe           # 使用保守配置(防OOM)
#   bash scripts/grpo/start_llama32_math.sh 0,1 medium         # 使用中等配置
#   bash scripts/grpo/start_llama32_math.sh 0,1 large          # 使用大batch配置
#
# 可用配置:
#   - safe:   保守配置，train_batch_size=128,  适合 8x40GB 或追求稳定
#   - medium: 中等配置，train_batch_size=512,  适合 8x40GB/80GB (推荐)
#   - large:  大batch配置，train_batch_size=1024, 适合 8x80GB
#
# 数据准备:
#   bash scripts/grpo/download_math_datasets.sh

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
DEFAULT_GPUS="0,1,2,3,4,5,6,7"

# 解析参数
GPUS="${1:-$DEFAULT_GPUS}"
CONFIG_TYPE="${2:-s}"

# 根据配置类型选择配置文件
case "$CONFIG_TYPE" in
  safe|s|small)
    CONFIG_NAME="llama32_3b_grpo_math"
    echo "使用配置: 保守配置 (safe) - batch_size=128"
    ;;
  medium|m)
    CONFIG_NAME="llama32_3b_grpo_math_medium"
    echo "使用配置: 中等配置 (medium) - batch_size=512 (推荐)"
    ;;
  large|l|aggressive)
    CONFIG_NAME="llama32_3b_grpo_math"
    echo "使用配置: 大batch配置 (large) - batch_size=1024 (需要 8x80GB)"
    ;;
  *)
    CONFIG_NAME="$CONFIG_TYPE"
    echo "使用配置: 自定义配置 - $CONFIG_TYPE"
    ;;
esac

echo "============================================================"
echo "Llama-3.2-3B GRPO Training (GSM8K + MATH)"
echo "============================================================"
echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${GPUS}"
echo ""
echo "Batch Size 参考:"
echo "  - safe:   train_batch_size=128,  micro_batch=4   (适合 8x40GB)"
echo "  - medium: train_batch_size=512,  micro_batch=16  (适合 8x40GB/80GB)"
echo "  - large:  train_batch_size=1024, micro_batch=40  (需要 8x80GB)"
echo ""
echo "Path Configuration:"
echo "  MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "  DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "  CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "  PROJECT_BASE_DIR=${PROJECT_BASE_DIR}"
echo "============================================================"
echo ""

# 检查数据集是否存在
if [ ! -f "$DATA_BASE_DIR/gsm8k/train.parquet" ]; then
    echo "Warning: GSM8K 训练数据不存在: $DATA_BASE_DIR/gsm8k/train.parquet"
    echo "请先下载数据集，参考:"
    echo "  python -c \"from datasets import load_dataset; ds = load_dataset('gsm8k', 'main'); ds['train'].to_parquet('$DATA_BASE_DIR/gsm8k/train.parquet'); ds['test'].to_parquet('$DATA_BASE_DIR/gsm8k/test.parquet')\""
fi

if [ ! -f "$DATA_BASE_DIR/math/train.parquet" ]; then
    echo "Warning: MATH 训练数据不存在: $DATA_BASE_DIR/math/train.parquet"
    echo "请先下载数据集，参考:"
    echo "  python -c \"from datasets import load_dataset; ds = load_dataset('hendrycks/competition_math'); ds['train'].to_parquet('$DATA_BASE_DIR/math/train.parquet'); ds['test'].to_parquet('$DATA_BASE_DIR/math/test.parquet')\""
fi

# 检查路径是否存在
if [ ! -d "$MODEL_BASE_DIR" ]; then
    echo "Error: MODEL_BASE_DIR 不存在: $MODEL_BASE_DIR"
    exit 1
fi

# 创建检查点目录
mkdir -p "$CHECKPOINT_BASE_DIR"

# 启动训练
exec bash "${SCRIPT_DIR}/run_train.sh" "${CONFIG_NAME}" "${GPUS}"
