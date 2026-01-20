#!/bin/bash
# ============================================================================
# 顺序训练脚本 - 4种策略 (使用2个GPU)
# ============================================================================

set -e

BASE_DIR="/mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad"
CONFIG_DIR="$BASE_DIR/LlamaFactory/examples/strategies"
LOG_DIR="$BASE_DIR/logs/strategies"

# 创建日志目录
mkdir -p $LOG_DIR

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1

# Patch applied via sitecustomize.py in site-packages

echo "=============================================="
echo "  顺序训练 4 种数据选择策略"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# 1. CD Optimal 训练
echo ""
echo "[1/4] 训练 CD Optimal 模型..."
echo "开始时间: $(date)"
llamafactory-cli train $CONFIG_DIR/qwen3_1_7b_cd_optimal.yaml 2>&1 | tee $LOG_DIR/train_cd_optimal.log
echo "完成时间: $(date)"

# 2. RV Optimal 训练
echo ""
echo "[2/4] 训练 RV Optimal 模型..."
echo "开始时间: $(date)"
llamafactory-cli train $CONFIG_DIR/qwen3_1_7b_rv_optimal.yaml 2>&1 | tee $LOG_DIR/train_rv_optimal.log
echo "完成时间: $(date)"

# 3. Combined 训练
echo ""
echo "[3/4] 训练 Combined 模型..."
echo "开始时间: $(date)"
llamafactory-cli train $CONFIG_DIR/qwen3_1_7b_combined.yaml 2>&1 | tee $LOG_DIR/train_combined.log
echo "完成时间: $(date)"

# 4. Combined + FSF 训练
echo ""
echo "[4/4] 训练 Combined + FSF 模型..."
echo "开始时间: $(date)"
llamafactory-cli train $CONFIG_DIR/qwen3_1_7b_combined_fsf.yaml 2>&1 | tee $LOG_DIR/train_combined_fsf.log
echo "完成时间: $(date)"

echo ""
echo "=============================================="
echo "  所有训练完成！"
echo "=============================================="
echo "模型保存位置:"
echo "  1. /mnt/moonfs/luoyashuo-ksyun/research/cdad/ckpt/strategies/qwen3_1_7b_base_sft_cd_optimal"
echo "  2. /mnt/moonfs/luoyashuo-ksyun/research/cdad/ckpt/strategies/qwen3_1_7b_base_sft_rv_optimal"
echo "  3. /mnt/moonfs/luoyashuo-ksyun/research/cdad/ckpt/strategies/qwen3_1_7b_base_sft_combined"
echo "  4. /mnt/moonfs/luoyashuo-ksyun/research/cdad/ckpt/strategies/qwen3_1_7b_base_sft_combined_fsf"
echo ""
echo "日志位置: $LOG_DIR"
# nohup bash examples/strategies/run.sh > ../logs/strategies/train_all.log 2>&1 &