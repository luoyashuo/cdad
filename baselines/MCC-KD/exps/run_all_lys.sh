#!/bin/bash

# MCC-KD 顺序训练脚本（llama1b -> llama3b）
# 参考 run_train.sh 和 run_exps.sh

set -e

cd /home/luoys/qzq/mcc-deepmath
mkdir -p logs

# 提取公共环境变量
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA 内存优化设置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 设置 CUDA 架构（A100 = 8.0）
export TORCH_CUDA_ARCH_LIST="8.0"

# 优化多进程数据加载
export OMP_NUM_THREADS=4

# DeepSpeed 优化
export DS_SKIP_CUDA_CHECK=0

# 日志目录
LOG_DIR="logs"

echo "===== MCC-KD 顺序训练任务 ====="
echo "顺序: llama1b -> llama3b"
echo "GPUs: 8 x 40GB"
echo "日志目录: ${LOG_DIR}/"
echo ""

# 训练任务列表
declare -a CONFIGS=(
    "./configs/training_config_llama_1B.yaml"
    "./configs/training_config_llama_3B.yaml"
)

declare -a NAMES=(
    "llama_1B"
    "llama_3B"
)

declare -a MODEL_NAMES=(
    "Llama-1B"
    "Llama-3B"
)

TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    LOG_FILE="${LOG_DIR}/${NAME}_$(date '+%Y%m%d_%H%M%S').log"
    NUM=$((i + 1))
    
    echo "[$NUM/$TOTAL] 开始训练: $MODEL_NAME"
    echo "    配置文件: $CONFIG"
    echo "    日志文件: $LOG_FILE"
    echo "    开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 使用 torchrun 启动分布式训练，同时输出到终端和日志
    torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_port=29500 \
        train_mcckd.py \
        --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "    ✓ 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    else
        echo ""
        echo "    ✗ 训练失败，退出码: $EXIT_CODE"
        echo "    请检查日志: $LOG_FILE"
        echo ""
        exit $EXIT_CODE
    fi
done

echo "===== 所有训练任务已完成 ====="
