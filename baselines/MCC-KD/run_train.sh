#!/bin/bash
# MCC-KD 训练启动脚本（8 GPU 分布式训练）

set -e

echo "===== MCC-KD Training ====="
echo "Model: Qwen2.5-7B-Instruct"
echo "GPUs: 8 x 40GB"
echo "Method: LoRA + DeepSpeed ZeRO-3"

# 环境变量设置
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CUDA 内存优化设置
# 使用可扩展内存段来减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 设置 CUDA 架构（A100 = 8.0），避免重复编译 CUDA 扩展
export TORCH_CUDA_ARCH_LIST="8.0"

# 优化多进程数据加载
export OMP_NUM_THREADS=4

# DeepSpeed 优化（可选）
export DS_SKIP_CUDA_CHECK=0

# 检查数据是否存在
if [ ! -f "/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl" ]; then
    echo "Error: Processed dataset not found!"
    echo "Please run run_preprocess.sh first."
    exit 1
fi

# 默认使用优化后的配置
#CONFIG_FILE="${1:-./configs/training_config_qwen2_5_7B.yaml}"   done
#CONFIG_FILE="${1:-./configs/training_config_llama_1B.yaml}"
CONFIG_FILE="${1:-./configs/training_config_llama_3B.yaml}"

echo "Using config: $CONFIG_FILE"

# 使用 torchrun 启动分布式训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_mcckd.py \
    --config "$CONFIG_FILE"

echo "Training completed!"
