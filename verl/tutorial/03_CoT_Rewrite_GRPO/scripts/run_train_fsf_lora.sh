#!/bin/bash
# FSF LoRA 训练脚本
#
# 使用方法:
#   bash scripts/run_train_fsf_lora.sh
#
# 步骤:
# 1. 首先使用现有的 FSF Judge 生成训练数据
# 2. 然后使用 PEFT 训练 LoRA adapter

set -e

# 配置
MODEL_PATH="./models/Qwen2.5-3B-Instruct"
TRAIN_DATA="./data/fsf_train.jsonl"
OUTPUT_DIR="./outputs/fsf_lora"
REASONING_DATA="./data/reasoning_samples.jsonl"  # 原始推理数据
FSF_API_URL="http://localhost:8000/v1/chat/completions"  # FSF Judge API

# 创建输出目录
mkdir -p $(dirname ${TRAIN_DATA})
mkdir -p ${OUTPUT_DIR}

# 步骤1: 生成训练数据（如果不存在）
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "生成 FSF 训练数据..."
    python scripts/train_fsf_lora.py \
        --model_path ${MODEL_PATH} \
        --train_data ${TRAIN_DATA} \
        --output_dir ${OUTPUT_DIR} \
        --generate_data \
        --reasoning_data ${REASONING_DATA} \
        --fsf_api_url ${FSF_API_URL}
fi

# 步骤2: 训练 FSF LoRA
echo "开始训练 FSF LoRA..."
python scripts/train_fsf_lora.py \
    --model_path ${MODEL_PATH} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05

echo "FSF LoRA 训练完成！"
echo "LoRA adapter 保存在: ${OUTPUT_DIR}"
