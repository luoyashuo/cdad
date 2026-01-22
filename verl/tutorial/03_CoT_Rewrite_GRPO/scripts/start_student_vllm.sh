#!/bin/bash
# 启动 Student Model vLLM 服务
#
# 两种模式：
# 1. 基础模式：只有 Student Model
# 2. LoRA 模式：Student Model + FSF LoRA
#
# 使用方法：
#   bash scripts/start_student_vllm.sh [base|lora]

set -e

MODE=${1:-base}
MODEL_PATH="${MODEL_PATH:-./models/Qwen2.5-3B-Instruct}"
FSF_LORA_PATH="${FSF_LORA_PATH:-./outputs/fsf_lora}"
PORT="${PORT:-8100}"
GPU_ID="${GPU_ID:-3}"  # 默认使用 GPU 3

echo "=========================================="
echo "启动 Student Model vLLM 服务"
echo "模式: $MODE"
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU:  $GPU_ID"
echo "=========================================="

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

if [ "$MODE" == "lora" ]; then
    echo "启用 FSF LoRA: $FSF_LORA_PATH"

    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --enable-lora \
        --lora-modules "fsf_lora=$FSF_LORA_PATH" \
        --max-lora-rank 64 \
        --trust-remote-code \
        --dtype bfloat16
else
    echo "基础模式（无 LoRA）"

    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --dtype bfloat16
fi
