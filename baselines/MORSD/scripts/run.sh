#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/luoys/cdad/.venv/bin/python"

INPUT="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl"
OUTPUT="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl"

NUM_GPUS=8
MAX_MODEL_LEN=16384
GPU_MEM=0.92

MODELS=(
    "/data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct"
    "/data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
    "/data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct"
)

# ============================================================
# Step 1: 依次用每个学生模型计算 RD（增量追加，断点续传）
# ============================================================
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    echo "============================================================"
    echo " 计算 RD: ${MODEL_NAME}"
    echo " 模型路径: ${MODEL}"
    echo "============================================================"

    # 第一个模型从原始 input 读；后续模型从已有 output 读（增量追加）
    if [ -f "$OUTPUT" ]; then
        CURRENT_INPUT="$OUTPUT"
    else
        CURRENT_INPUT="$INPUT"
    fi

    $PYTHON "$SCRIPT_DIR/rd_calc.py" \
        --input "$CURRENT_INPUT" \
        --output "$OUTPUT" \
        --model "$MODEL" \
        --model-name "$MODEL_NAME" \
        --num-gpus "$NUM_GPUS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM"

    echo ""
done

echo "所有模型 RD 计算完成: $OUTPUT"

# ============================================================
# Step 2: 用每个模型的 RD 分别构建数据集
# ============================================================
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_morsd_30k_${MODEL_NAME}"

    echo "============================================================"
    echo " 构建数据集: ${MODEL_NAME}"
    echo " 输出目录: ${OUTPUT_DIR}"
    echo "============================================================"

    $PYTHON "$SCRIPT_DIR/build_dataset_morsd.py" \
        --input "$OUTPUT" \
        --output "$OUTPUT_DIR" \
        --model-name "$MODEL_NAME" \
        --topk 1 \
        --train-ratio 0.99

    echo ""
done

echo "全部完成!"
