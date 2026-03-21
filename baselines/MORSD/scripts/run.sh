#!/bin/bash
set -e

TRAIN_PID=2555901  

echo "等待进程 PID $TRAIN_PID 结束..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "进程仍在进行中，10s 后检查..."
    sleep 10
done
echo "进程已结束"


SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/luoys/cdad/.venv/bin/python"

INPUT="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl"
OUTPUT="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl"

NUM_GPUS=8
MAX_MODEL_LEN=16384

MODELS=(
    "/data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct"
    "/data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
    "/data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct"
)

# 各模型专属配置
# gpu_mem: 模型越大留给 logprob 峰值的空余越多，所以越小
# max_batched_tokens: 控制 chunked prefill 每步 token 数，直接限制内存峰值
#   峰值公式: max_batched_tokens × vocab_size × 5B
#   Llama vocab≈128k: 2048 × 128256 × 5 ≈ 1.2 GB
#   Qwen  vocab≈152k: 2048 × 152064 × 5 ≈ 1.5 GB（已验证 gpu_mem=0.80 下安全）
declare -A MODEL_GPU_MEM
declare -A MODEL_MAX_BATCHED_TOKENS

MODEL_GPU_MEM["Llama-3.2-1B-Instruct"]=0.90
MODEL_MAX_BATCHED_TOKENS["Llama-3.2-1B-Instruct"]=2048

MODEL_GPU_MEM["Llama-3.2-3B-Instruct"]=0.88
MODEL_MAX_BATCHED_TOKENS["Llama-3.2-3B-Instruct"]=2048

MODEL_GPU_MEM["Qwen2.5-7B-Instruct"]=0.80
MODEL_MAX_BATCHED_TOKENS["Qwen2.5-7B-Instruct"]=2048

# ============================================================
# Step 1: 依次用每个学生模型计算 RD（增量追加，断点续传）
# ============================================================
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    GPU_MEM="${MODEL_GPU_MEM[$MODEL_NAME]}"
    MAX_BATCHED_TOKENS="${MODEL_MAX_BATCHED_TOKENS[$MODEL_NAME]}"

    echo "============================================================"
    echo " 计算 RD: ${MODEL_NAME}"
    echo " 模型路径: ${MODEL}"
    echo " gpu_memory_utilization: ${GPU_MEM}"
    echo " max_num_batched_tokens: ${MAX_BATCHED_TOKENS}"
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
        --gpu-memory-utilization "$GPU_MEM" \
        --max-num-batched-tokens "$MAX_BATCHED_TOKENS"

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
