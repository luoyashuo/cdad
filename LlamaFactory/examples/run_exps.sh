#!/bin/bash

# 设置工作目录
cd /home/luoys/cdad/LlamaFactory
mkdir -p logs

# 提取公共环境变量，对后续所有命令生效
# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export DISABLE_VERSION_CHECK=1
export SWANLAB_API_KEY="xmtHYzg2Bw5uh1dprg81E"

# 日志目录
LOG_DIR="logs"

TRAIN_PID=3028943  

echo "等待训练进程 PID $TRAIN_PID 结束..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    echo "训练仍在进行中，10s 后检查..."
    sleep 10
done
log "训练进程已结束"

# 训练任务列表
declare -a CONFIGS=(
    # "examples/vallina_sft/qwen2.5_1.5b_instruct_vallina_sft_50k.yaml"
    # "examples/vallina_sft/qwen2.5_3b_instruct_vallina_sft_50k.yaml"
    # "examples/vallina_sft/qwen2.5_7b_instruct_lora_sft_30k.yaml"
    # "examples/vallina_sft/qwen2.5_7b_instruct_full_sft_30k.yaml"
    # "examples/vallina_sft/qwen2.5_3b_instruct_full_sft_30k.yaml"
    # "examples/vallina_sft/llama3_8b_instruct_vallina_full_sft_30k.yaml"
    # "examples/vallina_sft/llama3_3b_instruct_vallina_full_sft_30k.yaml"
    # "examples/vallina_sft/llama3_1b_instruct_vallina_full_sft_30k.yaml"
    # "examples/rewrite_cot/qwen2.5_7b_instruct_full_sft_dual_30k.yaml"
    # "examples/rewrite_cot/qwen2.5_7b_instruct_lora_sft_dual_30k.yaml"
    # "examples/vallina_sft/qwen2.5_7b_instruct_lora_sft_30k.yaml"
    # "examples/rewrite_cot/qwen2.5_7b_instruct_lora_sft_mix_15k.yaml"
    "examples/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r25.yaml"
    "examples/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r50.yaml"
    "examples/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r75.yaml"

)

declare -a NAMES=(
    # "qwen2.5_1.5b_instruct_vallina_sft_50k"
    # "qwen2.5_3b_instruct_vallina_sft_50k"
    # "qwen2.5_7b_instruct_lora_sft_30k"
    # "qwen2.5_7b_instruct_full_sft_30k"
    # "qwen2.5_3b_instruct_full_sft_30k"
    # "llama3_8b_instruct_vallina_full_sft_30k"
    # "llama3_3b_instruct_vallina_full_sft_30k"
    # "llama3_1b_instruct_vallina_full_sft_30k"
    # "qwen2.5_7b_instruct_full_sft_dual_30k"
    # "qwen2.5_7b_instruct_lora_sft_dual_30k"
    # "qwen2.5_7b_instruct_lora_sft_30k"
    "qwen2.5_7b_instruct_lora_sft_mix_15k"
)

echo "=== 开始串行训练任务 ==="
echo "日志将保存到: ${LOG_DIR}/"
echo ""

TOTAL=${#CONFIGS[@]}

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    LOG_FILE="${LOG_DIR}/${NAME}.log"
    NUM=$((i + 1))
    
    echo "[$NUM/$TOTAL] 开始训练: $NAME"
    echo "    配置文件: $CONFIG"
    echo "    日志文件: $LOG_FILE"
    echo "    开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 直接运行（不使用 nohup），让输出同时显示在终端和写入日志
    llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG_FILE"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "    ✓ 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    else
        echo "    ✗ 训练失败，退出码: $EXIT_CODE"
        echo "    请检查日志: $LOG_FILE"
        echo ""
        exit $EXIT_CODE
    fi
done

echo "=== 所有训练任务已完成 ==="
