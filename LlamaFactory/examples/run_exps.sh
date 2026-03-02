#!/bin/bash

# 设置工作目录
cd /home/u20240075/luoyashuo/projects/cdad/LlamaFactory
mkdir -p logs

# 提取公共环境变量，对后续所有命令生效
export CUDA_VISIBLE_DEVICES="2,3"
export DISABLE_VERSION_CHECK=1
export SWANLAB_API_KEY="xmtHYzg2Bw5uh1dprg81E"

# 日志目录
LOG_DIR="logs"

# 训练任务列表
declare -a CONFIGS=(
    "examples/vallina_sft/llama3_1b_vallina_sft_20k.yaml"
    "examples/vallina_sft/qwen2.5_1.5b_vallina_sft_20k.yaml"
    "examples/rv_optimal/llama3_1b_rv_optimal_20k.yaml"
    "examples/rv_optimal/qwen2.5_1.5b_rv_optimal_20k.yaml"
)

declare -a NAMES=(
    "llama3_1b_vallina_sft_20k"
    "qwen2.5_1.5b_vallina_sft_20k"
    "llama3_1b_rv_optimal_20k"
    "qwen2.5_1.5b_rv_optimal_20k"
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
