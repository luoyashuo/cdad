#!/bin/bash

# 启动脚本：使用 vLLM 加载两个不同的 Qwen3 模型到同一张 GPU 并启动服务

# 配置参数
MODEL_8B_PATH="/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/Qwen/Qwen3-8B"
MODEL_1_7B_PATH="/data-share/LLM/Qwen3-1.7B"

LOG_8B_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/logs/judge_server/serve_qwen3_8b_fsf_judge.log"
LOG_1_7B_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/logs/judge_server/serve_qwen3_1_7b.log"

PORT_8B=8100
PORT_1_7B=8101

# 两个模型都加载到同一张 GPU 上
GPU_ID=0

# 根据模型大小比例分配显存
# Qwen3-8B 和 Qwen3-1.7B 的大小比例约为 8:1.7 ≈ 4.7:1
TOTAL_MEMORY_UTILIZATION=0.85  # 降低总内存使用率以避免量化
MODEL_8B_RATIO=0.825  # 8 / (8 + 1.7) ≈ 0.825
MODEL_1_7B_RATIO=0.175  # 1.7 / (8 + 1.7) ≈ 0.175

GPU_MEM_UTIL_8B=$(echo "$TOTAL_MEMORY_UTILIZATION * $MODEL_8B_RATIO" | bc -l | awk '{printf "%.3f", $1}')
GPU_MEM_UTIL_1_7B=$(echo "$TOTAL_MEMORY_UTILIZATION * $MODEL_1_7B_RATIO" | bc -l | awk '{printf "%.3f", $1}')

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_8B_PATH")"
mkdir -p "$(dirname "$LOG_1_7B_PATH")"

# 启动 Qwen3-8B 模型服务（不使用量化，保持高保真度）
echo "启动 Qwen3-8B 模型服务..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_8B_PATH" \
    --port "$PORT_8B" \
    --gpu-memory-utilization $GPU_MEM_UTIL_8B \
    --tensor-parallel-size 1 \
    --device cuda:$GPU_ID \
    --dtype float16 \
    --load-format lazy \


    > "$LOG_8B_PATH" 2>&1 &
echo "Qwen3-8B 服务已启动，PID: $!"

# 启动 Qwen3-1.7B 模型服务（不使用量化，保持高保真度）
echo "启动 Qwen3-1.7B 模型服务..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_1_7B_PATH" \
    --port "$PORT_1_7B" \
    --gpu-memory-utilization $GPU_MEM_UTIL_1_7B \
    --tensor-parallel-size 1 \
    --device cuda:$GPU_ID \
    --dtype float16 \
    --load-format lazy \


    > "$LOG_1_7B_PATH" 2>&1 &
echo "Qwen3-1.7B 服务已启动，PID: $!"

echo "两个 vLLM 服务已成功启动到 GPU $GPU_ID！"
echo "Qwen3-8B 服务：端口 $PORT_8B，显存分配比例：$MODEL_8B_RATIO，GPU内存使用率：$GPU_MEM_UTIL_8B，日志：$LOG_8B_PATH"
echo "Qwen3-1.7B 服务：端口 $PORT_1_7B，显存分配比例：$MODEL_1_7B_RATIO，GPU内存使用率：$GPU_MEM_UTIL_1_7B，日志：$LOG_1_7B_PATH"
echo "总GPU内存使用率：$TOTAL_MEMORY_UTILIZATION"
echo "注意：已移除量化选项，保持完整精度以确保最佳模型效果"