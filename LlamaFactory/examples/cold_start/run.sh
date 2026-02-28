#!/bin/bash

# 设置工作目录
cd /home/u20240075/luoyashuo/projects/cdad/LlamaFactory
mkdir -p logs

# 提取公共环境变量，对后续所有命令生效
export CUDA_VISIBLE_DEVICES="2,3"
export DISABLE_VERSION_CHECK=1
export SWANLAB_API_KEY="xmtHYzg2Bw5uh1dprg81E"

echo "=== 开始串行训练任务 ==="

# 1. 训练 Qwen2.5-1.5B
echo "[1/3] 正在训练 Qwen2.5-1.5B..."
nohup llamafactory-cli train examples/cold_start/qwen2.5_1.5b_meta_math_40k.yaml > logs/qwen2.5_1.5b_meta_math_40k.log 2>&1 &
wait

# 2. 训练 Mistral-7B
echo "[2/3] 正在训练 Mistral-7B..."
nohup llamafactory-cli train examples/cold_start/mistral_7b_meta_math_40k.yaml > logs/mistral_7b_meta_math_40k.log 2>&1 &
wait

# 3. 训练 Llama-3.2-1B
echo "[3/3] 正在训练 Llama-3.2-1B..."
nohup llamafactory-cli train examples/cold_start/llama3_1b_meta_math_40k.yaml > logs/llama3_1b_meta_math_40k.log 2>&1 &
wait
echo "=== 所有训练任务已完成 ==="
# nohup bash examples/cold_start/run.sh > logs/run.log 2>&1 &