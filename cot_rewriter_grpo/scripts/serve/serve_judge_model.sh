#!/bin/bash

# 启动 judge 模型服务脚本
# 支持手动设置 GPU
# 模型路径: /data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/openai/gpt-oss-20b

# 项目根目录
PROJECT_ROOT="/data-share/yeesuanAI08/lihaitao/luoys/projects/meta_circuit_distill"

# 模型路径
MODEL_PATH="/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/openai/gpt-oss-20b"

# 默认 GPU 设备 (可以通过命令行参数 --gpus 或 -g 设置)
GPUS="2"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--gpus)
            GPUS="$2"
            shift; shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -g, --gpus    Specify GPU devices (e.g., 0,1 or 0)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use -h or --help to see available options"
            exit 1
            ;;
    esac
done

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# 检查是否在项目根目录
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project root not found: $PROJECT_ROOT"
    exit 1
fi

# 进入项目根目录
cd "$PROJECT_ROOT" || exit 1

# 构建 vllm serve 命令
VLLM_CMD="uv run python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --max-num-seqs 256"

# 如果指定了 GPU，添加 CUDA_VISIBLE_DEVICES 环境变量
if [ -n "$GPUS" ]; then
    echo "Using GPU devices: $GPUS"
    VLLM_CMD="CUDA_VISIBLE_DEVICES=$GPUS $VLLM_CMD"
else
    echo "Using all available GPU devices"
fi

# 输出启动信息
echo "----------------------------------------"
echo "Starting judge model service..."
echo "Model: $MODEL_PATH"
echo "Project root: $PROJECT_ROOT"
echo "Port: 8000"
echo "Host: 0.0.0.0"
echo "----------------------------------------"

# 执行启动命令
eval $VLLM_CMD