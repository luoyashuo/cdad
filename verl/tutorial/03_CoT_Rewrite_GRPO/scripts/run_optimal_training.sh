#!/bin/bash
# =============================================================================
# 最优训练启动脚本
# 配置：7B 全参 GRPO + 1.7B 统一奖励模型
# 硬件：4×A100-80G
# =============================================================================

set -e

# ============ 颜色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "CoT Rewriter GRPO Training"
echo "=========================================="
echo -e "配置: 7B 全参 + 1.7B 统一奖励模型"
echo -e "硬件: 4×A100-80G"
echo -e "==========================================${NC}"

# ============ 检查 GPU ============
echo -e "\n${YELLOW}[1/4] 检查 GPU 环境...${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo -e "${GREEN}检测到 ${GPU_COUNT} 张 GPU${NC}"

if [ "$GPU_COUNT" -lt 4 ]; then
    echo -e "${RED}错误: 需要至少 4 张 GPU，当前只有 ${GPU_COUNT} 张${NC}"
    exit 1
fi

# ============ 环境变量配置 ============
echo -e "\n${YELLOW}[2/4] 配置环境变量...${NC}"

# GPU 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ray 配置
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0

# NCCL 配置 (分布式通信)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch 配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 配置
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-~/.cache/huggingface/transformers}

echo -e "${GREEN}环境变量配置完成${NC}"

# ============ 检查模型文件 ============
echo -e "\n${YELLOW}[3/4] 检查模型文件...${NC}"

ACTOR_MODEL="./models/Qwen2.5-7B-Instruct"
REWARD_MODEL="./models/Qwen2.5-1.7B-Instruct"

if [ ! -d "$ACTOR_MODEL" ]; then
    echo -e "${RED}错误: Actor 模型不存在: $ACTOR_MODEL${NC}"
    echo "请先下载模型: huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $ACTOR_MODEL"
    exit 1
fi

if [ ! -d "$REWARD_MODEL" ]; then
    echo -e "${YELLOW}警告: 奖励模型不存在: $REWARD_MODEL${NC}"
    echo "将尝试从 HuggingFace 自动下载..."
fi

echo -e "${GREEN}模型检查完成${NC}"

# ============ 启动训练 ============
echo -e "\n${YELLOW}[4/4] 启动训练...${NC}"

# 切换到项目目录
cd "$(dirname "$0")/.."

# 启动训练
python scripts/train.py \
    --config-name=optimal_7b_fullparam_grpo \
    "$@"

echo -e "\n${GREEN}=========================================="
echo "训练完成!"
echo -e "==========================================${NC}"
