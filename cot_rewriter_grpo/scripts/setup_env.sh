#!/bin/bash
# 环境变量配置脚本
# 用法: source scripts/setup_env.sh
#
# 此脚本设置训练所需的环境变量
# 包含四个基础目录配置：
#   - MODEL_BASE_DIR: 模型根目录
#   - DATA_BASE_DIR: 数据根目录
#   - CHECKPOINT_BASE_DIR: 检查点根目录
#   - PROJECT_BASE_DIR: 项目代码根目录

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "CoT Rewriter GRPO - 环境变量配置"
echo "============================================================"
echo "项目根目录: ${PROJECT_ROOT}"
echo ""

# ========================================
# 基础目录配置 - 根据环境修改这些变量
# ========================================

# 模型基础目录（存放 Qwen3-4B, Qwen3-1.7B 等模型）
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-/data-share/LLM}"

# 数据基础目录（存放 cot_rewrite_max_fsf 等数据）
export DATA_BASE_DIR="${DATA_BASE_DIR:-/data-share/yeesuanAI08/lihaitao/luoys/data/cdad}"

# 检查点基础目录（存放训练检查点）
export CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad}"

# 项目代码基础目录（用于拼接代码路径）
export PROJECT_BASE_DIR="${PROJECT_BASE_DIR:-${PROJECT_ROOT}}"

# ========================================
# 其他环境变量
# ========================================

# SwanLab 日志目录
export SWANLAB_LOG_DIR="${CHECKPOINT_BASE_DIR}/swanlab"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages:${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/verl:${PYTHONPATH}"

# Ray 配置
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

echo "已设置的环境变量:"
echo "  MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "  DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "  CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "  PROJECT_BASE_DIR=${PROJECT_BASE_DIR}"
echo "  SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR}"
echo "  RAY_TMPDIR=${RAY_TMPDIR}"
echo ""
echo "如需修改路径，请在运行此脚本前设置对应的环境变量，例如:"
echo "  export MODEL_BASE_DIR=/fs/fast/u20240075/models"
echo "  export DATA_BASE_DIR=/home/u20240075/data"
echo "  export CHECKPOINT_BASE_DIR=/fs/fast/u20240075/checkpoints"
echo "  source scripts/setup_env.sh"
echo "============================================================"

# 检查目录是否存在
echo ""
echo "检查路径..."

if [ -d "$MODEL_BASE_DIR" ]; then
    echo "  ✓ MODEL_BASE_DIR 存在: $MODEL_BASE_DIR"
    # 检查关键模型是否存在
    if [ -d "$MODEL_BASE_DIR/Qwen3-4B" ]; then
        echo "    ✓ Qwen3-4B 存在"
    else
        echo "    ✗ Qwen3-4B 不存在: $MODEL_BASE_DIR/Qwen3-4B"
    fi
    if [ -d "$MODEL_BASE_DIR/Qwen3-1.7B" ]; then
        echo "    ✓ Qwen3-1.7B 存在"
    else
        echo "    ✗ Qwen3-1.7B 不存在: $MODEL_BASE_DIR/Qwen3-1.7B"
    fi
else
    echo "  ✗ MODEL_BASE_DIR 不存在: $MODEL_BASE_DIR"
    echo "    请设置 MODEL_BASE_DIR 环境变量指向您的模型目录"
fi

if [ -d "$DATA_BASE_DIR" ]; then
    echo "  ✓ DATA_BASE_DIR 存在: $DATA_BASE_DIR"
    # 检查关键数据是否存在
    if [ -f "$DATA_BASE_DIR/cot_rewrite_max_fsf/train.jsonl" ]; then
        echo "    ✓ cot_rewrite_max_fsf/train.jsonl 存在"
    else
        echo "    ✗ cot_rewrite_max_fsf/train.jsonl 不存在"
    fi
else
    echo "  ✗ DATA_BASE_DIR 不存在: $DATA_BASE_DIR"
    echo "    请设置 DATA_BASE_DIR 环境变量指向您的数据目录"
fi

if [ -d "$CHECKPOINT_BASE_DIR" ]; then
    echo "  ✓ CHECKPOINT_BASE_DIR 存在: $CHECKPOINT_BASE_DIR"
else
    echo "  ○ CHECKPOINT_BASE_DIR 将自动创建: $CHECKPOINT_BASE_DIR"
    mkdir -p "$CHECKPOINT_BASE_DIR"
fi

if [ -d "$PROJECT_BASE_DIR" ]; then
    echo "  ✓ PROJECT_BASE_DIR 存在: $PROJECT_BASE_DIR"
else
    echo "  ✗ PROJECT_BASE_DIR 不存在: $PROJECT_BASE_DIR"
fi

echo ""
