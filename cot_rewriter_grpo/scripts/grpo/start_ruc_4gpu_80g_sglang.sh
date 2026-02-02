#!/bin/bash
# GRPO 训练启动脚本 (4 x 80G GPU 版本) - SGLang 后端
# Model: Qwen3-4B
#
# GPU 分配：全部 4 张 80G GPU 用于 GRPO 训练
# Rollout & Reward: 使用 SGLang 后端
#
# 使用方法:
#   bash scripts/grpo/start_ruc_4gpu_80g_sglang.sh              # 默认使用 GPU 0,1,2,3
#   bash scripts/grpo/start_ruc_4gpu_80g_sglang.sh 4,5,6,7      # 指定 GPU

set -e

# ============== 配置区域 ==============
# 默认 GPU 配置 (可通过命令行参数覆盖)
ALL_GPUS="${1:-0,1,2,3}"
NUM_GPUS=4
CONFIG_NAME="qwen3_4b_grpo_ruc_4gpu_80g"
# ======================================

export DISABLE_VERSION_CHECK=1

# CRITICAL: 使用短路径的 Ray 临时目录，避免 Unix socket 路径长度超过 107 字节限制
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

# SwanLab 配置
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/fs/fast/u20240075/luoyashuo/checkpoints/cdad/swanlab

# SGLang 相关环境变量
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true
# 禁用 flashinfer 以避免某些兼容性问题（可选，如果有问题可以取消注释）
# export SGL_DISABLE_FLASHINFER=1

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs/grpo"

echo "============================================================"
echo "GRPO 训练启动 (4 x 80G GPU) - SGLang 后端"
echo "============================================================"
echo "Model: Qwen3-4B (Full Parameter)"
echo "Config: ${CONFIG_NAME}"
echo "Training GPUs: ${ALL_GPUS} (${NUM_GPUS} GPUs)"
echo "Rollout Engine: SGLang"
echo "Reward Backend: SGLang"
echo "============================================================"

# 清理残留的 Ray 进程
echo ""
echo "[准备] 清理当前用户的 Ray 进程..."
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray._private" 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true
sleep 2

# 启动 GRPO 训练
echo ""
echo "[启动] GRPO 训练 (GPUs ${ALL_GPUS})..."
LOG_FILE="${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}_sglang.log"

nohup bash scripts/grpo/run_train.sh ${CONFIG_NAME} ${ALL_GPUS} \
    > "${LOG_FILE}" 2>&1 &
GRPO_PID=$!
echo "GRPO Training PID: ${GRPO_PID}"

echo ""
echo "============================================================"
echo "启动完成！"
echo "  GRPO Training: GPUs ${ALL_GPUS}, PID ${GRPO_PID}"
echo ""
echo "日志文件:"
echo "  ${LOG_FILE}"
echo ""
echo "实时查看日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "停止训练:"
echo "  kill ${GRPO_PID}"
echo "============================================================"
