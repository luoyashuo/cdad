#!/bin/bash
# GRPO 训练启动脚本 (最小显存开销配置 - 2x40GB)
# Model: Qwen3-4B with LoRA (rank=32)
#
# 此配置专为解决 2x40GB GPU OOM 设计
#
# Usage:
#   bash scripts/grpo/start_minimal.sh

set -e

# ============== 配置区域 ==============
ALL_GPUS="0,5"
NUM_GPUS=2
CONFIG_NAME="qwen3_4b_grpo_2x40g_lora_minimal"
# ======================================

export DISABLE_VERSION_CHECK=1
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/swanlab

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs/grpo"
mkdir -p "${PROJECT_ROOT}/logs/judge"

echo "============================================================"
echo "GRPO 训练启动 (最小显存开销配置)"
echo "============================================================"
echo "Model: Qwen3-4B (LoRA rank=32)"
echo "Training GPUs: ${ALL_GPUS} (${NUM_GPUS} GPUs)"
echo "Config: ${CONFIG_NAME}"
echo ""
echo "关键优化:"
echo "  - LoRA rank: 64 -> 32"
echo "  - rollout.n: 8 -> 4"
echo "  - max_prompt_length: 5500 -> 4096"
echo "  - max_response_length: 6700 -> 5120"
echo "  - vLLM gpu_memory_utilization: 0.65 -> 0.55"
echo "============================================================"

# 清理残留的 Ray 进程
echo ""
echo "[准备] 清理当前用户的 Ray 进程..."
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray._private" 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true

# 停止可能存在的旧 reward server
pkill -f "student_reward_server" 2>/dev/null || true
sleep 2

# 停止可能存在的旧占卡进程
echo "[准备] 检查并停止旧的占卡进程..."
FSF_PID_FILE="/tmp/train_fsf_judge.pid"
if [ -f "$FSF_PID_FILE" ]; then
    OLD_FSF_PID=$(cat "$FSF_PID_FILE")
    if ps -p "$OLD_FSF_PID" > /dev/null 2>&1; then
        echo "发现旧的占卡进程 (PID: $OLD_FSF_PID)，正在停止..."
        kill "$OLD_FSF_PID" 2>/dev/null || true
        sleep 3
        if ps -p "$OLD_FSF_PID" > /dev/null 2>&1; then
            echo "进程未响应，强制杀死..."
            kill -9 "$OLD_FSF_PID" 2>/dev/null || true
            sleep 1
        fi
        echo "旧占卡进程已停止"
    fi
    rm -f "$FSF_PID_FILE"
fi
pkill -f "llamafactory-cli train" 2>/dev/null || true
sleep 2

# ============ 步骤 1: 释放所有 GPU ============
echo ""
echo "[Step 1/2] 释放所有 GPUs (${ALL_GPUS}) 用于 GRPO 训练..."
bash scripts/train_fsf_judge/stop_all_train.sh --gpus ${ALL_GPUS}
sleep 2

# ============ 步骤 2: 启动 GRPO 训练 ============
echo ""
echo "[Step 2/2] 启动 GRPO 训练 (GPUs ${ALL_GPUS}, Config: ${CONFIG_NAME})..."
nohup bash scripts/grpo/run_train.sh ${CONFIG_NAME} ${ALL_GPUS} \
    > "${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log" 2>&1 &
GRPO_PID=$!
echo "GRPO Training PID: ${GRPO_PID}"

echo ""
echo "============================================================"
echo "启动完成！"
echo "  GRPO Training: GPUs ${ALL_GPUS}, PID ${GRPO_PID}"
echo ""
echo "日志文件:"
echo "  ${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
echo ""
echo "实时查看日志:"
echo "  tail -f ${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
echo "============================================================"

# 等待 GRPO 训练完成
set +e
echo ""
echo "等待 GRPO 训练完成..."
wait $GRPO_PID
GRPO_EXIT_CODE=$?
set -e

echo ""
echo "GRPO 训练结束 (exit code: ${GRPO_EXIT_CODE})"

# 启动 fsf_judge 占卡
echo ""
echo "启动 fsf_judge 训练占卡 (GPUs ${ALL_GPUS})..."
if [ -f "$FSF_PID_FILE" ]; then
    OLD_FSF_PID=$(cat "$FSF_PID_FILE")
    if ps -p "$OLD_FSF_PID" > /dev/null 2>&1; then
        echo "发现残留的占卡进程 (PID: $OLD_FSF_PID)，正在停止..."
        kill "$OLD_FSF_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$OLD_FSF_PID" 2>/dev/null || true
    fi
    rm -f "$FSF_PID_FILE"
fi
pkill -f "llamafactory-cli train" 2>/dev/null || true
sleep 2

bash scripts/train_fsf_judge/start_train.sh --gpus ${ALL_GPUS}
