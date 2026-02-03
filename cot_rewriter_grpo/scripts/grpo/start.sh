#!/bin/bash
# GRPO 训练一键启动脚本 (Offload 模式)
# Model: Qwen3-4B
#
# GPU 分配策略：
#   - 全部 2 张 GPU 用于 GRPO 训练
#   - Reward 模型使用 offload 模式：计算时临时加载到 GPU，计算完释放显存
#
# 启动顺序（最小化被抢卡风险）：
#   0. [防抢卡] 清理 Ray 后，先用 fsf_judge 占住所有卡
#   1. 释放所有 GPU → 启动 GRPO 训练
#   2. 等待 GRPO 完成
#   3. 启动 fsf_judge 占卡（无论成功失败）
#
# 全程保证：任何时刻都有进程占住 GPU，防止被抢

set -e

# ============== 配置区域 ==============
# 所有可用的 GPU（全部用于 GRPO 训练）
ALL_GPUS="0,5"
# GPU 数量
NUM_GPUS=2
# ======================================

export DISABLE_VERSION_CHECK=1

# CRITICAL: 使用短路径的 Ray 临时目录，避免 Unix socket 路径长度超过 107 字节限制
# 同时避免与其他用户的 Ray 集群冲突
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

# SwanLab 配置
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/swanlab

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs/grpo"
mkdir -p "${PROJECT_ROOT}/logs/judge"

echo "============================================================"
echo "GRPO 训练一键启动 (Offload 模式)"
echo "============================================================"
echo "Model: Qwen3-4B"
echo "Training GPUs: ${ALL_GPUS} (${NUM_GPUS} GPUs)"
echo "Reward Mode: Offload (临时加载到 GPU，计算完释放)"
echo "============================================================"

# 清理残留的 Ray 进程（只清理当前用户的，避免权限问题）
echo ""
echo "[准备] 清理当前用户的 Ray 进程..."
# 不使用 ray stop，因为它会尝试停止所有用户的 Ray 进程导致权限错误
# 只杀死当前用户拥有的 ray 相关进程
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray._private" 2>/dev/null || true
# 清理当前用户的 Ray 临时目录中的旧会话
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true

# 停止可能存在的旧 reward server（offload 模式不需要，但清理以防万一）
echo "[准备] 停止旧的 Reward Server（如果有）..."
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
        # 等待进程退出
        sleep 3
        # 如果还没退出，强制杀死
        if ps -p "$OLD_FSF_PID" > /dev/null 2>&1; then
            echo "进程未响应，强制杀死..."
            kill -9 "$OLD_FSF_PID" 2>/dev/null || true
            sleep 1
        fi
        echo "旧占卡进程已停止"
    else
        echo "PID 文件存在但进程已不存在，清理 PID 文件"
    fi
    rm -f "$FSF_PID_FILE"
fi
# 同时杀死可能残留的 llamafactory 训练进程
pkill -f "llamafactory-cli train" 2>/dev/null || true
sleep 2

# ============ 步骤 1: 释放所有 GPU ============
echo ""
echo "[Step 1/2] 释放所有 GPUs (${ALL_GPUS}) 用于 GRPO 训练..."
bash scripts/train_fsf_judge/stop_all_train.sh --gpus ${ALL_GPUS}
sleep 2

# ============ 步骤 2: 启动 GRPO 训练 ============
# 配置优化 (解决 OOM):
#   - gpu_memory_utilization: 0.65 -> 0.45 (为 backward 预留显存)
#   - vllm_gpu_memory_utilization: 0.5 -> 0.35 (Student 降低显存)
#   - max_response_length: 6700 -> 5500 (降低序列长度)
#   - max_model_len: 12200 -> 11000 (匹配新长度)
#   - dataloader_num_workers: 4 -> 2 (减少内存压力)
CONFIG_NAME="qwen3_4b_grpo_2x40g_lora"
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
echo "  GRPO: ${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
echo ""
echo "实时查看日志:"
echo "  tail -f ${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
echo "============================================================"

# 等待 GRPO 训练完成（关闭 set -e 以确保后续占卡命令能执行）
echo ""
echo "等待 GRPO 训练完成..."
set +e  # 关闭 errexit，确保即使 GRPO 失败也能继续执行占卡
wait $GRPO_PID
GRPO_EXIT_CODE=$?
set -e  # 重新开启 errexit

echo ""
echo "GRPO 训练结束 (exit code: ${GRPO_EXIT_CODE})"

# 启动 fsf_judge 占卡（无论 GRPO 成功还是失败都要执行）
echo ""
echo "启动 fsf_judge 训练占卡 (GPUs ${ALL_GPUS})..."

# 清理可能残留的旧占卡进程
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
