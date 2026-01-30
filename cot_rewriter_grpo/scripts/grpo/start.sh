#!/bin/bash
# GRPO 训练一键启动脚本
#
# GPU 分配策略：
#   - REWARD_GPU: 1张卡运行 Reward Server (Student Model)
#   - GRPO_GPUS: 其余卡运行 GRPO 训练
#
# 启动顺序（最小化被抢卡风险）：
#   0. [防抢卡] 清理 Ray 后，先用 fsf_judge 占住所有卡
#   1. 释放 REWARD_GPU → 启动 Reward Server → 等待 ready
#   2. 释放 GRPO_GPUS → 启动 GRPO 训练
#   3. 等待 GRPO 完成 → 停止 Reward Server
#   4. 启动 fsf_judge 占卡（无论成功失败）
#
# 全程保证：任何时刻都有进程占住 GPU，防止被抢

set -e

# ============== 配置区域 ==============
# 所有可用的 GPU
ALL_GPUS="1,2,3,5"
# Reward Server 使用的 GPU（单张卡）
REWARD_GPU="5"
# GRPO 训练使用的 GPU（其余卡）
GRPO_GPUS="1,2,3"
# Reward Server 端口
REWARD_PORT="8765"
# Student Model 路径
STUDENT_MODEL_PATH="/data-share/LLM/Qwen3-1.7B"
# Reward Server 健康检查超时（秒）
HEALTH_CHECK_TIMEOUT=300
# 健康检查间隔（秒）
HEALTH_CHECK_INTERVAL=5
# ======================================

export DISABLE_VERSION_CHECK=1

# SwanLab 配置
export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR=/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/swanlab

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs/grpo"
mkdir -p "${PROJECT_ROOT}/logs/reward_server"
mkdir -p "${PROJECT_ROOT}/logs/judge"

echo "============================================================"
echo "GRPO 训练一键启动"
echo "============================================================"
echo "Reward Server GPU: ${REWARD_GPU}"
echo "GRPO Training GPUs: ${GRPO_GPUS}"
echo "============================================================"

# 清理残留的 Ray 进程（避免权限问题和端口冲突）
echo ""
echo "[准备] 清理残留的 Ray 进程..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true

# 停止可能存在的旧 reward server
echo "[准备] 停止旧的 Reward Server..."
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

# ============ 关键：先用 fsf_judge 占住所有卡，防止被抢 ============
echo ""
echo "[防抢卡] 启动 fsf_judge 占住所有 GPU (${ALL_GPUS})..."
# 后台启动 fsf_judge 占卡（会自动循环训练）
nohup bash scripts/train_fsf_judge/start_train.sh --gpus ${ALL_GPUS} \
    > "${PROJECT_ROOT}/logs/judge/fsf_judge_placeholder.log" 2>&1 &
FSF_PLACEHOLDER_PID=$!
echo "fsf_judge 占位进程 PID: ${FSF_PLACEHOLDER_PID}"

# 等待 fsf_judge 启动并占住 GPU
echo "等待 fsf_judge 占住 GPU..."
sleep 30  # 等待训练进程启动并加载模型

# ============ 步骤 1: 只释放 Reward GPU ============
echo ""
echo "[Step 1/4] 释放 GPU ${REWARD_GPU} 用于 Reward Server..."
bash scripts/train_fsf_judge/stop_all_train.sh --gpus ${REWARD_GPU}
sleep 2

# ============ 步骤 2: 启动 Reward Server 并等待 ready ============
echo ""
echo "[Step 2/4] 启动 Reward Server (GPU ${REWARD_GPU})..."
nohup bash scripts/grpo/start_reward_server.sh ${REWARD_GPU} ${REWARD_PORT} ${STUDENT_MODEL_PATH} \
    > "${PROJECT_ROOT}/logs/reward_server/student_reward_server.log" 2>&1 &
REWARD_PID=$!
echo "Reward Server PID: ${REWARD_PID}"

# 等待 Reward Server ready
echo "等待 Reward Server 启动..."
HEALTH_URL="http://localhost:${REWARD_PORT}/health"
elapsed=0
while [ $elapsed -lt $HEALTH_CHECK_TIMEOUT ]; do
    # 检查进程是否还在运行
    if ! kill -0 $REWARD_PID 2>/dev/null; then
        echo "错误: Reward Server 进程已退出！"
        echo "查看日志: ${PROJECT_ROOT}/logs/reward_server/student_reward_server.log"
        exit 1
    fi

    # 尝试健康检查
    if curl -s --connect-timeout 2 "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "Reward Server 已就绪！(耗时 ${elapsed}s)"
        break
    fi

    echo "  等待中... (${elapsed}s/${HEALTH_CHECK_TIMEOUT}s)"
    sleep $HEALTH_CHECK_INTERVAL
    elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
done

if [ $elapsed -ge $HEALTH_CHECK_TIMEOUT ]; then
    echo "错误: Reward Server 启动超时 (${HEALTH_CHECK_TIMEOUT}s)！"
    echo "查看日志: ${PROJECT_ROOT}/logs/reward_server/student_reward_server.log"
    kill $REWARD_PID 2>/dev/null || true
    exit 1
fi

# ============ 步骤 3: 释放 GRPO GPU ============
echo ""
echo "[Step 3/4] 释放 GPUs ${GRPO_GPUS} 用于 GRPO 训练..."
bash scripts/train_fsf_judge/stop_all_train.sh --gpus ${GRPO_GPUS}
sleep 2

# ============ 步骤 4: 启动 GRPO 训练 ============
echo ""
echo "[Step 4/4] 启动 GRPO 训练 (GPUs ${GRPO_GPUS})..."
nohup bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu ${GRPO_GPUS} \
    > "${PROJECT_ROOT}/logs/grpo/qwen3_8b_grpo_4gpu.log" 2>&1 &
GRPO_PID=$!
echo "GRPO Training PID: ${GRPO_PID}"

echo ""
echo "============================================================"
echo "启动完成！"
echo "  Reward Server: GPU ${REWARD_GPU}, PID ${REWARD_PID}"
echo "  GRPO Training: GPUs ${GRPO_GPUS}, PID ${GRPO_PID}"
echo ""
echo "日志文件:"
echo "  Reward: ${PROJECT_ROOT}/logs/reward_server/student_reward_server.log"
echo "  GRPO:   ${PROJECT_ROOT}/logs/grpo/qwen3_8b_grpo_4gpu.log"
echo "============================================================"

# 等待 GRPO 训练完成（关闭 set -e 以确保后续占卡命令能执行）
echo ""
echo "等待 GRPO 训练完成..."
set +e  # 关闭 errexit，确保即使 GRPO 失败也能继续执行占卡
wait $GRPO_PID
GRPO_EXIT_CODE=$?
set -e  # 重新开启 errexit

# GRPO 训练结束后（无论成功失败），停止 Reward Server
echo ""
echo "GRPO 训练结束 (exit code: ${GRPO_EXIT_CODE})，停止 Reward Server..."
kill $REWARD_PID 2>/dev/null || true
pkill -f "student_reward_server" 2>/dev/null || true

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
