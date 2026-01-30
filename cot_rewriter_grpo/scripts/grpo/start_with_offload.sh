#!/bin/bash
# GRPO 训练启动脚本（使用 Reward Model Load/Offload）
#
# 与原始 start.sh 的区别：
#   - 不需要独立的 Reward Server
#   - 所有 GPU 都用于 GRPO 训练
#   - Reward models 通过 load/offload 机制动态共享 GPU
#   - 不需要复杂的 GPU 分配和占卡策略
#
# 启动顺序（简化）：
#   1. 清理残留进程（Ray, 旧训练）
#   2. 直接启动 GRPO 训练（使用所有可用 GPU）
#   3. Reward 计算时自动 load/offload 模型
#

set -e

# ============== 配置区域 ==============
# 所有可用的 GPU
ALL_GPUS="1,2,3,5"
# GRPO 配置文件
CONFIG_NAME="qwen3_8b_grpo_with_offload"
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

echo "============================================================"
echo "GRPO 训练启动 (使用 Reward Model Load/Offload)"
echo "============================================================"
echo "GPUs: ${ALL_GPUS}"
echo "Config: ${CONFIG_NAME}"
echo "============================================================"

# 清理残留的 Ray 进程
echo ""
echo "[准备] 清理残留的 Ray 进程..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
sleep 2

# 停止可能存在的旧训练进程
echo "[准备] 停止旧的训练进程..."
pkill -f "llamafactory-cli train" 2>/dev/null || true
pkill -f "train.py" 2>/dev/null || true
sleep 2

echo ""
echo "============================================================"
echo "启动 GRPO 训练 (GPUs ${ALL_GPUS})..."
echo "============================================================"

# 启动 GRPO 训练
# 注意: Reward models 会在需要时自动 load/offload
nohup bash scripts/grpo/run_train.sh ${CONFIG_NAME} ${ALL_GPUS} \
    > "${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log" 2>&1 &
GRPO_PID=$!

echo ""
echo "============================================================"
echo "启动完成！"
echo "  GRPO Training PID: ${GRPO_PID}"
echo "  GPUs: ${ALL_GPUS}"
echo ""
echo "日志文件:"
echo "  GRPO: ${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"
echo ""
echo "Reward Model 管理:"
echo "  - Student Model 和 FSF Judge 初始在 CPU"
echo "  - 计算 reward 时自动加载到 GPU"
echo "  - 计算完成后自动卸载回 CPU"
echo "  - 无需独立的 Reward Server"
echo "============================================================"

# 等待训练完成
echo ""
echo "等待 GRPO 训练完成..."
wait $GRPO_PID
GRPO_EXIT_CODE=$?

echo ""
echo "============================================================"
echo "GRPO 训练结束 (exit code: ${GRPO_EXIT_CODE})"
echo "============================================================"

exit $GRPO_EXIT_CODE
