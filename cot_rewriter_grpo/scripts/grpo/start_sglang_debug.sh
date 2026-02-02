#!/bin/bash
# SGLang Rollout 调试脚本
# 使用规则 reward 测试 SGLang rollout 是否正常工作
#
# Usage:
#   bash scripts/grpo/start_sglang_debug.sh              # 默认使用 GPU 0,1,2,3
#   bash scripts/grpo/start_sglang_debug.sh 4,5,6,7      # 指定 GPU

set -e

# ============== 配置区域 ==============
ALL_GPUS="${1:-0,1,2,3}"
NUM_GPUS=4
CONFIG_NAME="qwen3_4b_grpo_sglang_debug"
# ======================================

export DISABLE_VERSION_CHECK=1

# Ray 临时目录
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

# SGLang 相关环境变量
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true

# Ray 日志相关 - 禁用日志去重以便调试
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 创建日志目录
mkdir -p "${PROJECT_ROOT}/logs/grpo"

echo "============================================================"
echo "SGLang Rollout 调试 (规则 Reward)"
echo "============================================================"
echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${ALL_GPUS}"
echo "============================================================"

# 清理残留的 Ray 进程
echo ""
echo "[准备] 清理 Ray 进程..."
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true
sleep 2

# 设置 PYTHONPATH 以包含 asyncio patch
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

# 日志文件
LOG_FILE="${PROJECT_ROOT}/logs/grpo/${CONFIG_NAME}.log"

echo ""
echo "[启动] SGLang 调试训练..."
echo "日志文件: ${LOG_FILE}"
echo ""

cd "${PROJECT_ROOT}"

# 创建一个临时的启动脚本来应用 asyncio patch
TEMP_LAUNCHER=$(mktemp /tmp/sglang_launcher_XXXXXX.py)
cat > "${TEMP_LAUNCHER}" << 'LAUNCHER_EOF'
import sys
import os

# Apply asyncio patch FIRST (before any other imports)
import asyncio

_original_get_event_loop = asyncio.get_event_loop

def _patched_get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = _original_get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

asyncio.get_event_loop = _patched_get_event_loop
print("[asyncio_patch] Applied asyncio.get_event_loop() compatibility patch")

# Now run the actual training
if __name__ == "__main__":
    from verl.trainer.main_ppo import main
    main()
LAUNCHER_EOF

# 运行训练，输出到日志文件（后台运行）
nohup bash -c "CUDA_VISIBLE_DEVICES=${ALL_GPUS} python '${TEMP_LAUNCHER}' \
    --config-path='${PROJECT_ROOT}/examples' \
    --config-name='${CONFIG_NAME}' ; rm -f '${TEMP_LAUNCHER}'" \
    > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo "============================================================"
echo "训练已启动！"
echo "  PID: ${TRAIN_PID}"
echo "  日志: ${LOG_FILE}"
echo ""
echo "查看日志:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "停止训练:"
echo "  kill ${TRAIN_PID}"
echo "============================================================"
