#!/bin/bash
# CoT Rewriter GRPO 训练启动脚本
# 
# 使用方法:
#   通过 start_qc.sh 或 start_ruc_4gpu_80g.sh 调用，不要直接运行！
#
# 必需的环境变量 (由调用脚本设置):
#   - MODEL_BASE_DIR: 模型根目录
#   - DATA_BASE_DIR: 数据根目录  
#   - CHECKPOINT_BASE_DIR: 检查点根目录
#   - PROJECT_BASE_DIR: 项目代码根目录
#
# 参数:
#   $1: 配置文件名 (不含 .yaml 后缀，如: qwen3_4b_grpo_2x40g_lora_sglang)
#   $2: GPU 列表 (如: 4,5)

set -e

# 检查必需的环境变量
if [ -z "$MODEL_BASE_DIR" ] || [ -z "$DATA_BASE_DIR" ] || [ -z "$CHECKPOINT_BASE_DIR" ] || [ -z "$PROJECT_BASE_DIR" ]; then
    echo "Error: 必需的环境变量未设置"
    echo "请通过 start_qc.sh 或 start_ruc_4gpu_80g.sh 启动，不要直接运行此脚本"
    exit 1
fi

# 解析参数
CONFIG_NAME="${1}"
CUDA_VISIBLE_DEVICES="${2}"

if [ -z "$CONFIG_NAME" ] || [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: 缺少参数"
    echo "用法: $0 <config_name> <gpus>"
    echo "例如: $0 qwen3_4b_grpo_2x40g_lora_sglang 4,5"
    exit 1
fi

# 配置文件路径
CONFIG_FILE="${PROJECT_BASE_DIR}/examples/${CONFIG_NAME}.yaml"

# 日志目录
LOG_DIR="${PROJECT_BASE_DIR}/logs/grpo"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}.log"

# 清理之前的日志文件
rm -f "${LOG_FILE}"

echo "============================================================"
echo "CoT Rewriter GRPO Training"
echo "============================================================"
echo "Config: ${CONFIG_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Path Configuration:"
echo "  MODEL_BASE_DIR=${MODEL_BASE_DIR}"
echo "  DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "  CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR}"
echo "  PROJECT_BASE_DIR=${PROJECT_BASE_DIR}"
echo "============================================================"

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES
export ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"  # 保存原始 GPU ID
export PROJECT_BASE_DIR
export MODEL_BASE_DIR
export DATA_BASE_DIR
export CHECKPOINT_BASE_DIR

# Ray 配置
export RAY_DEDUP_LOGS=1
export RAY_memory_monitor_refresh_ms=0
export RAY_TMPDIR="/tmp/ray_${USER:-$(whoami)}"
mkdir -p "$RAY_TMPDIR"

# NCCL 配置
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0

# PyTorch 配置
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False
export TORCH_COMPILE_DISABLE=1

# vLLM/SGLang 配置
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Ray Dashboard 配置
export RAY_DASHBOARD_HOST=""
export RAY_enable_dashboard=0
export RAY_DISABLE_DASHBOARD_AGENT=1
export RAY_ADDRESS=local

# Python 路径
export PYTHONPATH="${PROJECT_BASE_DIR}:${PROJECT_BASE_DIR}/verl:${PROJECT_BASE_DIR}/.venv/lib/python3.12/site-packages:${PYTHONPATH}"

# SwanLab 配置
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E
export SWANLAB_LOG_DIR="${CHECKPOINT_BASE_DIR}/swanlab"

echo ""
echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo ""

# 激活虚拟环境
if [ -f "${PROJECT_BASE_DIR}/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_BASE_DIR}/.venv/bin/activate"
fi

# ============================================
# 清理目标 GPU 上的进程
# ============================================
echo ""
echo "Cleaning up processes on target GPUs [${CUDA_VISIBLE_DEVICES}]..."

# 将 GPU ID 字符串转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

# 获取目标 GPU 上的进程并杀死
for gpu_id in "${GPU_ARRAY[@]}"; do
    # 使用 nvidia-smi 获取 GPU 上的进程
    pids=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs)
    if [ -n "$pids" ]; then
        echo "  GPU $gpu_id: Killing processes $pids"
        for pid in $pids; do
            # 检查进程是否存在且属于当前用户
            if ps -p "$pid" -o user= 2>/dev/null | grep -q "^$(whoami)$"; then
                kill -9 "$pid" 2>/dev/null || true
                echo "    Killed process $pid"
            else
                echo "    Skipped process $pid (not owned by current user or not exists)"
            fi
        done
    else
        echo "  GPU $gpu_id: No processes found"
    fi
done

# 等待进程完全退出
sleep 3

# ============================================
# 清理 Ray 进程
# ============================================
echo ""
echo "Cleaning up Ray processes..."

# 清理 Ray 工作进程
pkill -u "$(whoami)" -f "ray::" 2>/dev/null || true
pkill -u "$(whoami)" -f "raylet" 2>/dev/null || true
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray._private" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray::IDLE" 2>/dev/null || true
pkill -u "$(whoami)" -f "ray::Actor" 2>/dev/null || true
pkill -u "$(whoami)" -f "python.*ray" 2>/dev/null || true

# 清理 Ray 临时目录
rm -rf "$RAY_TMPDIR"/session_* 2>/dev/null || true
rm -rf "/tmp/ray"/* 2>/dev/null || true

# 等待进程完全退出
sleep 3

echo "Ray cleanup complete."

# ============================================
# 最终检查：再次确认目标 GPU 是否空闲
# ============================================
echo ""
echo "Verifying GPU status..."
for gpu_id in "${GPU_ARRAY[@]}"; do
    pids=$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs)
    if [ -n "$pids" ]; then
        echo "  WARNING: GPU $gpu_id still has processes: $pids"
    else
        echo "  GPU $gpu_id: Clean"
    fi
done

echo ""
echo "All cleanup complete. Ready to start training."
echo ""

# 启动训练
echo "Starting training..."
cd "${PROJECT_BASE_DIR}"

# 使用 nohup 后台运行，输出到日志文件
nohup python verl/verl/trainer/main_ppo.py \
    --config-path="${PROJECT_BASE_DIR}/examples" \
    --config-name="${CONFIG_NAME}" > "${LOG_FILE}" 2>&1 &

# 获取进程 PID
PID=$!
echo "Training started with PID: $PID"
echo "PID: $PID" >> "${LOG_FILE}"
echo ""
echo "To monitor the log: tail -f ${LOG_FILE}"
echo "To check process: ps -p $PID"
echo "To kill process: kill $PID"
