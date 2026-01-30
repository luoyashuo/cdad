#!/bin/bash

# Start Student Reward Server on a dedicated GPU
#
# This server runs the Student Model on GPU and provides HTTP API for reward computation.
# RewardLoopWorker (which doesn't have GPU access) can call this server to compute rewards.
#
# Usage:
#   bash scripts/grpo/start_reward_server.sh [GPU_ID] [PORT] [MODEL_PATH]
#
# Examples:
#   bash scripts/grpo/start_reward_server.sh           # Use defaults (GPU 5, port 8765)
#   bash scripts/grpo/start_reward_server.sh 5 8765    # Specify GPU and port
#   bash scripts/grpo/start_reward_server.sh 5 8765 /path/to/model

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default parameters
GPU_ID="${1:-5}"
PORT="${2:-8765}"
MODEL_PATH="${3:-/data-share/LLM/Qwen3-1.7B}"

# Log configuration
LOG_DIR="${PROJECT_ROOT}/logs/reward_server"
LOG_FILE="${LOG_DIR}/student_reward_server_gpu${GPU_ID}.log"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Student Reward Server"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "Port: ${PORT}"
echo "Model: ${MODEL_PATH}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

# Activate virtual environment if exists
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

# Start the server with specified GPU
echo "Starting server..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.reward.student_reward_server \
    --model_path "${MODEL_PATH}" \
    --port ${PORT} \
    --device_id 0 \
    --dtype bfloat16 \
    2>&1 | tee "${LOG_FILE}"
