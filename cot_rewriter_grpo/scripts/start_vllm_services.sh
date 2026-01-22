#!/bin/bash
# Start vLLM Services for CoT Rewriter GRPO
#
# This script starts the vLLM inference services:
# - Student Model: For computing cognitive reward
# - FSF Judge Model: For computing logic reward (FSF analysis)
#
# Usage:
#   bash scripts/start_vllm_services.sh [student|fsf|both]
#
# Modes:
#   student: Start only Student Model service
#   fsf:     Start only FSF Judge service
#   both:    Start both services (default)
#
# Environment Variables:
#   STUDENT_MODEL_PATH: Path to Student Model (default: ./models/Qwen2.5-3B-Instruct)
#   FSF_MODEL_PATH:     Path to FSF Judge Model (default: ./models/Qwen2.5-7B-Instruct)
#   STUDENT_GPU:        GPU for Student Model (default: 3)
#   FSF_GPU:            GPU for FSF Judge (default: 3)
#   STUDENT_PORT:       Port for Student vLLM (default: 8100)
#   FSF_PORT:           Port for FSF vLLM (default: 8101)
#   STUDENT_MEM:        GPU memory utilization for Student (default: 0.45)
#   FSF_MEM:            GPU memory utilization for FSF (default: 0.45)

set -e

MODE=${1:-both}
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_ROOT

# ============ Configuration ============

STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"./models/Qwen2.5-3B-Instruct"}
FSF_MODEL_PATH=${FSF_MODEL_PATH:-"./models/Qwen2.5-7B-Instruct"}

STUDENT_GPU=${STUDENT_GPU:-3}
FSF_GPU=${FSF_GPU:-3}

STUDENT_PORT=${STUDENT_PORT:-8100}
FSF_PORT=${FSF_PORT:-8101}

STUDENT_MEM=${STUDENT_MEM:-0.45}
FSF_MEM=${FSF_MEM:-0.45}

LOG_DIR="logs"
mkdir -p $LOG_DIR

# ============ Helper Functions ============

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

start_student_vllm() {
    log_info "=========================================="
    log_info "Starting Student Model vLLM Service"
    log_info "Model: $STUDENT_MODEL_PATH"
    log_info "GPU: $STUDENT_GPU"
    log_info "Port: $STUDENT_PORT"
    log_info "Memory: $STUDENT_MEM"
    log_info "=========================================="

    if ! [ -d "$STUDENT_MODEL_PATH" ]; then
        log_error "Student model not found: $STUDENT_MODEL_PATH"
        exit 1
    fi

    if check_port $STUDENT_PORT; then
        log_info "Port $STUDENT_PORT is already in use. Student service may already be running."
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$STUDENT_GPU python -m vllm.entrypoints.openai.api_server \
        --model $STUDENT_MODEL_PATH \
        --port $STUDENT_PORT \
        --gpu-memory-utilization $STUDENT_MEM \
        --trust-remote-code \
        --dtype bfloat16 \
        --disable-log-requests \
        --served-model-name "Qwen2.5-3B-Instruct" \
        2>&1 | tee $LOG_DIR/student_vllm.log &

    STUDENT_PID=$!
    echo $STUDENT_PID > $LOG_DIR/student_vllm.pid
    log_info "Student vLLM started with PID $STUDENT_PID"
    log_info "Logs: $LOG_DIR/student_vllm.log"
}

start_fsf_vllm() {
    log_info "=========================================="
    log_info "Starting FSF Judge vLLM Service"
    log_info "Model: $FSF_MODEL_PATH"
    log_info "GPU: $FSF_GPU"
    log_info "Port: $FSF_PORT"
    log_info "Memory: $FSF_MEM"
    log_info "=========================================="

    if ! [ -d "$FSF_MODEL_PATH" ]; then
        log_error "FSF model not found: $FSF_MODEL_PATH"
        exit 1
    fi

    if check_port $FSF_PORT; then
        log_info "Port $FSF_PORT is already in use. FSF service may already be running."
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$FSF_GPU python -m vllm.entrypoints.openai.api_server \
        --model $FSF_MODEL_PATH \
        --port $FSF_PORT \
        --gpu-memory-utilization $FSF_MEM \
        --trust-remote-code \
        --dtype bfloat16 \
        --disable-log-requests \
        --served-model-name "Qwen2.5-7B-Instruct" \
        2>&1 | tee $LOG_DIR/fsf_vllm.log &

    FSF_PID=$!
    echo $FSF_PID > $LOG_DIR/fsf_vllm.pid
    log_info "FSF vLLM started with PID $FSF_PID"
    log_info "Logs: $LOG_DIR/fsf_vllm.log"
}

# ============ Main ============

case "$MODE" in
    student)
        start_student_vllm
        ;;
    fsf)
        start_fsf_vllm
        ;;
    both)
        start_student_vllm
        sleep 2
        start_fsf_vllm
        ;;
    *)
        log_error "Unknown mode: $MODE"
        echo "Usage: $0 [student|fsf|both]"
        exit 1
        ;;
esac

log_info "=========================================="
log_info "vLLM Services Started"
log_info ""
log_info "To check service status:"
log_info "  curl http://localhost:$STUDENT_PORT/v1/models"
log_info "  curl http://localhost:$FSF_PORT/v1/models"
log_info ""
log_info "To stop services:"
log_info "  bash scripts/stop_vllm_services.sh"
log_info "=========================================="
