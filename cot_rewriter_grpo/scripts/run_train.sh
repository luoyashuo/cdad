#!/bin/bash
# CoT Rewriter GRPO Training Launch Script
#
# This script:
# 1. Starts vLLM services (Student Model + FSF Judge)
# 2. Waits for services to be ready
# 3. Launches VERL training
#
# Usage:
#   bash scripts/run_train.sh examples/qwen3_8b_grpo_4gpu.yaml
#   bash scripts/run_train.sh examples/qwen3_8b_grpo_2gpu_lora.yaml
#
# Environment Variables:
#   STUDENT_MODEL_PATH: Path to Student Model (default: ./models/Qwen2.5-3B-Instruct)
#   FSF_MODEL_PATH:     Path to FSF Judge Model (default: ./models/Qwen2.5-7B-Instruct)
#   ACTOR_MODEL_PATH:   Path to Actor Model (default: ./models/Qwen3-8B)
#   STUDENT_GPU:        GPU for Student Model (default: 3)
#   FSF_GPU:            GPU for FSF Judge (use same as student, or separate)
#   STUDENT_PORT:       Port for Student vLLM (default: 8100)
#   FSF_PORT:           Port for FSF vLLM (default: 8101)
#   SKIP_VLLM:          Skip vLLM startup if services already running

set -e

# ============ Configuration ============
CONFIG_FILE=${1:-"examples/qwen3_8b_grpo_4gpu.yaml"}
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
cd $PROJECT_ROOT

# Model paths
STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"./models/Qwen2.5-3B-Instruct"}
FSF_MODEL_PATH=${FSF_MODEL_PATH:-"./models/Qwen2.5-7B-Instruct"}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"./models/Qwen3-8B"}

# GPU configuration
STUDENT_GPU=${STUDENT_GPU:-3}
FSF_GPU=${FSF_GPU:-3}  # Can share with Student if memory allows

# Port configuration
STUDENT_PORT=${STUDENT_PORT:-8100}
FSF_PORT=${FSF_PORT:-8101}

# Other settings
SKIP_VLLM=${SKIP_VLLM:-false}
LOG_DIR="logs"
mkdir -p $LOG_DIR

# ============ Helper Functions ============

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-60}
    local attempt=0

    log_info "Waiting for $name to be ready at $url..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url/models" > /dev/null 2>&1; then
            log_info "$name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    log_error "$name failed to start after $max_attempts attempts"
    return 1
}

cleanup() {
    log_info "Cleaning up..."
    if [ -n "$STUDENT_PID" ]; then
        kill $STUDENT_PID 2>/dev/null || true
    fi
    if [ -n "$FSF_PID" ]; then
        kill $FSF_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

# ============ Check Prerequisites ============

log_info "=========================================="
log_info "CoT Rewriter GRPO Training"
log_info "=========================================="
log_info "Config: $CONFIG_FILE"
log_info "Student Model: $STUDENT_MODEL_PATH"
log_info "FSF Model: $FSF_MODEL_PATH"
log_info "Actor Model: $ACTOR_MODEL_PATH"
log_info "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if models exist
for model_path in "$STUDENT_MODEL_PATH" "$ACTOR_MODEL_PATH"; do
    if [ ! -d "$model_path" ]; then
        log_error "Model not found: $model_path"
        log_error "Please download the model or update the path."
        exit 1
    fi
done

# ============ Start vLLM Services ============

if [ "$SKIP_VLLM" != "true" ]; then
    log_info "Starting vLLM services..."

    # Start Student Model vLLM
    log_info "Starting Student Model vLLM on GPU $STUDENT_GPU, port $STUDENT_PORT..."
    CUDA_VISIBLE_DEVICES=$STUDENT_GPU python -m vllm.entrypoints.openai.api_server \
        --model $STUDENT_MODEL_PATH \
        --port $STUDENT_PORT \
        --gpu-memory-utilization 0.45 \
        --trust-remote-code \
        --dtype bfloat16 \
        --disable-log-requests \
        > $LOG_DIR/student_vllm.log 2>&1 &
    STUDENT_PID=$!
    log_info "Student vLLM started with PID $STUDENT_PID"

    # Start FSF Judge vLLM (can share GPU with Student if different ports)
    if [ "$FSF_MODEL_PATH" != "$STUDENT_MODEL_PATH" ] || [ "$FSF_PORT" != "$STUDENT_PORT" ]; then
        log_info "Starting FSF Judge vLLM on GPU $FSF_GPU, port $FSF_PORT..."

        # If sharing GPU, reduce memory utilization
        if [ "$FSF_GPU" == "$STUDENT_GPU" ]; then
            FSF_MEM_UTIL=0.45
        else
            FSF_MEM_UTIL=0.9
        fi

        CUDA_VISIBLE_DEVICES=$FSF_GPU python -m vllm.entrypoints.openai.api_server \
            --model $FSF_MODEL_PATH \
            --port $FSF_PORT \
            --gpu-memory-utilization $FSF_MEM_UTIL \
            --trust-remote-code \
            --dtype bfloat16 \
            --disable-log-requests \
            > $LOG_DIR/fsf_vllm.log 2>&1 &
        FSF_PID=$!
        log_info "FSF vLLM started with PID $FSF_PID"
    else
        log_info "FSF Judge using same service as Student Model"
        FSF_PID=""
    fi

    # Wait for services to be ready
    wait_for_service "http://localhost:$STUDENT_PORT/v1" "Student vLLM" 120
    if [ -n "$FSF_PID" ]; then
        wait_for_service "http://localhost:$FSF_PORT/v1" "FSF vLLM" 120
    fi

    log_info "All vLLM services are ready!"
else
    log_info "Skipping vLLM startup (SKIP_VLLM=true)"
fi

# ============ Start Training ============

log_info "Starting VERL training..."

# Set training GPUs (exclude vLLM GPU)
if [ "$STUDENT_GPU" == "3" ]; then
    TRAINING_GPUS="0,1,2"
elif [ "$STUDENT_GPU" == "0" ]; then
    TRAINING_GPUS="1,2,3"
else
    TRAINING_GPUS="0,1,2"
fi

log_info "Training GPUs: $TRAINING_GPUS"

# Add src to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

# Run training with hydra
CUDA_VISIBLE_DEVICES=$TRAINING_GPUS python src/trainer/train.py \
    --config-path="$PROJECT_ROOT/examples" \
    --config-name="$(basename $CONFIG_FILE .yaml)" \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    reward_model.custom_reward_function.reward_kwargs.student_vllm_url="http://localhost:$STUDENT_PORT/v1" \
    reward_model.custom_reward_function.reward_kwargs.fsf_vllm_url="http://localhost:$FSF_PORT/v1" \
    "$@"

log_info "Training completed!"
