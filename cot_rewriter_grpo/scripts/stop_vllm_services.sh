#!/bin/bash
# Stop vLLM Services
#
# Usage:
#   bash scripts/stop_vllm_services.sh

PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
LOG_DIR="$PROJECT_ROOT/logs"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

# Stop Student vLLM
if [ -f "$LOG_DIR/student_vllm.pid" ]; then
    PID=$(cat "$LOG_DIR/student_vllm.pid")
    if kill -0 $PID 2>/dev/null; then
        log_info "Stopping Student vLLM (PID: $PID)..."
        kill $PID
    fi
    rm -f "$LOG_DIR/student_vllm.pid"
fi

# Stop FSF vLLM
if [ -f "$LOG_DIR/fsf_vllm.pid" ]; then
    PID=$(cat "$LOG_DIR/fsf_vllm.pid")
    if kill -0 $PID 2>/dev/null; then
        log_info "Stopping FSF vLLM (PID: $PID)..."
        kill $PID
    fi
    rm -f "$LOG_DIR/fsf_vllm.pid"
fi

# Also try to find and kill any remaining vLLM processes on common ports
for port in 8100 8101; do
    PID=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$PID" ]; then
        log_info "Stopping process on port $port (PID: $PID)..."
        kill $PID 2>/dev/null || true
    fi
done

log_info "All vLLM services stopped."
