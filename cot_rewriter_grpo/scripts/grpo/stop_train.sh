#!/bin/bash

# ============================================================================
# Stop CoT Rewriter GRPO Training
# ============================================================================

set -e

USER=$(whoami)

echo "========================================================================"
echo "Stopping CoT Rewriter GRPO Training Processes"
echo "========================================================================"
echo "User: ${USER}"
echo ""

# 函数：查找并终止 GRPO 训练进程
kill_grpo_processes() {
    local pids=$(ps aux | grep -E "(train\.py|main_ppo\.py|run_ppo)" | grep -v grep | grep "${USER}" | awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo "No GRPO training processes found."
        return 0
    fi
    
    echo "Found GRPO training processes:"
    ps aux | grep -E "(train\.py|main_ppo\.py|run_ppo)" | grep -v grep | grep "${USER}"
    echo ""
    
    echo "Terminating processes..."
    for pid in $pids; do
        echo "  Killing PID: $pid"
        kill -TERM "$pid" 2>/dev/null || true
    done
    
    # 等待进程终止
    sleep 2
    
    # 强制终止仍未结束的进程
    for pid in $pids; do
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "  Force killing PID: $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    echo "GRPO training processes terminated."
}

# 函数：清理 Ray 进程
cleanup_ray() {
    echo ""
    echo "Cleaning up Ray processes..."
    
    # 查找用户的 Ray 进程
    local ray_pids=$(ps aux | grep -E "(ray::|raylet|plasma_store)" | grep -v grep | grep "${USER}" | awk '{print $2}')
    
    if [ -n "$ray_pids" ]; then
        echo "Found Ray processes:"
        ps aux | grep -E "(ray::|raylet|plasma_store)" | grep -v grep | grep "${USER}"
        echo ""
        
        for pid in $ray_pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
        echo "Ray processes cleaned up."
    else
        echo "No Ray processes found."
    fi
}

# 函数：清理 GPU 上的用户进程
cleanup_gpu_processes() {
    echo ""
    echo "Checking for GPU processes..."
    
    # 获取用户在 GPU 上的进程
    local gpu_pids=$(nvidia-smi pmon -c 1 2>/dev/null | grep -v "#" | grep -v "Idx" | awk '{print $2}' | grep -v "-" | sort -u)
    
    local killed_count=0
    for pid in $gpu_pids; do
        local process_user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$process_user" = "$USER" ]; then
            local process_name=$(ps -o comm= -p "$pid" 2>/dev/null)
            echo "  Killing GPU process: PID=$pid, Name=$process_name"
            kill -9 "$pid" 2>/dev/null || true
            killed_count=$((killed_count + 1))
        fi
    done
    
    if [ $killed_count -eq 0 ]; then
        echo "No user GPU processes found."
    else
        echo "Killed $killed_count GPU process(es)."
    fi
}

# 主执行流程
echo "Step 1: Stopping GRPO training processes..."
kill_grpo_processes

echo ""
echo "Step 2: Cleaning up Ray processes..."
cleanup_ray

echo ""
echo "Step 3: Cleaning up GPU processes..."
cleanup_gpu_processes

echo ""
echo "========================================================================"
echo "Cleanup completed!"
echo "========================================================================"

# 显示当前 GPU 状态
echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader | while read line; do
    echo "  GPU: ${line}"
done
