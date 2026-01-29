#!/bin/bash

# ============================================================================
# Check CoT Rewriter GRPO Training Status
# ============================================================================

USER=$(whoami)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "CoT Rewriter GRPO Training Status"
echo "========================================================================"
echo "User: ${USER}"
echo ""

# 检查 GRPO 训练进程
echo "1. Training Processes:"
echo "   -------------------"
local grpo_pids=$(ps aux | grep -E "(train\.py|main_ppo\.py|run_ppo)" | grep -v grep | grep "${USER}")
if [ -n "$grpo_pids" ]; then
    echo -e "   ${GREEN}Running${NC}:"
    ps aux | grep -E "(train\.py|main_ppo\.py|run_ppo)" | grep -v grep | grep "${USER}" | while read line; do
        echo "     ${line}"
    done
else
    echo -e "   ${YELLOW}Not running${NC}"
fi
echo ""

# 检查 Ray 进程
echo "2. Ray Processes:"
echo "   --------------"
local ray_pids=$(ps aux | grep -E "(ray::|raylet)" | grep -v grep | grep "${USER}" | wc -l)
if [ "$ray_pids" -gt 0 ]; then
    echo -e "   ${GREEN}Running${NC}: $ray_pids process(es)"
    ps aux | grep -E "(ray::|raylet)" | grep -v grep | grep "${USER}" | head -5 | while read line; do
        echo "     ${line}"
    done
    if [ "$ray_pids" -gt 5 ]; then
        echo "     ... and $((ray_pids - 5)) more"
    fi
else
    echo -e "   ${YELLOW}Not running${NC}"
fi
echo ""

# 检查 GPU 状态
echo "3. GPU Status:"
echo "   -----------"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=',' read -r idx name temp util mem_used mem_total; do
    idx=$(echo "$idx" | xargs)
    name=$(echo "$name" | xargs)
    temp=$(echo "$temp" | xargs)
    util=$(echo "$util" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    
    # 检查该 GPU 上是否有用户的进程
    local user_procs=$(nvidia-smi pmon -i "$idx" -c 1 2>/dev/null | grep -v "#" | grep -v "Idx" | awk '{print $2}' | grep -v "-" | while read pid; do
        local proc_user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$proc_user" = "$USER" ]; then
            echo "1"
            break
        fi
    done)
    
    if [ -n "$user_procs" ]; then
        echo -e "   GPU ${idx}: ${GREEN}[ACTIVE]${NC} ${name}"
    else
        echo -e "   GPU ${idx}: ${YELLOW}[IDLE]${NC} ${name}"
    fi
    echo "           Temp: ${temp}°C | Util: ${util} | Memory: ${mem_used} / ${mem_total}"
done
echo ""

# 检查最近的日志
echo "4. Recent Logs:"
echo "   -----------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/grpo"

if [ -d "$LOG_DIR" ]; then
    local latest_logs=$(ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -3)
    if [ -n "$latest_logs" ]; then
        echo "$latest_logs" | while read line; do
            echo "   ${line}"
        done
    else
        echo "   No log files found"
    fi
else
    echo "   Log directory not found: ${LOG_DIR}"
fi
echo ""

# 检查输出目录
echo "5. Output Directory:"
echo "   ----------------"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/cot_rewriter_grpo"
if [ -d "$OUTPUT_DIR" ]; then
    local recent_dirs=$(ls -lt "$OUTPUT_DIR" 2>/dev/null | head -5)
    if [ -n "$recent_dirs" ]; then
        echo "   Recent experiments:"
        ls -lt "$OUTPUT_DIR" 2>/dev/null | grep "^d" | head -5 | while read line; do
            echo "     ${line}"
        done
    else
        echo "   No output directories found"
    fi
else
    echo "   Output directory not found: ${OUTPUT_DIR}"
fi
echo ""

echo "========================================================================"
