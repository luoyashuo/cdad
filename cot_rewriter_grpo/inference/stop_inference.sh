#!/bin/bash
# =============================================================================
# 推理进程停止脚本
# =============================================================================
# 功能：一键停止所有与 rewrite_dataset.py 相关的进程
# =============================================================================

echo "========================================"
echo "停止推理进程"
echo "========================================"

# 获取项目目录名（用于匹配进程）
PROJECT_NAME="cot_rewriter_grpo"

# 1. 停止 Python 推理脚本（通过脚本名）
echo "[1/7] 停止 rewrite_dataset.py 进程..."
pkill -9 -f "rewrite_dataset.py" 2>/dev/null && echo "  已停止 rewrite_dataset.py" || echo "  未发现 rewrite_dataset.py 进程"

# 2. 停止 multiprocessing 子进程（通过项目路径匹配）
echo "[2/7] 停止 multiprocessing 子进程..."
pkill -9 -f ".venv/bin/python.*multiprocessing-fork" 2>/dev/null && echo "  已停止 multiprocessing 子进程" || echo "  未发现 multiprocessing 子进程"

# 3. 停止项目目录下的所有 Python 进程（兜底方案）
echo "[3/7] 停止项目目录下的 Python 进程..."
# 通过 /proc/*/cmdline 查找并停止项目目录下的 Python 进程
stopped_count=0
for pid_dir in /proc/[0-9]*; do
    pid=$(basename "$pid_dir")
    if [ -f "$pid_dir/cmdline" ] && [ -r "$pid_dir/cmdline" ]; then
        cmdline=$(cat "$pid_dir/cmdline" 2>/dev/null | tr '\0' ' ')
        if echo "$cmdline" | grep -q "$PROJECT_NAME.*python" && ! echo "$cmdline" | grep -q "grep"; then
            kill -9 "$pid" 2>/dev/null && ((stopped_count++))
        fi
    fi
done
if [ $stopped_count -gt 0 ]; then
    echo "  已停止 $stopped_count 个 Python 进程"
else
    echo "  未发现项目目录下的 Python 进程"
fi

# 4. 停止 Ray Actors
echo "[4/7] 停止 Ray Actors..."
pkill -9 -f "ray::" 2>/dev/null && echo "  已停止 Ray Actors" || echo "  未发现 Ray Actors"

# 5. 停止 SGLang 进程
echo "[5/7] 停止 SGLang 进程..."
pkill -9 -f "sglang::" 2>/dev/null && echo "  已停止 SGLang" || echo "  未发现 SGLang 进程"

# 6. 停止 vLLM 进程
echo "[6/7] 停止 vLLM 进程..."
pkill -9 -f "vllm.entrypoints" 2>/dev/null && echo "  已停止 vLLM" || echo "  未发现 vLLM 进程"

# 7. 停止 Ray 核心进程
echo "[7/7] 停止 Ray 核心进程..."
pkill -9 -f "raylet" 2>/dev/null && echo "  已停止 raylet" || echo "  未发现 raylet"
pkill -9 -f "gcs_server" 2>/dev/null && echo "  已停止 gcs_server" || echo "  未发现 gcs_server"

# 8. 清理 Ray 临时文件
echo "[8/8] 清理 Ray 临时文件..."
rm -rf /tmp/ray_* 2>/dev/null && echo "  已清理 /tmp/ray_*" || echo "  清理失败或无需清理"

echo "========================================"
echo "停止完成！剩余相关进程:"
ps aux | grep -E "rewrite_dataset|multiprocessing-fork|$PROJECT_NAME.*python" | grep -v grep | grep -v "stop_inference" || echo "  无相关进程"
echo "========================================"

# 显示显存使用情况
echo "当前 GPU 使用情况:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  无法获取 GPU 信息"
