#!/bin/bash
# run_batch.sh - 等待训练进程结束后，依次评测多个 checkpoint
# 启动方式: nohup bash launch_scripts/run_batch.sh > vllm_logs/batch_main.log 2>&1 &

# export CUDA_VISIBLE_DEVICES="0"
export no_proxy=localhost,127.0.0.1
export HF_HOME=/data/hf_home
export MODELSCOPE_CACHE=/data/modelscope
export HF_ENDPOINT=https://hf-mirror.com
# 激活 uv 虚拟环境
source /home/luoys/cdad/.venv/bin/activate

# 切换到 eval 根目录
cd "$(dirname "$0")/.." || exit 1

LOG_DIR="vllm_logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# GPU 资源管理工具函数
# ============================================================

# 等待 GPU 显存释放：所有 GPU 的最小空闲显存 >= 阈值（MiB）
# 用法: wait_for_gpu_memory [threshold_gib]
GPU_MEM_THRESHOLD_GIB=30   # 默认：每张 GPU 至少需要 30 GiB 空闲

wait_for_gpu_memory() {
    local threshold_gib="${1:-$GPU_MEM_THRESHOLD_GIB}"
    local threshold_mib=$(( threshold_gib * 1024 ))
    local attempt=0

    log "等待 GPU 显存释放（阈值: ${threshold_gib} GiB / ${threshold_mib} MiB）..."

    while true; do
        local min_free
        min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
                   | awk 'BEGIN{m=-1} {if(m<0||$1<m) m=$1} END{print m}')

        if [ -z "$min_free" ] || [ "$min_free" -lt 0 ] 2>/dev/null; then
            log "  nvidia-smi 读取失败，10s 后重试..."
            sleep 10
            continue
        fi

        attempt=$(( attempt + 1 ))
        if [ "$min_free" -ge "$threshold_mib" ]; then
            log "  GPU 显存已充足：最小空闲 ${min_free} MiB >= ${threshold_mib} MiB，继续"
            return 0
        fi

        log "  [等待中 #${attempt}] 最小空闲显存 ${min_free} MiB < ${threshold_mib} MiB，15s 后再检查..."
        sleep 15
    done
}

# 清理残留的 Ray 进程（前一个 eval job 未正常退出时可能遗留）
kill_stray_ray() {
    log "清理残留 Ray 进程..."
    # 杀掉本用户下的 ray 子进程，不影响其他用户
    pkill -u "$(id -u)" -f "ray::" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/dashboard" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/log_monitor" 2>/dev/null || true
    pkill -u "$(id -u)" -f "gcs_server" 2>/dev/null || true
    sleep 3
    log "Ray 进程清理完成"
}

# ============================================================
# Step 1: 等待训练进程结束 (修改这里为实际的训练脚本 PID)
# ============================================================
TRAIN_PID=3298868  # run_exps.sh 的 PID

log "等待训练进程 PID $TRAIN_PID 结束..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    log "训练仍在进行中，10s 后检查..."
    sleep 10
done
log "训练进程已结束"

# 训练进程退出后，GPU 显存释放可能有延迟，等待显存真正空闲再开始评测
wait_for_gpu_memory

# ============================================================
# Step 2: 定义待评测的 checkpoint 列表及对应配置文件
# 格式: 每两个相邻元素为一组 (model_path, base_config)
# ============================================================
declare -a JOBS=(
#     "/data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct"    "eval_configs/eval_all_sft.yaml"
#     "/data/modelscope_cache/models/Qwen/Qwen2.5-3B-Instruct"    "eval_configs/eval_all_sft.yaml"
    "/data/cdad/checkpoints/vallina_sft/qwen2.5_3b_instruct_full_sft_30k"    "eval_configs/eval_all_sft.yaml"
    # "/data/cdad/checkpoints/vallina_sft/qwen2.5_7b_instruct_full_sft_30k"    "eval_configs/eval_all_sft.yaml"

    # "/data/cdad/checkpoints/vallina_sft/qwen2.5_7b_instruct_lora_sft_30k/merged"    "eval_configs/eval_math.yaml"
    # "/data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct"    "eval_configs/eval_math.yaml"
    # "/data/modelscope_cache/models/Qwen/Qwen2.5-1.5B-Instruct"    "eval_configs/eval_all_sft.yaml"
    # "/data/cdad/checkpoints/vallina_sft/qwen2.5_1.5b_instruct_vallina_sft_50k"    "eval_configs/eval_all_sft.yaml"


    
)

NUM_RUNS=1

# 断点续传配置（二选一，同时设置时 RESUME_DIRS 优先）：
#   RESUME=1               每个模型自动检测最新 version_* 目录续跑
#   RESUME_DIRS=(...)      按顺序为每个模型指定精确的版本目录（空字符串 "" 表示该模型全新跑）
#
# 示例（指定部分目录）：
#   RESUME_DIRS=(
#       "/data/.../Qwen2.5-1.5B-Instruct/evalscope/version_20260310_043148"
#       ""                                  # 3B 模型全新跑
#       "/data/.../vallina_sft/evalscope/version_20260310_001550"
#   )
RESUME=1
RESUME_DIRS=(
    "/data/cdad/checkpoints/vallina_sft/qwen2.5_3b_instruct_full_sft_30k/evalscope/version_20260311_125756"
)

# ============================================================

TOTAL=$(( ${#JOBS[@]} / 2 ))

if [ ${#RESUME_DIRS[@]} -gt 0 ]; then
    _RESUME_HINT="，断点续传（指定目录）"
elif [ "$RESUME" -eq 1 ]; then
    _RESUME_HINT="，断点续传（自动检测）"
else
    _RESUME_HINT=""
fi

log "========== 批量评测开始，共 $TOTAL 个模型，每个运行 $NUM_RUNS 次${_RESUME_HINT} =========="

FAILED=0
declare -a SUCCEEDED_PATHS=()

for (( i=0; i<${#JOBS[@]}; i+=2 )); do
    MODEL_PATH="${JOBS[$i]}"
    BASE_CONFIG="${JOBS[$i+1]}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    JOB_NUM=$(( i/2 + 1 ))
    JOB_IDX=$(( i / 2 ))
    LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"
    TMP_CONFIG="eval_configs/${MODEL_NAME}.yaml"

    log "---------- 任务 $JOB_NUM/$TOTAL ----------"
    log "  模型路径 : $MODEL_PATH"
    log "  基础配置 : $BASE_CONFIG"
    log "  任务日志 : $LOG_FILE"

    if [ ! -f "$BASE_CONFIG" ]; then
        log "  [错误] 基础配置文件不存在: $BASE_CONFIG"
        exit 1
    fi

    # 构造本次任务的 resume 参数
    THIS_RESUME_FLAG=""
    if [ ${#RESUME_DIRS[@]} -gt 0 ]; then
        THIS_DIR="${RESUME_DIRS[$JOB_IDX]:-}"
        if [ -n "$THIS_DIR" ]; then
            THIS_RESUME_FLAG="--resume-dir ${THIS_DIR}"
            log "  断点续传 : 指定目录 $THIS_DIR"
        else
            log "  断点续传 : 未指定，全新运行"
        fi
    elif [ "$RESUME" -eq 1 ]; then
        THIS_RESUME_FLAG="--resume"
        log "  断点续传 : 自动检测"
    fi

    # 生成临时配置文件，将 model_path 替换为当前模型路径
    sed "s|^model_path:.*|model_path: ${MODEL_PATH}|" "$BASE_CONFIG" > "$TMP_CONFIG"

    # 断点续传时追加日志（保留上次日志），首次运行时新建日志
    if [ -n "$THIS_RESUME_FLAG" ]; then
        LOG_REDIRECT=">>"
    else
        LOG_REDIRECT=">"
    fi
    # shellcheck disable=SC2086
    eval python eval.py "$TMP_CONFIG" "$NUM_RUNS" $THIS_RESUME_FLAG $LOG_REDIRECT "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "  [完成] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 成功"
        SUCCEEDED_PATHS+=("$MODEL_PATH")
    else
        log "  [失败] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 退出码: $EXIT_CODE"
        FAILED=$((FAILED + 1))
    fi
    rm -f "$TMP_CONFIG"

    # 任务结束后清理残留 Ray 进程，等待 GPU 显存释放，再启动下一个任务
    if [ $(( i + 2 )) -lt ${#JOBS[@]} ]; then
        kill_stray_ray
        wait_for_gpu_memory
    fi
done

log "========== 批量评测结束：$TOTAL 个模型，失败 $FAILED 个 =========="

# ============================================================
# Step 3: 汇总所有成功模型的结果到一个 Excel 文件
# ============================================================
if [ ${#SUCCEEDED_PATHS[@]} -gt 0 ]; then
    SUMMARY_XLSX="$LOG_DIR/batch_summary_$(date '+%Y%m%d_%H%M%S').xlsx"
    log "========== 开始汇总结果到 $SUMMARY_XLSX =========="
    python collect_results.py "${SUCCEEDED_PATHS[@]}" \
        --output "$SUMMARY_XLSX" \
        --num-runs "$NUM_RUNS"
    if [ $? -eq 0 ]; then
        log "========== 汇总完成：$SUMMARY_XLSX =========="
    else
        log "========== 汇总失败，请手动运行 collect_results.py =========="
    fi
else
    log "没有成功完成的任务，跳过汇总。"
fi
