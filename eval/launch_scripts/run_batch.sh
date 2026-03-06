#!/bin/bash
# run_batch.sh - 等待当前 GPU 0,1 上的训练进程（及祖先链）结束后，依次评测多个 checkpoint
# 启动方式: nohup bash launch_scripts/run_batch.sh > vllm_logs/batch_main.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0,1"
export no_proxy=localhost,127.0.0.1
export HF_HOME=/fs/fast/u20240075/hf_home
# 激活 uv 虚拟环境
source /home/u20240075/luoyashuo/projects/cdad/.venv/bin/activate

# 切换到 eval 根目录
cd "$(dirname "$0")/.." || exit 1

LOG_DIR="vllm_logs"
mkdir -p "$LOG_DIR"

BATCH_LOG="$LOG_DIR/batch_$(date '+%Y%m%d_%H%M%S').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# Step 1: 等待训练进程结束 (修改这里为实际的训练脚本 PID)
# ============================================================
TRAIN_PID=2854  # run_exps.sh 的 PID

log "等待训练进程 PID $TRAIN_PID 结束..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    log "训练仍在进行中，10s 后检查..."
    sleep 10
done
log "训练进程已结束，开始执行评测"

# ============================================================
# Step 2: 定义待评测的 checkpoint 列表及对应配置文件
# 格式: 每两个相邻元素为一组 (model_path, base_config)
# ============================================================
declare -a JOBS=(
    # "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"    "eval_configs/eval_all.yaml"
    "/fs/fast/u20240075/luoyashuo/cdad/checkpoints/rv_optimal/llama3_1b_instruct_rv_optimal_50k"  "eval_configs/eval_all_sft.yaml"
    "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct"    "eval_configs/eval_all.yaml"
)

NUM_RUNS=3

# ============================================================

TOTAL=$(( ${#JOBS[@]} / 2 ))
log "========== 批量评测开始，共 $TOTAL 个模型，每个运行 $NUM_RUNS 次 =========="

FAILED=0
for (( i=0; i<${#JOBS[@]}; i+=2 )); do
    MODEL_PATH="${JOBS[$i]}"
    BASE_CONFIG="${JOBS[$i+1]}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    JOB_NUM=$(( i/2 + 1 ))
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

    # 生成临时配置文件，将 model_path 替换为当前模型路径
    sed "s|^model_path:.*|model_path: ${MODEL_PATH}|" "$BASE_CONFIG" > "$TMP_CONFIG"

    python eval.py "$TMP_CONFIG" "$NUM_RUNS" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "  [完成] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 成功"
    else
        log "  [失败] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 退出码: $EXIT_CODE"
        FAILED=$((FAILED + 1))
    fi
    rm "$TMP_CONFIG"
done

log "========== 批量评测结束：$TOTAL 个模型，失败 $FAILED 个 =========="
