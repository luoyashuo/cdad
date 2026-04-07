#!/bin/bash
# run_batch_v2.sh - 批量运行 eval_v2.py（跨 Benchmark 流式推理版本）
# 
# 启动方式: 
#   nohup bash launch_scripts/run_batch_v2.sh > vllm_logs/batch_v2_main.log 2>&1 &
#   或前台运行:
#   bash launch_scripts/run_batch_v2.sh 2>&1 | tee vllm_logs/batch_v2_main.log

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
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

# 清理残留的 Ray 进程
kill_stray_ray() {
    log "清理残留 Ray 进程..."
    pkill -u "$(id -u)" -f "ray::" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/dashboard" 2>/dev/null || true
    pkill -u "$(id -u)" -f "ray/log_monitor" 2>/dev/null || true
    pkill -u "$(id -u)" -f "gcs_server" 2>/dev/null || true
    sleep 3
    log "Ray 进程清理完成"
}

# ============================================================
# 可选：等待训练进程结束
# ============================================================
# 如果需要等待训练结束，取消下面两行的注释并设置正确的 PID
TRAIN_PID=3991719  # 训练脚本 PID

if [ -n "$TRAIN_PID" ]; then
    log "等待训练进程 PID $TRAIN_PID 结束..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        log "训练仍在进行中，10s 后检查..."
        sleep 10
    done
    log "训练进程已结束"
fi

# ============================================================
# LoRA merge 工具函数
# ============================================================
maybe_merge_lora() {
    local model_path="$1"

    if [ ! -f "${model_path}/adapter_config.json" ]; then
        return 0   # 非 LoRA 格式，直接跳过
    fi

    if [ -f "${model_path}/config.json" ]; then
        log "  检测到 LoRA 格式，但已存在 config.json，跳过 merge"
        return 0
    fi

    log "  检测到未 merge 的 LoRA，开始合并权重到 ${model_path} ..."

    LORA_MERGE_PATH="$model_path" python - <<'PYEOF'
import os, json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = os.environ["LORA_MERGE_PATH"]

with open(os.path.join(model_path, "adapter_config.json")) as f:
    adapter_config = json.load(f)

base_model_path = adapter_config["base_model_name_or_path"]
print(f"  基础模型: {base_model_path}", flush=True)
print(f"  合并目标: {model_path}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("  merge 完成", flush=True)
PYEOF

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "  [错误] LoRA merge 失败（退出码: $exit_code），终止当前任务"
        return 1
    fi
    log "  LoRA merge 完成，合并权重已保存到 $model_path"
    return 0
}

# ============================================================
# Step 1: 定义待评测的 checkpoint 列表及对应配置文件
# ============================================================
# 格式: 每两个相邻元素为一组 (model_path, base_config)
declare -a JOBS=(
    # "/data/cdad/checkpoints/rewrite_cot/qwen2.5_7b_instruct_lora_sft_dual_30k"    "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/vallina_sft/qwen2.5_7b_instruct_lora_sft_30k"          "eval_configs/eval_all_baseline.yaml"
    # "/data/cdad/checkpoints/rewrite_cot/qwen2.5_7b_instruct_lora_sft_mix_15k"       "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r25"       "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r50"       "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r75"       "eval_configs/eval_all_sft_v2.yaml"
    # "/data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"      "eval_configs/eval_all_baseline.yaml"
    # "/data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct"      "eval_configs/eval_all_baseline.yaml"
    # "/data/cdad/checkpoints/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r100"       "eval_configs/eval_all_sft_v2.yaml"
    # "/data/hf_home/models/Yaseal-llama3_1b_instruct_vallina_full_sft_30k"      "eval_configs/eval_all_sft_v2.yaml"
    # "/data/hf_home/models/Yaseal-llama3_3b_instruct_vallina_full_sft_30k"      "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/morsd/llama3_1b_instruct_full_sft_morsd"      "eval_configs/eval_all_sft_v2.yaml"
    # "/data/cdad/checkpoints/morsd/llama3_3b_instruct_full_sft_morsd"      "eval_configs/eval_all_sft_v2.yaml"
    # "/data/qzq/ckpt_mcckd_llama_1b"      "eval_configs/eval_all_sft_v2.yaml"
    # "/data/qzq/ckpt_mcckd_llama_3b"      "eval_configs/eval_all_sft_v2.yaml"
    "/data/cdad/checkpoints/vallina_sft/qwen2.5_7b_instruct_lora_sft_30k"          "eval_configs/eval_all_sft_v2.yaml"
    /data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct  "eval_configs/eval_all_baseline.yaml"
    /data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct  "eval_configs/eval_all_sft_v2.yaml" 
)

# ============================================================
# Step 2: 断点续传配置
# ============================================================
# RESUME=1: 每个模型自动检测最新 version_* 目录续跑
# RESUME_DIRS: 为每个模型指定精确的版本目录（空字符串 "" 表示该模型全新跑）
#
# 示例（指定部分目录）:
#   RESUME_DIRS=(
#       "/data/.../model1/evalscope/version_20260318_003654"
#       ""                                  # model2 全新跑
#       "/data/.../model3/evalscope/version_20260318_005432"
#   )
RESUME=0
RESUME_DIRS=()

# FORCE_RESCORE=1: 强制重新评分，即使 repeats 数量与上次相同
# 默认情况下，如果 repeats 相同，会跳过评分直接复用上次结果
FORCE_RESCORE=0

# ============================================================
# Step 3: 自定义输出目录（可选）
# ============================================================
# 如果设置，所有评测结果将保存到这个目录下，而不是默认的 model_path/evalscope/
# OUTPUT_BASE_DIR="/data/eval_results/batch_v2"

# ============================================================

TOTAL=$(( ${#JOBS[@]} / 2 ))

if [ "$TOTAL" -eq 0 ]; then
    log "[错误] JOBS 数组为空，请添加要评测的模型路径和配置文件"
    exit 1
fi

if [ ${#RESUME_DIRS[@]} -gt 0 ]; then
    _RESUME_HINT="，断点续传（指定目录）"
elif [ "$RESUME" -eq 1 ]; then
    _RESUME_HINT="，断点续传（自动检测）"
else
    _RESUME_HINT=""
fi

log "========== 批量评测开始 (eval_v2.py)，共 $TOTAL 个模型${_RESUME_HINT} =========="

FAILED=0
declare -a SUCCEEDED_PATHS=()

for (( i=0; i<${#JOBS[@]}; i+=2 )); do
    MODEL_PATH="${JOBS[$i]}"
    BASE_CONFIG="${JOBS[$i+1]}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    JOB_NUM=$(( i/2 + 1 ))
    JOB_IDX=$(( i / 2 ))
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_v2.log"
    TMP_CONFIG="eval_configs/.tmp_${MODEL_NAME}_v2.yaml"

    log "---------- 任务 $JOB_NUM/$TOTAL ----------"
    log "  模型路径 : $MODEL_PATH"
    log "  基础配置 : $BASE_CONFIG"
    log "  任务日志 : $LOG_FILE"

    if [ ! -f "$BASE_CONFIG" ]; then
        log "  [错误] 基础配置文件不存在: $BASE_CONFIG"
        FAILED=$((FAILED + 1))
        continue
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

    # 构造输出目录参数（如果设置了 OUTPUT_BASE_DIR）
    OUTPUT_DIR_FLAG=""
    if [ -n "${OUTPUT_BASE_DIR:-}" ]; then
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_DIR_FLAG="--output-dir ${OUTPUT_DIR}"
        log "  输出目录 : $OUTPUT_DIR"
    fi

    # 构造强制重新评分参数
    FORCE_RESCORE_FLAG=""
    if [ "$FORCE_RESCORE" -eq 1 ]; then
        FORCE_RESCORE_FLAG="--force-rescore"
        log "  评分模式 : 强制重新评分"
    fi

    # 若是未 merge 的 LoRA，先自动 merge
    maybe_merge_lora "$MODEL_PATH" || { FAILED=$((FAILED + 1)); continue; }

    # 生成临时配置文件，将 model_path 替换为当前模型路径
    sed "s|^model_path:.*|model_path: ${MODEL_PATH}|" "$BASE_CONFIG" > "$TMP_CONFIG"

    # 运行 eval_v2.py
    log "  开始运行 eval_v2.py..."
    # shellcheck disable=SC2086
    python -u eval_v2.py "$TMP_CONFIG" $THIS_RESUME_FLAG $OUTPUT_DIR_FLAG $FORCE_RESCORE_FLAG > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "  [完成] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 成功"
        SUCCEEDED_PATHS+=("$MODEL_PATH")
    else
        log "  [失败] 任务 $JOB_NUM/$TOTAL ($MODEL_NAME) 退出码: $EXIT_CODE"
        log "  查看日志: tail -100 $LOG_FILE"
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
# Step 4: 汇总所有成功模型的结果
# ============================================================
if [ ${#SUCCEEDED_PATHS[@]} -gt 0 ]; then
    SUMMARY_XLSX="$LOG_DIR/batch_v2_summary_$(date '+%Y%m%d_%H%M%S').xlsx"
    log "========== 开始汇总结果到 $SUMMARY_XLSX =========="
    
    # 使用 collect_results.py 汇总（它会查找每个模型的最新 evalscope 目录）
    python collect_results.py "${SUCCEEDED_PATHS[@]}" \
        --output "$SUMMARY_XLSX"
    
    if [ $? -eq 0 ]; then
        log "========== 汇总完成：$SUMMARY_XLSX =========="
    else
        log "========== 汇总失败，请手动运行 collect_results.py =========="
    fi
else
    log "没有成功完成的任务，跳过汇总。"
fi

# 输出每个成功模型的详细结果路径
if [ ${#SUCCEEDED_PATHS[@]} -gt 0 ]; then
    log ""
    log "成功模型的详细结果位置："
    for path in "${SUCCEEDED_PATHS[@]}"; do
        MODEL_NAME=$(basename "$path")
        if [ -n "${OUTPUT_BASE_DIR:-}" ]; then
            EVAL_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"
        else
            EVAL_DIR="${path}/evalscope"
        fi
        log "  - $MODEL_NAME: $EVAL_DIR"
    done
fi
