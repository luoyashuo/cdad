#!/bin/bash
export DISABLE_VERSION_CHECK=1
# 防止重复执行检测
PID_FILE="/tmp/train_fsf_judge.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "脚本已经在运行中，进程ID: $PID"
        exit 1
    else
        echo "PID 文件存在但进程不存在，删除旧的PID文件"
        rm -f "$PID_FILE"
    fi
fi
# 写入当前进程ID到PID文件
echo $$ > "$PID_FILE"
# 退出时删除PID文件
trap 'rm -f "$PID_FILE"' EXIT

# 配置
USER=$(whoami)
VENV_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate"
LLAMAFACTORY_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/LlamaFactory"

# 训练配置（根据GPU显存大小选择）
TRAIN_YAML_40G="examples/fsf_judge/qwen3_1_7b_judge.yaml"   # 40G显存
TRAIN_YAML_80G="examples/fsf_judge/qwen2_5_3b_judge.yaml"   # 80G显存
TRAIN_CKPT="/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/judge/qwen3_1_7b_fsf_judge"  # 与yaml中的output_dir一致
LOG_DIR="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/logs/judge"

# 函数：获取指定GPU的显存大小（MB）
get_gpu_memory() {
    local gpu_id="$1"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' '
}

# 函数：根据GPU显存选择配置文件
get_train_yaml() {
    local gpu_id="$1"
    local mem_mb=$(get_gpu_memory "$gpu_id")

    if [ -z "$mem_mb" ]; then
        echo "警告: 无法获取GPU $gpu_id 的显存大小，使用默认40G配置" >&2
        echo "$TRAIN_YAML_40G"
        return
    fi

    # 将MB转换为GB（大约）
    local mem_gb=$((mem_mb / 1024))

    if [ "$mem_gb" -ge 70 ]; then
        # 80G卡（通常显示为80GB左右）
        echo "$TRAIN_YAML_80G"
    else
        # 40G卡（通常显示为40GB左右）
        echo "$TRAIN_YAML_40G"
    fi
}

# 信号处理
stop_flag=0
trap 'stop_flag=1; echo "收到停止信号，等待当前训练完成后退出..."' INT TERM

# 函数：获取用户进程占用的GPU
get_user_gpus() {
    local gpus=()
    # 使用nvidia-smi pmon获取进程和GPU的对应关系
    while read -r gpu_id pid; do
        # 跳过空行和表头
        [[ -z "$gpu_id" ]] && continue
        [[ "$gpu_id" == "#" ]] && continue
        [[ "$gpu_id" == "Idx" ]] && continue
        [[ "$gpu_id" == "gpu" ]] && continue
        
        # 检查PID是否是数字
        if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
            continue
        fi
        
        # 检查进程所属用户
        process_user=$(ps -o user= -p "$pid" 2>/dev/null)
        
        # 如果是当前用户的进程，记录GPU ID
        if [[ "$process_user" == "$USER" ]]; then
            gpus+=("$gpu_id")
        fi
    done < <(nvidia-smi pmon -c 1 2>/dev/null | awk '{print $1, $2}')
    
    # 去重
    local unique_gpus=()
    declare -A seen
    for gpu in "${gpus[@]}"; do
        if [[ -z "${seen[$gpu]}" ]]; then
            unique_gpus+=("$gpu")
            seen[$gpu]=1
        fi
    done
    
    echo "${unique_gpus[@]}"
}

# 函数：获取空闲GPU（返回第一个空闲GPU）
get_free_gpus() {
    # 使用nvidia-smi获取所有GPU的使用情况，返回第一个空闲的GPU
    while IFS= read -r line; do
        # 跳过表头和空行
        if [[ "$line" == *"index"* ]] || [[ -z "$line" ]]; then
            continue
        fi
        
        local gpu_id=$(echo "$line" | awk -F',' '{print $1}' | tr -d ' ')
        local gpu_util=$(echo "$line" | awk -F',' '{print $2}' | tr -d ' ')
        local mem_util=$(echo "$line" | awk -F',' '{print $3}' | tr -d ' ')
        
        # 判断GPU是否空闲：GPU利用率 < 10% 且 显存利用率 < 10%
        if [[ "$gpu_util" -lt 10 && "$mem_util" -lt 10 ]]; then
            echo "$gpu_id"
            return  # 只返回第一个空闲GPU
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv,noheader)
}

# 函数：kill用户进程
kill_user_processes() {
    local gpus=($1)
    echo "正在kill用户 $USER 在GPU ${gpus[*]} 上的进程..."
    
    while read -r gpu_id pid; do
        # 跳过空行和表头
        [[ -z "$gpu_id" ]] && continue
        [[ "$gpu_id" == "#" ]] && continue
        [[ "$gpu_id" == "Idx" ]] && continue
        [[ "$gpu_id" == "gpu" ]] && continue
        
        # 检查PID是否是数字
        if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
            continue
        fi
        
        # 检查进程所属用户
        process_user=$(ps -o user= -p "$pid" 2>/dev/null)
        
        # 如果是当前用户的进程，且在指定的GPU上，则kill
        if [[ "$process_user" == "$USER" ]] && [[ " ${gpus[*]} " == *" $gpu_id "* ]]; then
            echo "Killing process $pid (GPU $gpu_id)..."
            kill -9 "$pid" 2>/dev/null
            # 检查kill是否成功
            if [ $? -eq 0 ]; then
                echo "Successfully killed process $pid"
            else
                echo "Failed to kill process $pid (already terminated or permission denied)"
            fi
        fi
    done < <(nvidia-smi pmon -c 1 2>/dev/null | awk '{print $1, $2}')
}

# 函数：删除指定GPU的旧日志文件
clean_old_logs() {
    local gpu_id="$1"
    local old_logs="${LOG_DIR}/train_fsf_judge_*_gpu_${gpu_id}.log"
    if ls $old_logs 1> /dev/null 2>&1; then
        echo "删除GPU $gpu_id 的旧日志文件..."
        rm -f $old_logs
    fi
}

# 函数：生成带时间戳的日志文件路径
generate_log_path() {
    local gpu_id="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/train_fsf_judge_${timestamp}_gpu_${gpu_id}.log"
    echo "$log_file"
}

# 函数：执行单卡训练
run_single_card_training() {
    local cuda_visible_devices="$1"

    # 删除该GPU的旧日志
    clean_old_logs "$cuda_visible_devices"

    local log_path=$(generate_log_path "$cuda_visible_devices")

    # 根据GPU显存选择配置文件
    local train_yaml=$(get_train_yaml "$cuda_visible_devices")

    echo "执行单卡训练，CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
    echo "使用配置文件: $train_yaml"
    echo "日志文件: $log_path"

    export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    export DISABLE_VERSION_CHECK=1
    export FORCE_TORCHRUN=1
    source "$VENV_PATH"
    cd "$LLAMAFACTORY_PATH"

    # 执行训练（使用 overwrite_output_dir 确保覆盖）
    nohup llamafactory-cli train "$train_yaml" 2>&1 > "$log_path" &
    TRAIN_PID=$!
    echo "训练进程PID: $TRAIN_PID"

    # 等待训练进程稳定运行（确保GPU已经被占用）
    echo "等待训练进程稳定运行..."
    sleep 180

    # 等待训练完成
    wait "$TRAIN_PID"
    echo "训练完成"
}

# 主函数
main() {
    # 解析参数
    local target_gpus=""
    local is_manual=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpus)
                target_gpus="$2"
                is_manual=true
                shift 2
                ;;
            -h|--help)
                echo "使用说明: $0 [--gpus <GPU_IDs>] [--help]"
                echo "  --gpus <GPU_IDs>: 手动指定GPU ID，多个用逗号分隔（如: 0,1,2）"
                echo "  --help: 显示帮助信息"
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
    
    # 获取GPU IDs
    if [ "$is_manual" = true ]; then
        echo "手动指定GPU: $target_gpus"
    else
        echo "自动检测用户 $USER 占用的GPU..."
        target_gpus=$(get_user_gpus)
        if [ -z "$target_gpus" ]; then
            echo "未检测到用户 $USER 占用的GPU，直接退出"
            exit 0
        fi
        echo "检测到的GPU: $target_gpus"
        
        kill_user_processes "$target_gpus"
    fi
    
    # 确保日志目录存在
    mkdir -p "$LOG_DIR"

    # 将GPU列表转换为数组（支持逗号分隔或空格分隔）
    IFS=', ' read -ra gpu_array <<< "$target_gpus"
    echo "将在以下GPU上分别启动单卡训练: ${gpu_array[*]}"

    while [ $stop_flag -eq 0 ]; do
        # 存储所有训练进程的PID
        declare -a train_pids=()

        # 为每个GPU启动单独的单卡训练
        for gpu_id in "${gpu_array[@]}"; do
            run_single_card_training "$gpu_id" &
            train_pids+=($!)
            echo "GPU $gpu_id 训练后台进程PID: ${train_pids[-1]}"
            # 稍等一下避免日志文件时间戳冲突
            sleep 1
        done

        # 等待所有训练进程完成
        echo "等待所有GPU训练完成..."
        for pid in "${train_pids[@]}"; do
            wait "$pid" 2>/dev/null
        done
        echo "本轮所有训练完成"

        # 检查是否需要停止
        if [ $stop_flag -eq 1 ]; then
            echo "退出循环训练"
            break
        fi

        echo "等待3秒后开始下一轮训练..."
        sleep 3
    done
    
    echo "程序已停止"
}

# 执行主函数
main "$@"
