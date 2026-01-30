#!/bin/bash

# 停止指定GPU或所有GPU上当前用户的训练进程
#
# 用法:
#   bash stop_all_train.sh              # 停止所有GPU上的进程
#   bash stop_all_train.sh --gpus 5     # 只停止GPU 5上的进程
#   bash stop_all_train.sh --gpus 1,2,3 # 停止GPU 1,2,3上的进程

USER=$(whoami)

# 解析参数
TARGET_GPUS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            TARGET_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [--gpus <GPU_IDs>]"
            echo "  --gpus <GPU_IDs>: 指定要停止的GPU ID，多个用逗号分隔（如: 1,2,3）"
            echo "                    不指定则停止所有GPU上的进程"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 将逗号分隔的GPU列表转换为数组
if [[ -n "$TARGET_GPUS" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$TARGET_GPUS"
    echo "目标GPU: ${GPU_ARRAY[*]}"
else
    GPU_ARRAY=()
    echo "目标: 所有GPU"
fi

# 函数：检查GPU是否在目标列表中
is_target_gpu() {
    local gpu_id="$1"
    # 如果没有指定目标GPU，则所有GPU都是目标
    if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
        return 0
    fi
    # 检查GPU是否在目标列表中
    for target in "${GPU_ARRAY[@]}"; do
        if [[ "$gpu_id" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

# 函数：停止指定GPU上当前用户的进程
stop_user_processes() {
    echo "正在停止用户 $USER 的进程..."

    local killed_count=0

    # 使用nvidia-smi pmon获取结构化的进程信息
    # 格式: gpu_id pid
    while read -r gpu_id pid; do
        # 跳过空行和表头
        [[ -z "$gpu_id" ]] && continue
        [[ "$gpu_id" == "#" ]] && continue
        [[ "$gpu_id" == "gpu" ]] && continue
        [[ "$gpu_id" == "Idx" ]] && continue

        # 检查PID是否是数字
        if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
            continue
        fi

        # 检查是否是目标GPU
        if ! is_target_gpu "$gpu_id"; then
            continue
        fi

        # 检查进程所属用户
        process_user=$(ps -o user= -p "$pid" 2>/dev/null)

        # 如果是当前用户的进程，则kill
        if [[ "$process_user" == "$USER" ]]; then
            echo "Killing process $pid (GPU $gpu_id)..."
            kill -9 "$pid" 2>/dev/null
            # 检查kill是否成功
            if [ $? -eq 0 ]; then
                echo "Successfully killed process $pid"
                ((killed_count++))
            else
                echo "Failed to kill process $pid (already terminated or permission denied)"
            fi
        fi
    done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NR>2 && $2 != "-" {print $1, $2}')

    echo "共停止 $killed_count 个进程"
}

# 函数：检查GPU状态
check_gpu_status() {
    echo "检查GPU状态..."
    nvidia-smi
}

# 主函数
main() {
    if [[ -n "$TARGET_GPUS" ]]; then
        echo "正在停止GPU ${TARGET_GPUS} 上用户 $USER 的训练进程..."
    else
        echo "正在停止所有GPU上用户 $USER 的训练进程..."
    fi

    # 停止进程
    stop_user_processes

    # 等待几秒钟让进程完全停止
    sleep 3

    # 显示GPU状态
    echo
    echo "当前GPU状态："
    check_gpu_status

    echo
    if [[ -n "$TARGET_GPUS" ]]; then
        echo "GPU ${TARGET_GPUS} 上用户 $USER 的进程已停止"
    else
        echo "所有GPU上用户 $USER 的进程已停止"
    fi
}

# 执行主函数
main
