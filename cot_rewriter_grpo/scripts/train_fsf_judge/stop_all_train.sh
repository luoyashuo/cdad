#!/bin/bash

# 停止所有GPU上当前用户的训练进程

USER=$(whoami)

# 函数：停止所有GPU上当前用户的进程
stop_all_user_processes() {
    echo "正在停止所有GPU上用户 $USER 的进程..."
    
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
        
        # 检查进程所属用户
        process_user=$(ps -o user= -p "$pid" 2>/dev/null)
        
        # 如果是当前用户的进程，则kill
        if [[ "$process_user" == "$USER" ]]; then
            echo "Killing process $pid (GPU $gpu_id)..."
            kill -9 "$pid" 2>/dev/null
            # 检查kill是否成功
            if [ $? -eq 0 ]; then
                echo "Successfully killed process $pid"
            else
                echo "Failed to kill process $pid (already terminated or permission denied)"
            fi
        fi
    done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NR>2 && $2 != "-" {print $1, $2}')
}

# 函数：检查GPU状态
check_gpu_status() {
    echo "检查GPU状态..."
    nvidia-smi
}

# 主函数
main() {
    echo "正在停止所有GPU上用户 $USER 的训练进程..."
    
    # 停止进程
    stop_all_user_processes
    
    # 等待几秒钟让进程完全停止
    sleep 3
    
    # 显示GPU状态
    echo
    echo "当前GPU状态："
    check_gpu_status
    
    echo
    echo "所有GPU上用户 $USER 的进程已停止"
}

# 执行主函数
main
