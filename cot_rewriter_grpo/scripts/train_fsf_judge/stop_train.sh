#!/bin/bash

# 停止占卡程序的脚本

# 配置
USER=$(whoami)
LOG_PATH="/home/luoyashuo/research/graph_kd/cdad/logs/train_fsf_judge.log"
SINGLE_CARD_CKPT="/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/judge/qwen3_1_7b_judge"
MULTI_CARD_CKPT="/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/judge/qwen3_8b_judge"

# 函数：停止占卡脚本进程
stop_occupy_script() {
    echo "停止占卡脚本进程..."
    script_pids=$(ps aux | grep "occupy_gpu.sh" | grep -v grep | awk '{print $2}')
    
    if [ -n "$script_pids" ]; then
        for pid in $script_pids; do
            echo "Killing script process $pid..."
            kill -9 "$pid" 2>/dev/null
        done
    else
        echo "未找到占卡脚本进程"
    fi
}

# 函数：停止训练进程
stop_training_processes() {
    echo "停止训练进程..."
    
    # 查找训练进程
    training_pids=$(ps aux | grep "llamafactory-cli" | grep -v grep | awk '{print $2}')
    
    if [ -n "$training_pids" ]; then
        for pid in $training_pids; do
            echo "Killing training process $pid..."
            kill -9 "$pid" 2>/dev/null
        done
    else
        echo "未找到训练进程"
    fi
}

# 函数：清理检查点和临时文件
cleanup_files() {
    echo "清理检查点和临时文件..."
    
    # 删除ckpt
    if [ -d "$SINGLE_CARD_CKPT" ]; then
        rm -rf "$SINGLE_CARD_CKPT"
        echo "删除单卡ckpt: $SINGLE_CARD_CKPT"
    fi
    
    if [ -d "$MULTI_CARD_CKPT" ]; then
        rm -rf "$MULTI_CARD_CKPT"
        echo "删除多卡ckpt: $MULTI_CARD_CKPT"
    fi
    
    # 清理临时文件（如需要）
    # rm -f /tmp/*.tmp
}

# 函数：检查GPU状态
check_gpu_status() {
    echo "检查GPU状态..."
    nvidia-smi
}

# 主函数
main() {
    echo "正在停止占卡程序..."
    
    # 执行停止操作
    stop_training_processes
    stop_occupy_script
    cleanup_files
    
    # 等待几秒钟让进程完全停止
    sleep 3
    
    # 显示GPU状态
    echo
    echo "当前GPU状态："
    check_gpu_status
    
    echo
    echo "占卡程序已停止，GPU资源已释放"
}

# 执行主函数
main
