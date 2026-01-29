export CUDA_VISIBLE_DEVICES=1,2,3,5
export DISABLE_VERSION_CHECK=1

# SwanLab 配置
export SWANLAB_MODE=cloud
# SwanLab 日志目录与训练输出目录保持一致
export SWANLAB_LOG_DIR=/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/swanlab

# 清理残留的 Ray 进程（避免权限问题和端口冲突）
echo "Cleaning up Ray processes..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
sleep 2

bash scripts/train_fsf_judge/stop_all_train.sh
wait
nohup bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu $CUDA_VISIBLE_DEVICES > logs/grpo/qwen3_8b_grpo_4gpu.log 2>&1 &
wait
bash scripts/train_fsf_judge/start_train.sh --gpus $CUDA_VISIBLE_DEVICES
