export CUDA_VISIBLE_DEVICES=0,5
if [ $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) -eq 1 ]; then
    export FORCE_TORCHRUN=1
fi
export DISABLE_VERSION_CHECK=1
# 激活虚拟环境
source /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate

# 切换到 LlamaFactory 目录
cd /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/LlamaFactory
bash /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/scripts/train_fsf_judge/stop_all_train.sh
# 清理旧的日志
rm -f /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/logs/qwen3_1_7b_reward.log

# 运行训练
nohup llamafactory-cli train examples/fsf_judge/qwen3_1_7b_reward.yaml 2>&1 > /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/logs/qwen3_1_7b_reward.log &
wait
bash /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/scripts/train_fsf_judge/start_train.sh --gpus ${CUDA_VISIBLE_DEVICES}
