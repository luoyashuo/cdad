export CUDA_VISIBLE_DEVICES=3
if [ $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) -eq 1 ]; then
    export FORCE_TORCHRUN=1
fi
export DISABLE_VERSION_CHECK=1

# 禁用 DeepSpeed CUDA ops 编译（系统缺少 nvcc）
export DS_BUILD_OPS=0
export DS_BUILD_FP_QUANTIZER=0
export DS_BUILD_AIO=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_UTILS=0
export DS_BUILD_RANDOM_LTD=0
export DS_BUILD_INFERENCE=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_CPU_ADAGRAD=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_EMBED_ADAM_OP=0

# 设置 CUDA_HOME 为虚拟环境中的 CUDA（如果存在）
export CUDA_HOME=$(python -c 'import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)) + "/share/cuda")' 2>/dev/null || echo "/usr/local/cuda-12.8" 2>/dev/null || echo "/usr/local/cuda")

# 激活虚拟环境
source /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate

# 切换到 LlamaFactory 目录
cd /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/LlamaFactory

# 清理旧的日志
rm -f /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/logs/qwen3_1_7b_reward.log

# 运行训练
nohup llamafactory-cli train examples/fsf_judge/qwen3_1_7b_reward.yaml 2>&1 > /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/logs/qwen3_1_7b_reward.log &
wait
bash /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/scripts/train_fsf_judge/start_train.sh --gpus ${CUDA_VISIBLE_DEVICES}
