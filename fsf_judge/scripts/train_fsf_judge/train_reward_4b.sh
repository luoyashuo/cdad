export CUDA_VISIBLE_DEVICES=2
if [ $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) -eq 1 ]; then
    export FORCE_TORCHRUN=1
fi
export DISABLE_VERSION_CHECK=1
nohup llamafactory-cli train examples/fsf_judge/qwen3_4b_reward.yaml 2>&1 > /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/fsf_judge/logs/qwen3_4b_reward.log &
wait
bash /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/scripts/train_fsf_judge/start_train.sh --gpus ${CUDA_VISIBLE_DEVICES}
