export CUDA_VISIBLE_DEVICES=1,2,3,5
nohup llamafactory-cli examples/fsf_judge/qwen3_8b_fsf_judge.yaml 2>&1 > /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/logs/fsf_judge/qwen3_8b_fsf_judge.log &
wait
bash /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/scripts/train_fsf_judge/start_train.sh
