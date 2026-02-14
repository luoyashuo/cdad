# 确保脱离终端控制
setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/fsf_judge/qwen3_4b_fsf_grm.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/train_fsf_grm.log 2>&1 &

sleep 1
PID=$(pgrep -f 'llamafactory-cli train' | head -1)
echo "Training started with PID: $PID"
echo "Check log: tail -f /data/luoys/projects/cdad/LlamaFactory/logs/train_fsf_grm.log"