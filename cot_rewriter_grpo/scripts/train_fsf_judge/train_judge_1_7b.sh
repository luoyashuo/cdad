export CUDA_VISIBLE_DEVICES=6
source /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate
cd ../LlamaFactory
nohup llamafactory-cli train examples/fsf_judge/qwen3_1_7b_judge.yaml 2>&1 > /home/luoyashuo/research/graph_kd/cdad/logs/train_fsf_judge.log &
wait
rm -r /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/judge/qwen3_1_7b_judge
