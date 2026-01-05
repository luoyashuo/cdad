#!/bin/bash

# 0,1卡训练 qwen3_1_7b_fsf.yaml
CUDA_VISIBLE_DEVICES=0,1 nohup llamafactory-cli train /home/luoyashuo/research/graph_kd/cdad/LlamaFactory/examples/baseline/qwen3_1_7b_fsf.yaml > /home/luoyashuo/research/graph_kd/cdad/logs/train_fsf.log 2>&1 &

# 2,3卡训练 qwen3_1_7b_rv.yaml
CUDA_VISIBLE_DEVICES=2,3 nohup llamafactory-cli train /home/luoyashuo/research/graph_kd/cdad/LlamaFactory/examples/baseline/qwen3_1_7b_rv.yaml > /home/luoyashuo/research/graph_kd/cdad/logs/train_rv.log 2>&1 &

# 显示后台任务
echo "训练任务已启动："
echo "- fsf 任务使用 GPU 0,1，日志: /home/luoyashuo/research/graph_kd/cdad/logs/train_fsf.log"
echo "- rv 任务使用 GPU 2,3，日志: /home/luoyashuo/research/graph_kd/cdad/logs/train_rv.log"
jobs
