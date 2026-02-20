python eval.py eval_configs/eval_cg.yaml
python eval.py eval_configs/eval_rv.yaml
python eval.py eval_configs/eval_fsf.yaml
python eval.py eval_configs/eval_random.yaml


# nohup bash /home/luoyashuo/research/graph_kd/cdad/eval/launch_scripts/run.sh > /home/luoyashuo/research/graph_kd/cdad/eval/vllm_logs/run.log 2>&1 &
# nohup python eval.py eval_configs/eval_rv_optimal_top2000.yaml > /data/luoys/projects/cdad/eval/vllm_logs/eval_rv_optimal_top2000.log


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py \
    /data/luoys/projects/cdad/eval/eval_configs/eval_rv_optimal_top2000_rewrite.yaml \
    > /data/luoys/projects/cdad/eval/vllm_logs/llama3_2_1b_rv_optimal_top50k_rewrite.log 2>&1 &

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py \
    /data/luoys/projects/cdad/eval/eval_configs/eval_rv_optimal_top2000_rewrite_mix.yaml \
    > /data/luoys/projects/cdad/eval/vllm_logs/llama3_2_1b_rv_optimal_top50k_rewrite_mix_2k_2.log 2>&1 &

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py \
    /data/luoys/projects/cdad/eval/eval_configs/eval_rv_optimal_top2000_rewrite_continue.yaml \
    > /data/luoys/projects/cdad/eval/vllm_logs/qwen3_1_7b_rv_optimal_top2000_rewrite_continue.log 2>&1 &
    
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py \
    /data/luoys/projects/cdad/eval/eval_configs/eval_rv_optimal_top2000.yaml \
    > /data/luoys/projects/cdad/eval/vllm_logs/qwen3_1_7b_rv_optimal_top2000.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python eval.py \
    /data/luoys/projects/cdad/eval/eval_configs/eval_student.yaml \
    > /data/luoys/projects/cdad/eval/vllm_logs/eval_student.log 2>&1 &

