
export CUDA_VISIBLE_DEVICES=2,3
export no_proxy=localhost,127.0.0.1
nohup python eval.py \
    eval_configs/eval_all.yaml \
    3 \
    > vllm_logs/llama3_1b_meta_math_40k.log 2>&1 &

