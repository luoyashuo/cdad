# setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
# llamafactory-cli train \
#     /data/luoys/projects/cdad/LlamaFactory/examples/warmup/qwen3_1_7b_rv_optimal_top2000.yaml \
#     > /data/luoys/projects/cdad/LlamaFactory/logs/qwen3_1_7b_rv_optimal_top2000.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top2000.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top2000.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top2000_rewrite_mix.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top2000_rewrite_mix.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top2000_rewrite.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top2000_rewrite.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_base_rv_optimal_top2000_rewrite.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_base_rv_optimal_top2000_rewrite.log 2>&1 &




setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_base_rv_optimal_top2000.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_base_rv_optimal_top2000.log 2>&1 &



setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top20k.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top20k.log 2>&1 &

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top50k.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top50k.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top50k_rewrite.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top50k_rewrite.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top50k_rewrite_mix.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top50k_rewrite_mix.log 2>&1 &


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top50k_rewrite_mix_2k.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top50k_rewrite_mix_2k.log 2>&1 & 


setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/warmup/llama3_1b_rv_optimal_top50k_rewrite_mix_2k_2.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_1b_rv_optimal_top50k_rewrite_mix_2k_2.log 2>&1 & 
