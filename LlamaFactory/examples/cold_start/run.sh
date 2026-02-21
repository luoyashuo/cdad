setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DISABLE_VERSION_CHECK=1 SWANLAB_API_KEY=xmtHYzg2Bw5uh1dprg81E \
llamafactory-cli train \
    /data/luoys/projects/cdad/LlamaFactory/examples/cold_start/llama3_3b_instruct_top2k_cs.yaml \
    > /data/luoys/projects/cdad/LlamaFactory/logs/llama3_3b_instruct_top2k_cs.log 2>&1 &
