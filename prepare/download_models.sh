export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data-share/yeesuanAI08/lihaitao/luoys/hf_home
model_name=Qwen/Qwen3-8B
local_dir=/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/${model_name}

huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name}

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data-share/yeesuanAI08/lihaitao/luoys/hf_home
model_name=Qwen/Qwen3-4B
local_dir=/data-share/yeesuanAI08/lihaitao/luoys/hf_home/models/${model_name}

huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name}