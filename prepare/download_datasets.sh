export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/moonfs/luoyashuo-m2/hf_home
base_model_dir=/mnt/moonfs/luoyashuo-m2/hf_home/dataset
dataset_name=alibaba-pai/OmniThought
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --force-download \
    ${dataset_name} \
    --local-dir-use-symlinks False  \
    --local-dir ${base_model_dir}/${dataset_name}

