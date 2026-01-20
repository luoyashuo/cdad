export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/moonfs/luoyashuo-ksyun/hf_home
base_model_dir=/mnt/moonfs/luoyashuo-ksyun/hf_home/dataset
dataset_name=alibaba-pai/OmniThought
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --force-download \
    ${dataset_name} \
    --local-dir-use-symlinks False  \
    --local-dir ${base_model_dir}/${dataset_name}



export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/moonfs/luoyashuo-ksyun/hf_home
base_model_dir=/mnt/moonfs/luoyashuo-ksyun/hf_home/dataset
dataset_name=LiteSSLHub/NanoNet
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --force-download \
    ${dataset_name} \
    --local-dir-use-symlinks False  \
    --local-dir ${base_model_dir}/${dataset_name}


rsync -avz --progress --partial --bwlimit=5120 \
  -e "ssh -o ServerAliveInterval=60 -o ForwardAgent=yes" \
  dev.luoyashuo.msh-dev.dm@m2-launchpad.msh.work:/mnt/moonfs/luoyashuo-ksyun/hf_home/dataset/LiteSSLHub/NanoNet \
  /Users/moonshot/Downloads/