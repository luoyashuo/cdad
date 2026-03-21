export HF_HUB_DISABLE_XET=1
#   export HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home

model_name=Yaseal/qwen3_4b_fsf_grm
local_dir=/data/hf_home/models/${model_name//\//-}

http_proxy=http://127.0.0.1:17890 https_proxy=http://127.0.0.1:17890 nohup huggingface-cli download \
    --resume-download \
    --local-dir "$local_dir" \
    "$model_name" > "${local_dir}.log" 2>&1 &




export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &


export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=Yaseal/CDAD_CKPT
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &


export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=asas-ai/Llama-3.2-1B-Open-R1-Distill
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &


# export HF_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=asas-ai/Llama-3.2-1B-Open-R1-Distill
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &



export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=Yaseal/llama3_3b_instruct_vallina_full_sft_30k
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &


export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/data/hf_home
model_name=Yaseal/llama3_1b_instruct_vallina_full_sft_30k
local_dir=/data/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &
