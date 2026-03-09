export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/fs/fast/u20240075/hf_home
model_name=Yaseal/qwen3_4b_fsf_grm
local_dir=/fs/fast/u20240075/hf_home/models/${model_name//\//-}
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
export HF_HOME=/fs/fast/u20240075/hf_home
model_name=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
local_dir=/fs/fast/u20240075/hf_home/models/${model_name//\//-}
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
export HF_HOME=/fs/fast/u20240075/hf_home
model_name=Yaseal/CDAD_CKPT
local_dir=/fs/fast/u20240075/hf_home/models/${model_name//\//-}
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
export HF_HOME=/fs/fast/u20240075/hf_home
model_name=asas-ai/Llama-3.2-1B-Open-R1-Distill
local_dir=/fs/fast/u20240075/hf_home/models/${model_name//\//-}
log_file=${local_dir}.log
mkdir -p ${log_file%/*}
echo ${log_file}
nohup huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks False \
    --local-dir ${local_dir} \
    ${model_name} &> ${log_file} 2>&1 &
