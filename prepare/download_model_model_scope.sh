export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=LLM-Research/Meta-Llama-3.1-8B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &

export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=LLM-Research/Llama-3.2-3B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &



export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=LLM-Research/Llama-3.2-1B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}

mkdir -p ${local_dir} ${MODELSCOPE_CACHE}/logs/

nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log &


export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen2.5-3B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &


export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen2.5-1.5B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &


export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen2.5-7B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &


export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen2.5-Math-1.5B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &

export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen3-4B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &


export MODELSCOPE_CACHE=/data/modelscope_cache
model_name=Qwen/Qwen3-8B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &
