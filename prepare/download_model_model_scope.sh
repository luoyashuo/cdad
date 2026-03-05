export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Meta-Llama-3.1-8B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &




export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Phi-4-mini-instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &



export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Meta-Llama-3.1-8B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &



export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Llama-3.2-3B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &



export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Llama-3.2-1B-Instruct
local_dir=${MODELSCOPE_CACHE}/models/${model_name}

mkdir -p ${local_dir} ${MODELSCOPE_CACHE}/logs/

nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log &

export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=LLM-Research/Llama-3.2-1B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &

export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
model_name=mistralai/Mistral-7B-Instruct-v0.3
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
mkdir -p ${local_dir}
nohup modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir} &> ${MODELSCOPE_CACHE}/logs/${model_name//\//-}.log 2>&1 &

