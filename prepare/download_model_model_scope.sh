export MODELSCOPE_CACHE=/data/luoys/modelscope_cache
model_name=Qwen/Qwen3-1.7B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir}


export MODELSCOPE_CACHE=/data/luoys/modelscope_cache
model_name=Qwen/Qwen3-4B
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir}

export MODELSCOPE_CACHE=/data/luoys/modelscope_cache
model_name=Qwen/Qwen3-1.7B-Base
local_dir=${MODELSCOPE_CACHE}/models/${model_name}
modelscope download \
    --model ${model_name} \
    --local_dir ${local_dir}