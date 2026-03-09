
export MODELSCOPE_CACHE=/data/modelscope
dataset_name=modelscope/gpqa
local_dir=${MODELSCOPE_CACHE}/datasets/${model_name}

mkdir -p "${local_dir}"

nohup modelscope download \
  --dataset "${dataset_name}" \
  --local_dir "${local_dir}" \
  > "${MODELSCOPE_CACHE}/datasets/gpqa_modelscope.log" 2>&1 &
