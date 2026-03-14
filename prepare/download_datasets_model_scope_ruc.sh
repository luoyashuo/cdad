
export MODELSCOPE_CACHE=/fs/fast/u20240075/luoyashuo/modelscope_cache
export dataset_name=modelscope/gpqa
export local_dir=${MODELSCOPE_CACHE}/datasets/${dataset_name}

mkdir -p "${local_dir}"

nohup modelscope download \
  --dataset "${dataset_name}" \
  --local_dir "${local_dir}" \
  > "${MODELSCOPE_CACHE}/datasets/gpqa_modelscope.log" 2>&1 &
