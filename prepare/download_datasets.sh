export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs/fast/u20240075/hf_home
base_model_dir=/fs/fast/u20240075/hf_home/dataset
dataset_name=alibaba-pai/OmniThought
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --force-download \
    ${dataset_name} \
    --local-dir-use-symlinks False  \
    --local-dir ${base_model_dir}/${dataset_name}


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs/fast/u20240075/hf_home
base_model_dir=/fs/fast/u20240075/hf_home/datasets
dataset_name=Yaseal/CDAD_EXP
mkdir -p "$(dirname "${base_model_dir}/${dataset_name}")"
nohup huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --force-download \
  "${dataset_name}" \
  --local-dir-use-symlinks False \
  --local-dir "${base_model_dir}/${dataset_name}" \
  > "${base_model_dir}/$(basename "${dataset_name}").log" 2>&1 &






export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs/fast/u20240075/hf_home
base_model_dir=/fs/fast/u20240075/hf_home/datasets
dataset_name=meta-math/MetaMathQA-40K
mkdir -p "$(dirname "${base_model_dir}/${dataset_name}")"
nohup huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --force-download \
  "${dataset_name}" \
  --local-dir-use-symlinks False \
  --local-dir "${base_model_dir}/${dataset_name}" \
  > "${base_model_dir}/$(basename "${dataset_name}").log" 2>&1 &


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs/fast/u20240075/hf_home
base_model_dir=/fs/fast/u20240075/hf_home/datasets
dataset_name=hails/agieval-math
mkdir -p "$(dirname "${base_model_dir}/${dataset_name}")"
nohup huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --force-download \
  "${dataset_name}" \
  --local-dir-use-symlinks False \
  --local-dir "${base_model_dir}/${dataset_name}" \
  > "${base_model_dir}/$(basename "${dataset_name}").log" 2>&1 &


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/fs/fast/u20240075/hf_home
base_model_dir=/fs/fast/u20240075/hf_home/datasets
dataset_name=OpenEvals/IMO-AnswerBench
mkdir -p "$(dirname "${base_model_dir}/${dataset_name}")"
nohup huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --force-download \
  "${dataset_name}" \
  --local-dir-use-symlinks False \
  --local-dir "${base_model_dir}/${dataset_name}" \
  > "${base_model_dir}/$(basename "${dataset_name}").log" 2>&1 &




