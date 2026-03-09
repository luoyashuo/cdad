export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/dataset
dataset_name=alibaba-pai/OmniThought
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    --force-download \
    ${dataset_name} \
    --local-dir-use-symlinks False  \
    --local-dir ${base_model_dir}/${dataset_name}


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
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

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=bigcode/bigcodebench
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=KodCode/KodCode-V1-SFT-R1
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=livecodebench/code_generation_lite
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=alibaba-pai/OmniThought
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=zwhe99/DeepMath-103K
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=Idavidrein/gpqa
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
export HF_HOME=/data/hf_home
base_model_dir=/data/hf_home/datasets
dataset_name=TIGER-Lab/MMLU-Pro
mkdir -p "$(dirname "${base_model_dir}/${dataset_name}")"
nohup huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  --force-download \
  "${dataset_name}" \
  --local-dir-use-symlinks False \
  --local-dir "${base_model_dir}/${dataset_name}" \
  > "${base_model_dir}/$(basename "${dataset_name}").log" 2>&1 &
