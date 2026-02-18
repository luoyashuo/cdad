# 禁用 HuggingFace 网络请求，启用离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python inference/rewrite_dataset.py \
    --config inference/rewrite_config_example.yaml
# nohup bash inference/run.sh > ./inference/logs/run_inference.log 2>&1 &
