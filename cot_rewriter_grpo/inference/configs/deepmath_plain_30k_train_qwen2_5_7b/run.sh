bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_dual_15k.yaml
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_dual_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_single_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_mix_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_qwen2_5_7b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_qwen2_5_7b/deepmath_plain_30k_train_rds_30k_r50.yaml


bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_qwen2_5_7b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_qwen2_5_7b/deepmath_plain_30k_train_mix_30k_r100.yaml
