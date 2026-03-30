# mix_rewrite_ratio = 0.75 (75% rewrite, 25% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_mix_30k_r75.yaml

# mix_rewrite_ratio = 1.0 (100% rewrite, 0% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_mix_30k_r100.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_dual_30k.yaml


bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_rds_30k_r50.yaml


bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_rds_30k_r20.yaml
