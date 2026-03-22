# mix_rewrite_ratio = 0.25 (25% rewrite, 75% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_mix_30k_r25.yaml

# mix_rewrite_ratio = 0.50 (50% rewrite, 50% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_mix_30k_r50.yaml

# mix_rewrite_ratio = 0.75 (75% rewrite, 25% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_mix_30k_r75.yaml

# mix_rewrite_ratio = 1.0 (100% rewrite, 0% original)
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_mix_30k_r100.yaml