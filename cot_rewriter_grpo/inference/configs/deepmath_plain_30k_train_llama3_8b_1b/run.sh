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

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_dual_30k.yaml

python inference/fill_original_fsf_by_match.py \
    --target /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/llama3_8b_1b_grpo_8x40g/intermediate/fsf_tag_output.jsonl \
    --donor-glob /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/qwen2_5_7b_grpo_8x40g/intermediate/fsf_tag_output.jsonl

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_1b/deepmath_plain_30k_train_rep_fsf_r80_30k.yaml
