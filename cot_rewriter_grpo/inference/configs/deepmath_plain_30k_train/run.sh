python inference/fill_original_fsf_by_match.py \
    --target /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/qwen2_5_7b_grpo_8x40g/intermediate/fsf_tag_output.jsonl \
    --donor-glob "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/*/intermediate/fsf_tag_output.jsonl"

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_dual_15k.yaml
bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_dual_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_single_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train/deepmath_plain_30k_train_mix_30k.yaml
