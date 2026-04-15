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

bash inference/build_dataset.sh \
    --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_mix_30k_r50_v3.yaml \
    --input-file /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/llama3_8b_3b_grpo_8x40g_v3/intermediate/correct_tag_output.jsonl


bash inference/build_dataset.sh \
    --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_mix_30k_r100_v3.yaml \
    --input-file /data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite/llama3_8b_3b_grpo_8x40g_v3/intermediate/correct_tag_output.jsonl


bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_rep_fsf_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_rep_no_fsf_30k.yaml

bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_mix_30k_r100_long.yaml


bash inference/build_dataset.sh --config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/rollout.yaml  \
    --dataset-config inference/configs/deepmath_plain_30k_train_llama3_8b_3b/deepmath_plain_30k_train_rep_fsf_r80_30k.yaml
