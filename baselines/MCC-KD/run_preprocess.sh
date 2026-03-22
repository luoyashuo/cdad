#!/bin/bash
# MCC-KD 数据预处理脚本

set -e

echo "===== MCC-KD Data Preprocessing ====="
echo "Starting rationale filtering..."

python data_preprocess.py \
    --input_path /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl \
    --output_path /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl \
    --sample_size 15000 \
    --rationales_per_question 2 \
    --seed 42

echo "Preprocessing completed!"
echo "Output saved to: /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_mcckd.jsonl"
