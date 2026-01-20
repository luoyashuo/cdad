#!/bin/bash
# ============================================================================
# 数据集构建脚本 - 4种策略
# ============================================================================

set -e

cd /mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/reasoning_graph

echo "=============================================="
echo "  构建 OmniThought 数据集 (4种策略)"
echo "=============================================="

# 公共参数
INPUT_FILES="/mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
    /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl"
TRAIN_RATIO="0.99"
DATASET_INFO="../LlamaFactory/data/dataset_info.json"

# 1. CD Optimal 策略（7B 模型配置：CD 范围 0-6）
echo ""
echo "[1/4] 构建 CD Optimal 数据集..."
python build_sharegpt_dataset.py \
    -i $INPUT_FILES \
    -s cd_optimal \
    --cd_min 0 --cd_max 6 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_cd_optimal \
    --train-ratio $TRAIN_RATIO \
    -r $DATASET_INFO

# 2. RV Optimal 策略（7B 模型配置：RV 范围 3-5）
echo ""
echo "[2/4] 构建 RV Optimal 数据集..."
python build_sharegpt_dataset.py \
    -i $INPUT_FILES \
    -s rv_optimal \
    --rv_min 3 --rv_max 5 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_rv_optimal \
    --train-ratio $TRAIN_RATIO \
    -r $DATASET_INFO

# 3. Combined 双规则概率采样（7B 模型配置）
echo ""
echo "[3/4] 构建 Combined 数据集..."
python build_sharegpt_dataset.py \
    -i $INPUT_FILES \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --mu_cd 6 \
    --beta 0.5 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_combined \
    --train-ratio $TRAIN_RATIO \
    -r $DATASET_INFO

# 4. Combined + FSF 最低（在候选范围内选择 FSF 最低的）
echo ""
echo "[4/4] 构建 Combined + FSF 最低数据集..."
python build_sharegpt_dataset.py \
    -i $INPUT_FILES \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --fsf_lowest \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_combined_fsf \
    --train-ratio $TRAIN_RATIO \
    -r $DATASET_INFO

echo ""
echo "=============================================="
echo "  所有数据集构建完成！"
echo "=============================================="
echo "生成的数据集:"
echo "  1. omnithought_cd_optimal_train / omnithought_cd_optimal_val"
echo "  2. omnithought_rv_optimal_train / omnithought_rv_optimal_val"
echo "  3. omnithought_combined_train / omnithought_combined_val"
echo "  4. omnithought_combined_fsf_train / omnithought_combined_fsf_val"
