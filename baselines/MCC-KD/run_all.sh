#!/bin/bash
# MCC-KD 完整流程脚本：数据预处理 + 训练

set -e

echo "========================================"
echo "  MCC-KD: Multi-CoT Consistent KD"
echo "========================================"

# Step 1: 数据预处理
echo ""
echo "Step 1: Data Preprocessing (Rationale Filtering)"
echo "------------------------------------------------"
bash run_preprocess.sh

# Step 2: 模型训练
echo ""
echo "Step 2: Model Training (Multi-CoT Consistent Distillation)"
echo "-----------------------------------------------------------"
bash run_train.sh

echo ""
echo "========================================"
echo "  All tasks completed successfully!"
echo "========================================"
