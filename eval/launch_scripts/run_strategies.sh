#!/bin/bash
# ============================================================================
# 顺序评测脚本 - 4种策略
# ============================================================================

set -e

cd /mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/eval

echo "=============================================="
echo "  顺序评测 4 种数据选择策略"
echo "=============================================="

# 1. CD Optimal 评测
echo ""
echo "[1/4] 评测 CD Optimal 模型..."
echo "开始时间: $(date)"
python eval.py eval_configs/eval_cd_optimal.yaml
echo "完成时间: $(date)"

# 2. RV Optimal 评测
echo ""
echo "[2/4] 评测 RV Optimal 模型..."
echo "开始时间: $(date)"
python eval.py eval_configs/eval_rv_optimal.yaml
echo "完成时间: $(date)"

# 3. Combined 评测
echo ""
echo "[3/4] 评测 Combined 模型..."
echo "开始时间: $(date)"
python eval.py eval_configs/eval_combined.yaml
echo "完成时间: $(date)"

# 4. Combined + FSF 评测
echo ""
echo "[4/4] 评测 Combined + FSF 模型..."
echo "开始时间: $(date)"
python eval.py eval_configs/eval_combined_fsf.yaml
echo "完成时间: $(date)"

echo ""
echo "=============================================="
echo "  所有评测完成！"
echo "=============================================="
# nohup bash /mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/eval/launch_scripts/run_strategies.sh > /mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/eval/vllm_logs/run_strategies.log 2>&1 &