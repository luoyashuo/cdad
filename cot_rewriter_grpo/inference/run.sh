#!/bin/bash
# =============================================================================
# 数据集改写运行脚本
# =============================================================================
#
# 新工作流程（指标计算与过滤分离）：
#   1. 完整运行（所有步骤）: bash inference/run.sh
#   2. 只运行指标计算: bash inference/run.sh --step all
#   3. 重新应用过滤（调整阈值）: bash inference/run.sh --step filter --filter-config filter_adjusted.yaml
#   4. 单独运行某个指标计算: bash inference/run.sh --step similarity|cot_effect|fsf
#
# 过滤配置文件示例 (filter_adjusted.yaml):
#   enable_similarity_filter: true
#   similarity_penalty_threshold: 0.6
#   enable_cot_effect_filter: true
#   cot_effect_ratio_threshold: 0.9
#   enable_fsf_filter: true
# =============================================================================

# 禁用 HuggingFace 网络请求，启用离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 默认配置文件
CONFIG_FILE="${CONFIG_FILE:-inference/rewrite_config_example.yaml}"

# 解析命令行参数
STEP=""
FILTER_CONFIG=""
NO_PLOTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --filter-config)
            FILTER_CONFIG="$2"
            shift 2
            ;;
        --no-plots)
            NO_PLOTS="--no-plots"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python inference/rewrite_dataset.py --config $CONFIG_FILE"

if [ -n "$STEP" ]; then
    CMD="$CMD --step $STEP"
fi

if [ -n "$FILTER_CONFIG" ]; then
    CMD="$CMD --filter-config $FILTER_CONFIG"
fi

if [ -n "$NO_PLOTS" ]; then
    CMD="$CMD $NO_PLOTS"
fi

# 运行
echo "Running: $CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $CMD

# 后台运行示例（取消注释以使用）:
# nohup bash inference/run.sh > ./inference/logs/run_inference.log 2>&1 &
