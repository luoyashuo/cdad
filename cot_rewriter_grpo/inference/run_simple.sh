#!/bin/bash
# =============================================================================
# 简化版数据集改写运行脚本
# =============================================================================
#
# 功能：
#   使用 rewrite_model 生成 n_rewrites 个改写结果，自动打乱，生成训练集格式
#
# 使用方法：
#   bash inference/run_simple.sh --config inference/rewrite_simple_config.yaml
#   bash inference/run_simple.sh --config your_config.yaml --no-shuffle
# =============================================================================

# 禁用 HuggingFace 网络请求，启用离线模式
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 默认配置文件
CONFIG_FILE="${CONFIG_FILE:-inference/rewrite_simple_config.yaml}"

# 解析命令行参数
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --no-shuffle)
            EXTRA_ARGS="$EXTRA_ARGS --no-shuffle"
            shift
            ;;
        --drop-on-fail)
            # 改写失败时放弃该样本（默认使用原始CoT）
            EXTRA_ARGS="$EXTRA_ARGS --drop-on-fail"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--no-shuffle] [--drop-on-fail]"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python inference/rewrite_simple.py --config $CONFIG_FILE $EXTRA_ARGS"

# 运行
echo "Running: $CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $CMD

# 后台运行示例（取消注释以使用）:
# nohup bash inference/run_simple.sh > ./inference/logs/run_simple.log 2>&1 &
