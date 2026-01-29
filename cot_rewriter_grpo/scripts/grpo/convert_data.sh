#!/bin/bash

# ============================================================================
# CoT Rewriter GRPO Data Conversion Script
# ============================================================================
# 
# Convert JSON/JSONL data to Parquet format for training
#
# Usage:
#   bash scripts/grpo/convert_data.sh <input_path> [output_dir] [train_ratio]
#
# Examples:
#   bash scripts/grpo/convert_data.sh data/raw/train.jsonl
#   bash scripts/grpo/convert_data.sh data/raw/data.jsonl data/processed 0.9
#
# ============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_path> [output_dir] [train_ratio]"
    echo ""
    echo "Arguments:"
    echo "  input_path   Path to input JSON/JSONL file"
    echo "  output_dir   Output directory (default: data/processed)"
    echo "  train_ratio  Training set ratio (default: 0.9)"
    echo ""
    echo "Examples:"
    echo "  $0 data/raw/train.jsonl"
    echo "  $0 data/raw/data.json data/processed 0.8"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="${2:-data/processed}"
TRAIN_RATIO="${3:-0.9}"

# 检查输入文件
if [ ! -f "$INPUT_PATH" ]; then
    echo "Error: Input file not found: $INPUT_PATH"
    exit 1
fi

# 激活虚拟环境
VENV_PATH="/data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

# 进入项目目录
cd "$PROJECT_ROOT"

# 设置 Python 路径
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "========================================================================"
echo "CoT Rewriter GRPO Data Conversion"
echo "========================================================================"
echo "Input:       $INPUT_PATH"
echo "Output Dir:  $OUTPUT_DIR"
echo "Train Ratio: $TRAIN_RATIO"
echo "========================================================================"
echo ""

# 执行转换
python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
sys.path.insert(0, '${PROJECT_ROOT}/src')

from data.dataset import prepare_dataset

prepare_dataset(
    input_path='${INPUT_PATH}',
    output_dir='${OUTPUT_DIR}',
    train_ratio=${TRAIN_RATIO},
    seed=42,
)
"

echo ""
echo "========================================================================"
echo "Conversion completed!"
echo "========================================================================"
echo "Train file: ${OUTPUT_DIR}/train.parquet"
echo "Val file:   ${OUTPUT_DIR}/val.parquet"
echo ""
echo "Next steps:"
echo "  1. Update your config file:"
echo "     data:"
echo "       train_files: ${OUTPUT_DIR}/train.parquet"
echo "       val_files: ${OUTPUT_DIR}/val.parquet"
echo ""
echo "  2. Run training:"
echo "     bash scripts/grpo/run_train.sh <config_name> <gpus>"
echo "========================================================================"
