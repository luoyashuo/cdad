#!/bin/bash
# 使用 ModelScope 镜像下载 GSM8K 和 MATH 数据集

set -e

DATA_BASE_DIR="${1:-/data/luoys/data/cdad}"

echo "============================================================"
echo "使用 ModelScope 镜像下载数学数据集到: ${DATA_BASE_DIR}"
echo "============================================================"

mkdir -p "${DATA_BASE_DIR}/gsm8k"
mkdir -p "${DATA_BASE_DIR}/math"

cat > /tmp/download_datasets_ms.py << 'EOF'
import os
import sys
from pathlib import Path

data_base_dir = sys.argv[1]

# 设置 ModelScope 环境
os.environ['MODELSCOPE_CACHE'] = os.path.expanduser('~/.cache/modelscope')
os.environ['MODELSCOPE_PARALLEL_DOWNLOAD'] = '1'

try:
    from modelscope import MsDataset
    import pyarrow.parquet as pq
    import pandas as pd
    
    print("正在下载 GSM8K 数据集 (from ModelScope)...")
    gsm8k_train_path = Path(data_base_dir) / "gsm8k" / "train.parquet"
    gsm8k_test_path = Path(data_base_dir) / "gsm8k" / "test.parquet"
    
    if not gsm8k_train_path.exists() or not gsm8k_test_path.exists():
        # 从 ModelScope 下载
        # 数据集 ID: modelscope/gsm8k
        ds = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
        df = ds.to_pandas()
        df.to_parquet(str(gsm8k_train_path), index=False)
        print(f"✓ GSM8K 训练集下载完成: {len(df)} 条")
        
        ds_test = MsDataset.load('modelscope/gsm8k', subset_name='main', split='test')
        df_test = ds_test.to_pandas()
        df_test.to_parquet(str(gsm8k_test_path), index=False)
        print(f"✓ GSM8K 测试集下载完成: {len(df_test)} 条")
    else:
        print(f"✓ GSM8K 数据集已存在，跳过下载")
        
except Exception as e:
    print(f"✗ GSM8K 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n正在下载 MATH 数据集 (from ModelScope)...")
    math_train_path = Path(data_base_dir) / "math" / "train.parquet"
    math_test_path = Path(data_base_dir) / "math" / "test.parquet"
    
    if not math_train_path.exists() or not math_test_path.exists():
        # MATH 数据集在 ModelScope 上的 ID: modelscope/competition_math
        ds = MsDataset.load('modelscope/competition_math', split='train')
        df = ds.to_pandas()
        df.to_parquet(str(math_train_path), index=False)
        print(f"✓ MATH 训练集下载完成: {len(df)} 条")
        
        ds_test = MsDataset.load('modelscope/competition_math', split='test')
        df_test = ds_test.to_pandas()
        df_test.to_parquet(str(math_test_path), index=False)
        print(f"✓ MATH 测试集下载完成: {len(df_test)} 条")
    else:
        print(f"✓ MATH 数据集已存在，跳过下载")
        
except Exception as e:
    print(f"✗ MATH 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n============================================================")
print("所有数据集下载完成！")
print("============================================================")
EOF

# 运行下载脚本
python /tmp/download_datasets_ms.py "${DATA_BASE_DIR}"

# 清理临时文件
rm -f /tmp/download_datasets_ms.py

echo ""
echo "数据集位置:"
echo "  GSM8K: ${DATA_BASE_DIR}/gsm8k/"
echo "  MATH:  ${DATA_BASE_DIR}/math/"
ls -lh ${DATA_BASE_DIR}/gsm8k/*.parquet 2>/dev/null || true
ls -lh ${DATA_BASE_DIR}/math/*.parquet 2>/dev/null || true
