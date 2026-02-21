#!/bin/bash
# 使用 ModelScope 镜像下载 GSM8K 和 MATH 数据集，并预处理为 verl 格式

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

from modelscope import MsDataset
import pandas as pd
import re

def extract_gsm8k_solution(answer_str):
    """从 GSM8K 答案中提取最终数值答案"""
    match = re.search(r"####\s*([\-]?[0-9\.\,]+)", answer_str)
    if match:
        return match.group(1).replace(",", "")
    return answer_str.strip()

def extract_math_solution(solution_str):
    """从 MATH 答案中提取 \boxed{} 内容"""
    match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if match:
        return match.group(1)
    lines = solution_str.strip().split('\n')
    last_line = lines[-1]
    match = re.search(r"=\s*([\-]?[0-9\.\,]+)\s*$", last_line)
    if match:
        return match.group(1)
    return solution_str.strip()[:100]

print("=" * 60)
print("1. 下载并处理 GSM8K 数据集...")
print("=" * 60)

try:
    data_source = "openai/gsm8k"
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    
    for split in ['train', 'test']:
        output_file = Path(data_base_dir) / 'gsm8k' / f'{split}.parquet'
        
        if output_file.exists():
            print(f"   {split}.parquet 已存在，跳过下载")
            continue
            
        print(f"   下载 {split} 集...")
        ds = MsDataset.load('modelscope/gsm8k', subset_name='main', split=split, trust_remote_code=True)
        df = ds.to_pandas()
        print(f"   原始数据: {len(df)} 条")
        
        processed_data = []
        for idx, row in df.iterrows():
            question = row['question'] + " " + instruction
            answer_raw = row['answer']
            solution = extract_gsm8k_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": row['question'],
                },
            }
            processed_data.append(data)
        
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_file, index=False)
        print(f"   ✓ {split} 集处理完成: {len(processed_df)} 条 -> {output_file}")
        
except Exception as e:
    print(f"   ✗ GSM8K 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("2. 下载并处理 MATH 数据集...")
print("=" * 60)

try:
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    
    for split in ['train', 'test']:
        output_file = Path(data_base_dir) / 'math' / f'{split}.parquet'
        
        if output_file.exists():
            print(f"   {split}.parquet 已存在，跳过下载")
            continue
            
        print(f"   下载 {split} 集...")
        ds = MsDataset.load('modelscope/competition_math', split=split, trust_remote_code=True)
        df = ds.to_pandas()
        print(f"   原始数据: {len(df)} 条")
        
        processed_data = []
        for idx, row in df.iterrows():
            question = row['problem'] + " " + instruction
            answer_raw = row['solution']
            solution = extract_math_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "level": row.get('level', ''),
                    "type": row.get('type', ''),
                    "solution": answer_raw,
                    "problem": row['problem'],
                },
            }
            processed_data.append(data)
        
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_file, index=False)
        print(f"   ✓ {split} 集处理完成: {len(processed_df)} 条 -> {output_file}")
        
except Exception as e:
    print(f"   ✗ MATH 下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("所有数据集下载并处理完成！")
print("=" * 60)
EOF

# 运行下载脚本
python /tmp/download_datasets_ms.py "${DATA_BASE_DIR}"

# 清理临时文件
rm -f /tmp/download_datasets_ms.py

echo ""
echo "数据集位置:"
echo "  GSM8K: ${DATA_BASE_DIR}/gsm8k/"
echo "  MATH:  ${DATA_BASE_DIR}/math/"
echo ""
ls -lh ${DATA_BASE_DIR}/gsm8k/*.parquet 2>/dev/null || true
ls -lh ${DATA_BASE_DIR}/math/*.parquet 2>/dev/null || true
