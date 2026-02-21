#!/usr/bin/env python3
"""
预处理 GSM8K 和 MATH 数据集为 verl 格式
"""

import argparse
import os
import re
import pandas as pd
from pathlib import Path


def extract_gsm8k_solution(answer_str):
    """从 GSM8K 答案中提取最终数值答案"""
    match = re.search(r"####\s*([\-]?[0-9\.\,]+)", answer_str)
    if match:
        return match.group(1).replace(",", "")
    return answer_str.strip()


def extract_math_solution(solution_str):
    """从 MATH 答案中提取 \boxed{} 内容"""
    # 尝试匹配 \boxed{content}
    match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
    if match:
        return match.group(1)
    # 如果没有 \boxed{}, 尝试匹配最后的数字或表达式
    lines = solution_str.strip().split('\n')
    last_line = lines[-1]
    # 尝试匹配行末的数值
    match = re.search(r"=\s*([\-]?[0-9\.\,]+)\s*$", last_line)
    if match:
        return match.group(1)
    return solution_str.strip()[:100]  # 返回前100字符作为备选


def preprocess_gsm8k(data_dir):
    """预处理 GSM8K 数据集"""
    print("预处理 GSM8K 数据集...")
    
    data_source = "openai/gsm8k"
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    
    for split in ['train', 'test']:
        input_file = Path(data_dir) / 'gsm8k' / f'{split}.parquet'
        output_file = Path(data_dir) / 'gsm8k' / f'{split}_processed.parquet'
        
        if not input_file.exists():
            print(f"  ✗ 文件不存在: {input_file}")
            continue
            
        df = pd.read_parquet(input_file)
        print(f"  原始 {split} 集: {len(df)} 条")
        
        processed_data = []
        for idx, row in df.iterrows():
            question = row['question'] + " " + instruction
            answer_raw = row['answer']
            solution = extract_gsm8k_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
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
        
        # 保存为 parquet
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_file, index=False)
        print(f"  ✓ 处理后 {split} 集: {len(processed_df)} 条 -> {output_file}")
        
        # 覆盖原始文件
        processed_df.to_parquet(input_file, index=False)
        print(f"  ✓ 已覆盖原始文件: {input_file}")


def preprocess_math(data_dir):
    """预处理 MATH 数据集"""
    print("\n预处理 MATH 数据集...")
    
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    
    for split in ['train', 'test']:
        input_file = Path(data_dir) / 'math' / f'{split}.parquet'
        output_file = Path(data_dir) / 'math' / f'{split}_processed.parquet'
        
        if not input_file.exists():
            print(f"  ✗ 文件不存在: {input_file}")
            continue
            
        df = pd.read_parquet(input_file)
        print(f"  原始 {split} 集: {len(df)} 条")
        
        processed_data = []
        for idx, row in df.iterrows():
            question = row['problem'] + " " + instruction
            answer_raw = row['solution']
            solution = extract_math_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
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
        
        # 保存为 parquet
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_parquet(output_file, index=False)
        print(f"  ✓ 处理后 {split} 集: {len(processed_df)} 条 -> {output_file}")
        
        # 覆盖原始文件
        processed_df.to_parquet(input_file, index=False)
        print(f"  ✓ 已覆盖原始文件: {input_file}")


def main():
    parser = argparse.ArgumentParser(description="预处理 GSM8K 和 MATH 数据集为 verl 格式")
    parser.add_argument(
        "--data_dir",
        default="/data/luoys/data/cdad",
        help="数据根目录，包含 gsm8k 和 math 子目录"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    preprocess_gsm8k(args.data_dir)
    preprocess_math(args.data_dir)
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)
    print("\n处理后的数据位置:")
    print(f"  GSM8K: {args.data_dir}/gsm8k/train.parquet, test.parquet")
    print(f"  MATH:  {args.data_dir}/math/train.parquet, test.parquet")
    print("\n备份文件:")
    print(f"  GSM8K: {args.data_dir}/gsm8k/*_processed.parquet")
    print(f"  MATH:  {args.data_dir}/math/*_processed.parquet")


if __name__ == "__main__":
    main()
