#!/usr/bin/env python3
"""
上传数据文件到 HuggingFace Hub 仓库 Yaseal/CDAD_EXP

用法:
    python upload_data_hf.py                    # 上传所有预定义的文件/目录
    python upload_data_hf.py --file-only        # 只上传单个文件
    python upload_data_hf.py --dir-only         # 只上传目录
    python upload_data_hf.py --overwrite        # 上传并覆盖已存在的文件
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi


# ==================== 配置区域 ====================

# HuggingFace 配置
REPO_ID = "Yaseal/CDAD_EXP"
TOKEN = "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"

# 要上传的目录列表 (本地路径, 远程路径)
# 本地路径基于 /data/hf_home/datasets/Yaseal/CDAD_EXP/
UPLOAD_DIRS: List[Tuple[str, str]] = [
    # ("cot_rewrite/qwen2_5_7b_grpo_8x40g/final", "cot_rewrite/qwen2_5_7b_grpo_8x40g/final"),
    # ("fsf_grm", "fsf_grm"),
    # ("cold_start/llama3_3b_instruct_top2k", "cold_start/llama3_3b_instruct_top2k"),
    
]

# 要上传的单个文件列表 (本地路径, 远程路径)
UPLOAD_FILES: List[Tuple[str, str]] = [
    # ("cot_rewrite/qwen2_5_7b_grpo_8x40g/final/dual_30k.jsonl", "cot_rewrite/qwen2_5_7b_grpo_8x40g/final/dual_30k.jsonl"),
    # ("fsf_grm/train.jsonl", "fsf_grm/train.jsonl"),
]

# 基础目录
BASE_DIR = Path("/data/hf_home/datasets/Yaseal/CDAD_EXP")

# ==================================================


def upload_file(
    local_path: Path,
    remote_path: str,
    repo_id: str = REPO_ID,
    token: str = TOKEN,
    overwrite: bool = False
) -> bool:
    """上传单个文件到 HuggingFace Hub"""
    
    if not local_path.exists():
        print(f"  ✗ 文件不存在: {local_path}")
        return False
    
    if not local_path.is_file():
        print(f"  ✗ 路径不是文件: {local_path}")
        return False
    
    # 规范化远程路径
    remote_path = remote_path.lstrip('/')
    
    try:
        api = HfApi(token=token)
        
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            overwrite=overwrite
        )
        
        print(f"  ✓ 上传成功: {remote_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ 上传失败: {e}")
        return False


def upload_directory(
    local_dir: Path,
    remote_base: str,
    repo_id: str = REPO_ID,
    token: str = TOKEN,
    overwrite: bool = False
) -> int:
    """上传整个目录到 HuggingFace Hub"""
    
    if not local_dir.exists():
        print(f"  ✗ 目录不存在: {local_dir}")
        return 0
    
    if not local_dir.is_dir():
        print(f"  ✗ 路径不是目录: {local_dir}")
        return 0
    
    # 收集所有文件
    files_to_upload = []
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_dir)
            remote_path = f"{remote_base}/{rel_path}".replace("//", "/")
            files_to_upload.append((file_path, remote_path))
    
    if not files_to_upload:
        print(f"  ! 目录为空: {local_dir}")
        return 0
    
    print(f"  找到 {len(files_to_upload)} 个文件")
    
    success_count = 0
    for local_file, remote_file in files_to_upload:
        if upload_file(local_file, remote_file, repo_id, token, overwrite):
            success_count += 1
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="上传数据文件到 HuggingFace Hub 仓库 Yaseal/CDAD_EXP"
    )
    parser.add_argument(
        "--file-only", "-f",
        action="store_true",
        help="只上传单个文件列表中的文件"
    )
    parser.add_argument(
        "--dir-only", "-d",
        action="store_true",
        help="只上传目录列表中的目录"
    )
    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="覆盖已存在的文件"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定，默认都上传
    upload_files_flag = not args.dir_only or args.file_only
    upload_dirs_flag = not args.file_only or args.dir_only
    
    print("=" * 60)
    print(f"目标仓库: {REPO_ID}")
    print(f"基础目录: {BASE_DIR}")
    print(f"覆盖模式: {args.overwrite}")
    print("=" * 60)
    
    total_success = 0
    total_files = 0
    
    # 上传单个文件
    if upload_files_flag and UPLOAD_FILES:
        print(f"\n【上传单个文件】共 {len(UPLOAD_FILES)} 个")
        for local_rel, remote_rel in UPLOAD_FILES:
            local_path = BASE_DIR / local_rel
            print(f"\n  上传: {local_rel}")
            if upload_file(local_path, remote_rel, REPO_ID, TOKEN, args.overwrite):
                total_success += 1
            total_files += 1
    
    # 上传目录
    if upload_dirs_flag and UPLOAD_DIRS:
        print(f"\n【上传目录】共 {len(UPLOAD_DIRS)} 个")
        for local_rel, remote_rel in UPLOAD_DIRS:
            local_path = BASE_DIR / local_rel
            print(f"\n  目录: {local_rel} -> {remote_rel}")
            count = upload_directory(local_path, remote_rel, REPO_ID, TOKEN, args.overwrite)
            total_success += count
            total_files += count
    
    # 汇总
    print("\n" + "=" * 60)
    if total_files == 0:
        print("提示: 请在 UPLOAD_DIRS 或 UPLOAD_FILES 中配置要上传的内容")
        print("""
配置示例:
    UPLOAD_DIRS = [
        ("cot_rewrite/qwen2_5_7b_grpo_8x40g/final", "cot_rewrite/qwen2_5_7b_grpo_8x40g/final"),
    ]
    
    UPLOAD_FILES = [
        ("fsf_grm/train.jsonl", "fsf_grm/train.jsonl"),
    ]
""")
    else:
        print(f"上传完成: {total_success}/{total_files} 个文件成功")
    print("=" * 60)


if __name__ == "__main__":
    main()
