#!/usr/bin/env python3
"""
临时修复 DeepSpeed 在缺少 CUDA_HOME 时的问题
"""

import sys
import os

# 在导入 deepspeed 之前修改 builder.py 的 installed_cuda_version 函数
import deepspeed.ops.op_builder.builder as builder_module

# 保存原始函数
original_installed_cuda_version = builder_module.installed_cuda_version


def modified_installed_cuda_version(name=""):
    try:
        return original_installed_cuda_version(name)
    except Exception as e:
        print(f"WARNING: CUDA installation check failed: {e}")
        return (12, 8)  # 返回 PyTorch 的 CUDA 版本


builder_module.installed_cuda_version = modified_installed_cuda_version

# 保存原始的 is_compatible 方法
import deepspeed.ops.op_builder.fp_quantizer as fp_quantizer_module

original_is_compatible = fp_quantizer_module.FPQuantizerBuilder.is_compatible


def modified_is_compatible(self, verbose=False):
    try:
        return original_is_compatible(self, verbose)
    except Exception as e:
        print(f"WARNING: fp_quantizer compatibility check failed: {e}, skipping")
        return False


fp_quantizer_module.FPQuantizerBuilder.is_compatible = modified_is_compatible

# 现在可以安全地导入 deepspeed 了
print("DeepSpeed 修复完成，可以正常导入")
