#!/usr/bin/env python3
"""
CoT Rewriter GRPO 训练入口

使用方法:
    python scripts/train.py
    python scripts/train.py trainer.total_epochs=5 data.train_batch_size=32
"""

import os
import sys

# 添加 src 到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import hydra
from omegaconf import DictConfig, OmegaConf
import ray


def setup_environment():
    """配置环境变量"""
    # Ray 配置
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

    # NCCL 配置
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    # 如果有多个 GPU，设置可见设备
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


@hydra.main(config_path="../configs", config_name="cot_rewriter_grpo", version_base=None)
def main(config: DictConfig):
    """
    训练主函数
    """
    setup_environment()

    print("=" * 60)
    print("CoT Rewriter GRPO Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=os.cpu_count(),
            num_gpus=4,  # 根据实际情况修改
            object_store_memory=50 * 1024**3,
        )

    try:
        # 导入 Verl 训练入口
        from verl.trainer.main_ppo import main as verl_main

        # 启动训练
        verl_main(config)
    finally:
        # 清理 Ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
