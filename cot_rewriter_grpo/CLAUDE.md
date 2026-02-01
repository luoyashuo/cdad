# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 VERL (Volcano Engine Reinforcement Learning) 框架的深度学习项目，使用 GRPO (Group Relative Policy Optimization) 算法训练 CoT (Chain-of-Thought) Rewriter 模型。目标是优化大语言模型的推理过程。

## 技术栈

- Python 3.10+ (测试于 3.12)
- PyTorch >= 2.6.0 with CUDA >= 12.4
- vLLM >= 0.8.5 (高性能推理引擎)
- Ray >= 2.41.0 (分布式训练)
- Hydra/OmegaConf (配置管理)
- 基础模型: Qwen3 系列 (4B/8B/1.7B)

## 常用命令

### 环境安装
```bash
# 使用 uv 作为包管理器
cd verl && uv pip install -e .
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm==0.8.5.post1 transformers==4.55.0
```

### 启动训练
```bash
# 一键启动（推荐，带 GPU 管理）
bash scripts/grpo/start.sh

# 指定 GPU 启动
bash scripts/grpo/run_train.sh qwen3_4b_grpo_2gpu 0,1

# 直接 Python 启动
python verl/verl/trainer/main_ppo.py --config-path=examples --config-name=qwen3_4b_grpo_2gpu
```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "from verl.trainer.main_ppo import run_ppo; print('VERL OK')"
```

## 系统架构

```
训练流程 (GPU 0-2/3)              奖励计算 (Offload Mode)
┌─────────────────────┐          ┌──────────────────────┐
│ Actor Model (Qwen3) │   HTTP   │ Student Model → R_cog│
│ vLLM Rollout Engine │ ───────▶ │ FSF Judge → R_logic  │
│ GRPO Optimization   │          │ (按需加载到 GPU)     │
└─────────────────────┘          └──────────────────────┘

R_total = R_cog + λ * R_logic
```

## 关键目录和文件

| 路径 | 用途 |
|------|------|
| `examples/*.yaml` | 训练配置文件 |
| `scripts/grpo/run_train.sh` | 核心训练启动脚本 |
| `scripts/grpo/start.sh` | 一键启动脚本（含 GPU 管理） |
| `verl/verl/trainer/main_ppo.py` | VERL PPO 训练入口 |
| `verl/verl/experimental/reward/cot_rewriter/combined_reward.py` | 自定义奖励函数实现 |
| `logs/grpo/` | 训练日志 |
| `outputs/` | 检查点和输出 |

## 配置结构 (examples/*.yaml)

```yaml
trainer:
  n_gpus_per_node: 2
  total_training_steps: 500

actor_rollout_ref:
  model:
    path: /path/to/Qwen3-4B
    lora_rank: 64        # LoRA 参数（40GB GPU）
  rollout:
    n: 8                 # GRPO 采样数

reward_model:
  custom_reward_function:
    reward_kwargs:
      student_model_path: /path/to/Qwen3-1.7B
      fsf_model_path: /path/to/fsf_judge
      lambda_logic: 0.5
      enable_offload: true
```

## 数据格式

训练数据为 JSONL 格式，支持通过 `field_mapping` 配置自动映射字段：
- `problem/prompt`: 输入提示
- `question`: 任务问题
- `thought/original_cot`: 原始推理过程
- `solution/ground_truth`: 正确答案

## 关键环境变量

```bash
RAY_TMPDIR=/tmp/ray_short    # Ray 临时目录（避免 socket 路径过长）
RAY_DEDUP_LOGS=1             # 日志去重
CUDA_VISIBLE_DEVICES=0,1     # GPU 选择
```

## 常见问题

**ModuleNotFoundError: verl** - 需要在 verl 目录执行 `uv pip install -e .`

**Ray 集群错误** - 执行 `pkill -f "ray::" && rm -rf /tmp/ray_*` 清理后重试

**OOM** - 启用 LoRA (`lora_rank: 64`)、降低 batch size、调低 `gpu_memory_utilization`
