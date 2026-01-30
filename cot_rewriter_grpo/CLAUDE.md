# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
你目前工作在SFTP模式下，你在本地只能访问当前workspace的文件，其他文件都在远程服务器，你也无法执行远程服务器上的命令。

## 项目概述

基于 VERL 框架的 CoT 改写模型训练项目，使用 GRPO (Group Relative Policy Optimization) 算法训练模型将原始 Chain-of-Thought 推理改写得更清晰高效。

核心奖励函数: `R = R_cog + λ × R_logic`
- R_cog (认知奖励): 使用冻结的 Student Model 评估改写 CoT 的质量
- R_logic (逻辑奖励): 使用 FSF Judge 分析推理过程中的失败分支

## 常用命令

### 训练启动
```bash
# 默认配置训练 (使用 GPU 1,2,3,5)
bash scripts/grpo/run_train.sh

# 指定配置和GPU
bash scripts/grpo/run_train.sh qwen3_8b_grpo_4gpu 1,2,3,5
```

### 奖励服务器 (当 RewardLoopWorker 无 GPU 时)
```bash
# 启动独立学生奖励计算服务器
bash scripts/grpo/start_reward_server.sh [GPU_ID] [PORT] [MODEL_PATH]
# 默认: GPU 5, 端口 8765
```

### 训练脚本直接调用
```bash
python src/trainer/train.py --config-name=qwen3_8b_grpo_4gpu
```

## 核心架构

```
src/
├── trainer/train.py                    # 训练入口 (Hydra 配置)
├── reward/
│   ├── cot_rewriter_reward_manager.py  # VERL RewardManager 适配器
│   ├── combined_reward.py              # 组合奖励计算入口
│   ├── student_reward.py               # 认知奖励 (log P_student)
│   ├── fsf_reward.py                   # 逻辑奖励 (FSF Judge)
│   └── student_reward_server.py        # 独立奖励计算 HTTP 服务
└── data/
    ├── dataset.py                      # 数据集处理
    └── rl_dataset_jsonl.py             # JSONL 数据加载器

examples/
└── qwen3_8b_grpo_4gpu.yaml            # 主训练配置文件
```

## GPU 分配
- GPU 0-2 (或 1-3): Actor Model + Rollout Engine (vLLM) + GRPO 训练
- GPU 3 (或 5): Reward Models (Student Model + FSF Judge)

## 配置要点

主配置文件: `examples/qwen3_8b_grpo_4gpu.yaml`

关键配置路径:
- `actor_rollout_ref.model.path`: Actor 模型路径
- `reward_model.custom_reward_function`: 自定义奖励函数配置
- `reward_model.reward_kwargs`: 奖励计算参数
  - `student_model_path`: Student Model 路径
  - `fsf_model_path`: FSF Judge 路径
  - `lambda_logic`: 逻辑奖励权重
  - `use_remote_student`: 是否使用远程奖励服务器
  - `field_mapping`: 数据字段映射

## 数据格式

JSONL 格式，支持字段映射:
```json
{
  "problem": "改写提示",
  "question": "具体问题",
  "thought": "原始CoT",
  "solution": "正确答案",
  "fsf": 0.2
}
```

## 奖励函数开发

自定义奖励函数签名:
```python
def compute_cot_rewriter_reward(data: DataProto, tokenizer, **reward_kwargs) -> torch.Tensor:
    # 返回 shape: [batch_size, response_length] 的 token-level 奖励张量
```

RewardManager 注册名: `cot_rewriter_reward_manager`

## 常见问题

1. **CUDA 不可用 (RewardLoopWorker)**: 设置 `reward_kwargs.allow_cpu=true` 或启用 `use_remote_student=true` 并启动奖励服务器
2. **模型路径错误**: 检查配置中的 `student_model_path`、`fsf_model_path`、`actor_rollout_ref.model.path`
3. **字段映射错误**: 检查 `field_mapping` 配置是否与数据格式匹配

## 相关文档

- `README.md`: 项目概述和快速开始
- `VERL_DEV_GUIDE.md`: VERL 框架开发指南和 Reward 实现详解
- `DEBUG.md`: 错误调试和问题排查
