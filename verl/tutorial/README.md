# Verl GRPO 实战教程

> 构建一个基于 GRPO (Group Relative Policy Optimization) 算法的 CoT Rewriter 训练系统

## 教程目标

本教程将指导你使用 **Verl** 框架训练一个 Chain-of-Thought 改写模型，目标是让改写后的推理过程能更好地帮助 Student Model 预测正确答案。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    CoT Rewriter 训练系统                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: Question + Original CoT                                 │
│      ↓                                                           │
│   ┌─────────────────────────────────────────────────────┐       │
│   │         Actor Model (Qwen-7B + LoRA)                │       │
│   │              生成 Rewrite CoT                        │       │
│   └─────────────────────────────────────────────────────┘       │
│      ↓                                                           │
│   ┌─────────────────────────────────────────────────────┐       │
│   │              Reward Computation                      │       │
│   │  ┌──────────────────┐  ┌──────────────────┐         │       │
│   │  │  R_cog (Student) │  │  R_logic (FSF)   │         │       │
│   │  │  认知奖励         │  │  逻辑奖励         │         │       │
│   │  └──────────────────┘  └──────────────────┘         │       │
│   │           R = R_cog + λ · R_logic                    │       │
│   └─────────────────────────────────────────────────────┘       │
│      ↓                                                           │
│   ┌─────────────────────────────────────────────────────┐       │
│   │            GRPO Advantage Estimation                 │       │
│   │     advantage = (R - group_mean) / group_std         │       │
│   └─────────────────────────────────────────────────────┘       │
│      ↓                                                           │
│   Output: Updated Actor Model                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 教程目录

### [01. Verl 架构详解](01_Verl_Architecture/README.md)

- Verl 核心设计理念
- Driver-Worker 分布式架构
- Hybrid Engine (Actor + Rollout 共享显存)
- FSDP 集成与配置
- 优势估计算法 (GAE/GRPO/RLOO)
- DataProto 数据协议

### [02. 环境配置](02_Environment_Setup/README.md)

- 硬件要求
- 使用 uv 管理 Python 环境
- 安装 Verl 及依赖
- vLLM/SGLang 推理引擎配置
- 模型下载
- 数据准备
- 环境验证

### [03. CoT Rewriter GRPO 训练](03_CoT_Rewrite_GRPO/README.md)

- 项目结构设计
- 认知奖励 (R_cog) 实现
- 逻辑奖励 (R_logic) 实现
- 组合奖励函数
- 配置文件详解
- 显存优化策略
- 训练监控
- 评估与分析

## 快速开始

```bash
# 1. 克隆项目
cd verl/tutorial/03_CoT_Rewrite_GRPO

# 2. 安装依赖
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. 准备数据和模型
# 下载 Qwen-2.5-7B-Instruct (Actor) 和 Qwen-2.5-3B-Instruct (Student)
# 准备训练数据到 data/processed/train.parquet

# 4. 验证奖励函数
python scripts/verify_reward.py

# 5. 启动训练
python scripts/train.py
```

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| GPU | 4 × A100 (40G) | 4 × A100 (80G) |
| CPU 内存 | 128 GB | 256 GB |
| 磁盘 | 500 GB SSD | 1 TB NVMe |
| CUDA | 12.1+ | 12.4 |

## 核心概念

### GRPO vs PPO

| 特性 | PPO | GRPO |
|------|-----|------|
| 需要 Critic | ✅ | ❌ |
| 奖励类型 | Token-level | Outcome |
| 显存占用 | 高 | 低 |
| 适用场景 | 细粒度反馈 | 序列级反馈 |

### 组合奖励函数

```
R = R_cog + λ · R_logic

R_cog   = log P_student(Answer | Question, Rewrite_CoT)  # 认知奖励
R_logic = 1 if FSF(Rewrite) < FSF(Original) else 0      # 逻辑奖励
```

## 参考资料

- [Verl GitHub](https://github.com/volcengine/verl)
- [GRPO 论文](https://arxiv.org/abs/2402.03300)
- [Dr.GRPO 论文](https://arxiv.org/abs/2503.20783)
- [PyTorch FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)

## 许可证

本教程代码遵循 Apache 2.0 许可证。
