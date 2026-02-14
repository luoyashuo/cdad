# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**CoT Rewriter GRPO** 是基于 VERL 框架的 Chain-of-Thought (CoT) 改写模型训练系统，使用 GRPO (Group Relative Policy Optimization) 算法训练一个能将原始推理过程改写得更清晰、高效的模型。

**核心目标**: 训练一个 CoT Rewriter 模型，能够将原始的 Chain-of-Thought 推理过程改写得更加清晰、高效，减少失败分支，帮助下游模型（Student Model）更好地理解和利用推理过程。

## 环境要求

- **Python**: 3.12 (使用 uv 管理)
- **PyTorch**: 2.6.0 + CUDA 12.4
- **VERL**: 依赖父目录的 verl 框架
- **关键依赖**: vllm==0.8.4, sglang==0.4.6.post5, transformers==4.55.0

### 环境配置

```bash
# 1. 使用 uv 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 2. 安装 VERL 框架（从父目录的 verl/ 中）
cd verl
uv pip install -e .

# 3. 安装其他依赖
uv pip install vllm==0.8.4 sglang[all]==0.4.6.post5 transformers==4.55.0
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install swanlab zmq loguru cachetools==5.3.3
```

### RUC 服务器环境特殊配置

如果在 RUC 服务器上运行（CUDA Driver 525），需要设置：
```bash
export VLLM_USE_V1=0                      # 强制使用 vLLM V0 引擎
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 使用 Flash Attention
```

### QC 服务器环境特殊配置

如果在 QC 服务器上运行（CUDA Driver 525+），需要设置：
```bash
export VLLM_USE_V1=0                      # 强制使用 vLLM V0 引擎
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 使用 Flash Attention
```

## 常用命令

### 启动训练

使用配置文件启动训练：

```bash
# QC 环境 8x80GB GPU (全参数微调)
bash scripts/grpo/start_qc.sh              # 默认使用 GPU 0-7
bash scripts/grpo/start_qc.sh 0,1,2,3      # 指定 GPU
bash scripts/grpo/start_qc.sh 0,1,2,3 llama3_8b_grpo_8x80g  # 指定 GPU 和配置

# RUC 环境 4x80GB GPU (全参数微调)
bash scripts/grpo/start_ruc_4gpu_80g.sh     # 默认使用 GPU 0,1,2,3
bash scripts/grpo/start_ruc_4gpu_80g.sh 4,5,6,7  # 指定 GPU

# 通用 2x40GB GPU (LoRA 微调)
bash scripts/grpo/start.sh
```

### 停止训练

```bash
# RUC 环境
bash scripts/grpo/stop_train_ruc_4gpu.sh

# 手动停止
ps aux | grep main_ppo.py
kill <PID>
```

### 运行测试

```bash
# 运行 sanity tests (CPU)
cd verl
pytest tests/special_sanity/test_import.py -v

# 运行单元测试 (需要 GPU)
pytest tests/utils/test_config_on_cpu.py -v

# 测试 SGLang Student Reward
python scripts/grpo/test_sglang_simple.py
```

### 调试命令

```bash
# 查看训练日志
tail -f logs/grpo/<config_name>.log

# 查看 GPU 状态
watch -n 1 nvidia-smi

# 检查 Ray 进程
ps aux | grep ray

# 清理 Ray 临时文件
rm -rf /tmp/ray_*
```

## 关键配置文件

| 配置文件 | GPU 配置 | 微调方式 | Reward Backend |
|----------|-----------|----------|----------------|
| `examples/llama3_8b_grpo_8x80g.yaml` | 8x80GB | 全参数 | SGLang |
| `examples/qwen3_4b_grpo_8x80g.yaml` | 8x80GB | 全参数 | vLLM |
| `examples/llama3_8b_zero_grpo_8x80g.yaml` | 8x80GB | ZeRO | vLLM |
| `examples/phi_4b_grpo_8x80g.yaml` | 8x80GB | 全参数 | vLLM |
| `examples/qwen3_4b_gspo_8x80g.yaml` | 8x80GB | 全参数 | GSPO |
| `examples/qwen2_5_7b_gspo_8x80g.yaml` | 8x80GB | 全参数 | GSPO |

### 数据字段映射

数据格式配置详见: `examples/field_mapping_examples.md`

```yaml
# 在配置文件中自定义字段映射
custom_reward_function:
  reward_kwargs:
    field_mapping:
      prompt_key: problem             # 问题字段名
      reward_model_key: reward_model  # 奖励模型字段名
      question_key: question          # 问题文本字段名
      original_cot_key: original_cot  # 原始 CoT 字段名
      ground_truth_key: ground_truth  # 正确答案字段名
```

## 系统架构

### Hybrid Engine 设计

VERL 使用 Hybrid Engine 架构，同一模型有两种形态：

1. **FSDP Actor** (PyTorch 训练形态) - 用于梯度计算和参数更新
2. **vLLM Engine** (推理引擎形态) - 用于高效生成 response

### 三阶段训练流程

每个训练步骤包含三个阶段：

1. **Rollout** (~60GB 显存)
   - Actor vLLM 生成改写后的 CoT
   - 代码: `verl/workers/fsdp_workers.py:720-756`

2. **Reward** (~45GB 显存)
   - Student Model 计算认知奖励 R_cog
   - 可选: FSF Judge 计算逻辑奖励 R_logic
   - 代码: `verl/experimental/reward/cot_rewriter/combined_reward.py`

3. **Training** (~55GB 显存)
   - FSDP Actor 执行梯度更新
   - 代码: `verl/workers/fsdp_workers.py:684-707`

### 奖励函数

**总奖励公式**: `R = R_cog + lambda * R_logic`

**认知奖励** (Cognitive Reward):
```
answer_log_prob = mean(token_log_probs) + alpha * min(token_log_probs)
R_cog = answer_log_prob(rewritten) - answer_log_prob(original)
```

**逻辑奖励** (Logic Reward, 可选):
```
FSF(Q, CoT) = 失败分支占比
R_logic = FSF(Q, Original_CoT) - FSF(Q, Rewritten_CoT)
```

当前配置: `use_logic_reward: false`，所以 `R = R_cog`

## 核心代码位置

### VERL 框架修改

本项目基于 VERL 框架做了以下关键修改（位于 `verl/verl/` 下）：

1. **奖励计算模块** - `experimental/reward/cot_rewriter/`
   - `combined_reward.py` - 组合奖励计算入口
   - `student_reward.py` - Student Model vLLM 实现
   - `sglang_student_reward.py` - Student Model SGLang 实现
   - `sglang_fsf_reward.py` - FSF Judge SGLang 实现
   - `reward_manager.py` - 自定义 Reward Manager

2. **FSDP Workers** - `workers/fsdp_workers.py`
   - 支持 vLLM offload/reload 机制
   - 三阶段训练流程实现

3. **显存管理** - `workers/sharding_manager/fsdp_vllm.py`
   - vLLM sleep/wake_up 实现
   - FSDP 参数和 Optimizer offload

4. **数据集处理** - `utils/dataset/rl_dataset_jsonl.py`
   - 支持 JSONL 格式数据
   - 自动转换为 chat message 格式

5. **训练主循环** - `trainer/ppo/ray_trainer.py`
   - GRPO 训练循环
   - 检查点管理

### 训练脚本

- `scripts/grpo/start_qc.sh` - QC 环境 8x80G 启动脚本
- `scripts/grpo/start_ruc_4gpu_80g.sh` - RUC 环境 4x80G 启动脚本
- `scripts/grpo/run_train.sh` - 通用训练脚本（由 start 脚本调用）
- `scripts/grpo/stop_train_ruc_4gpu.sh` - RUC 环境停止脚本

- `scripts/train_fsf_judge/` - FSF Judge 模型训练脚本
- `scripts/serve_judge/` - 奖励模型服务脚本

### 调试/测试脚本

- `scripts/grpo/test_sglang_simple.py` - SGLang Student Reward 测试
- `scripts/grpo/test_model_format_sglang.py` - SGLang 模型格式测试
- `scripts/grpo/test_model_format.py` - 通用模型格式测试

## 文档

- `docs/training_architecture.md` - 详细的训练架构文档
- `docs/MODIFICATIONS_SUMMARY.md` - VERL 框架修改总结
- `docs/batch_size_parameters.md` - Batch size 参数说明
- `docs/DEBUG.md` / `docs/DEBUG_RUC.md` / `docs/DEBUG_QC.md` - 调试指南
- `examples/field_mapping_examples.md` - 数据字段映射说明

## 显存管理关键点

### GPU 内存预算 (8x80GB 配置)

每张 GPU 在不同阶段的显存占用：

| 阶段 | 显存占用 | 主要组件 |
|------|---------|---------|
| Rollout | ~60GB | Actor vLLM (56GB) + FSDP (2GB) |
| Reward | ~45GB | Student vLLM (32GB) + FSDP 残留 (10GB) |
| Training | ~55GB | FSDP Actor (10GB) + Optimizer (20GB) + 梯度 (10GB) + 激活值 (15GB) |

### 关键配置

```yaml
# 启用 offload 以释放显存
actor_rollout_ref.actor.fsdp_config:
  param_offload: true
  optimizer_offload: true

# vLLM 显存配置
actor_rollout_ref.rollout:
  gpu_memory_utilization: 0.7
  sleep_level: 2

# Student Model 显存配置
custom_reward_function.reward_kwargs:
  vllm_gpu_memory_utilization: 0.4
  sglang_mem_fraction_static: 0.5
```

