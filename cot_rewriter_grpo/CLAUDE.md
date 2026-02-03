# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 VERL 框架的 CoT (Chain-of-Thought) Rewriter 训练系统，使用 GRPO (Group Relative Policy Optimization) 算法。训练目标是将原始 CoT 推理过程改写得更清晰高效，减少失败分支。

**技术栈**: Python 3.12, PyTorch 2.6.0, CUDA 12.4, vLLM 0.8.5, Ray 2.41+, Hydra/OmegaConf, Qwen3 系列模型

## 常用命令

```bash
# 环境安装
cd verl && uv pip install -e .
uv pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm==0.8.5.post1 transformers==4.55.0

# 启动训练
bash scripts/grpo/start.sh                              # 一键启动（推荐）
bash scripts/grpo/run_train.sh qwen3_4b_grpo_2gpu 0,1   # 指定 GPU
python verl/verl/trainer/main_ppo.py --config-path=examples --config-name=qwen3_4b_grpo_2gpu

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "from verl.trainer.main_ppo import run_ppo; print('VERL OK')"

# 清理 Ray 状态
pkill -f "ray::" && rm -rf /tmp/ray_*

# 查看训练日志
tail -f logs/grpo/qwen3_4b_grpo_2gpu.log
```

## 系统架构

```
训练流程 (GPU 0-N)                     奖励计算 (Offload Mode)
┌─────────────────────────┐           ┌──────────────────────────┐
│ Actor Model (Qwen3-4B)  │   HTTP    │ Student Model → R_cog    │
│ + vLLM Rollout Engine   │ ────────▶ │ FSF Judge → R_logic      │
│ + GRPO + LoRA           │           │ (按需加载，计算后卸载)   │
└─────────────────────────┘           └──────────────────────────┘

奖励公式:
R_cog = answer_log_prob(rewritten) - answer_log_prob(original)
      其中: answer_log_prob = avg(token_log_probs) + α * min(token_log_probs)
R_logic = 1 if FSF(rewrite_cot) < FSF(original_cot) else 0
R_total = R_cog + λ * R_logic
```

## 关键目录和文件

| 路径 | 用途 |
|------|------|
| `examples/*.yaml` | 训练配置文件 |
| `scripts/grpo/run_train.sh` | 核心训练启动脚本 |
| `verl/verl/trainer/main_ppo.py` | GRPO 训练入口 |
| `verl/verl/trainer/ppo/ray_trainer.py` | Ray 分布式训练器 |
| `verl/verl/experimental/reward/cot_rewriter/` | 自定义奖励函数模块 |
| `verl/verl/experimental/reward/cot_rewriter/combined_reward.py` | 综合奖励计算 |
| `verl/verl/experimental/reward/cot_rewriter/student_reward.py` | Student 模型客户端 |
| `verl/verl/experimental/reward/cot_rewriter/fsf_reward.py` | FSF Judge 实现 |
| `verl/verl/utils/dataset/rl_dataset_jsonl.py` | JSONL 数据集支持 |
| `verl/verl/workers/fsdp_workers.py` | FSDP 训练工作节点 |
| `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM 推理节点 |

## 配置结构 (examples/*.yaml)

```yaml
trainer:
  n_gpus_per_node: 2
  total_training_steps: 500

actor_rollout_ref:
  model:
    path: /path/to/Qwen3-4B
    lora_rank: 64        # LoRA 参数
  rollout:
    n: 8                 # GRPO 采样数
    free_cache_engine: true
    sleep_level: 1       # 必须用 1，不能用 2（见常见问题）

reward_model:
  task_runner_num_gpus: 1              # TaskRunner GPU 资源
  offload_rollout_for_reward: true     # 奖励计算前卸载 vLLM
  custom_reward_function:
    reward_kwargs:
      student_model_path: /path/to/Qwen3-1.7B
      fsf_model_path: /path/to/fsf_judge
      lambda_logic: 0.5
      alpha_cog: 0.01
      enable_offload: true
      field_mapping:                   # 数据字段映射
        question_key: "question"
        original_cot_key: "thought"
        ground_truth_key: "solution"
```

## 数据格式

训练数据为 JSONL 格式，通过 `field_mapping` 自动映射字段到 reward function:
- `problem/prompt` → 输入提示 (转换为 chat 格式)
- `question` → 任务问题
- `thought/original_cot` → 原始 CoT
- `solution/ground_truth` → 正确答案

## 关键环境变量

```bash
RAY_TMPDIR=/tmp/ray_${USER}           # Ray 临时目录（避免 socket 路径过长）
RAY_DISABLE_DASHBOARD_AGENT=1         # 禁用 Dashboard（避免 OpenTelemetry 冲突）
VLLM_USE_V1=0                         # 强制 vLLM V0 引擎
VLLM_ATTENTION_BACKEND=FLASH_ATTN     # 使用 Flash Attention
VERL_ROLLOUT_LOGGING_LEVEL=INFO       # vLLM 详细日志
```

## VERL 框架修改要点

本项目对 VERL 0.6.1 进行了以下关键修改（详见 `MODIFICATIONS_SUMMARY.md`）:

1. **新增 `cot_rewriter` 奖励模块**: 支持 Student 模型 + FSF Judge 的组合奖励
2. **vLLM 内存管理**: `offload_for_reward` / `reload_after_reward` 方法，通过销毁+重建绕过 `sleep(level=2)` 的问题
3. **TaskRunner GPU 配置**: `task_runner_num_gpus` 参数为奖励计算分配 GPU
4. **JSONL 数据集**: `RLHFDatasetJSONL` 支持纯文本 prompt 自动转 chat 格式

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| Ray 集群/socket 错误 | `pkill -f "ray::" && rm -rf /tmp/ray_*` |
| OOM | 启用 LoRA、降低 batch size、调低 `gpu_memory_utilization` |
| vLLM 无法释放内存 | 启用 `offload_rollout_for_reward: true` |
| NVIDIA 驱动版本低 | 升级到 535+ (RUC 服务器需使用 fp16 而非 bf16) |
| **第二个 step 开始输出乱码** | **必须设置 `sleep_level: 1`，不能用 2**（见下方详解） |

### sleep_level=2 导致乱码问题（重要）

**现象**：训练第一个 step 正常，第二个 step 开始 rollout 输出变成乱码（如 `"1 2 the def2 `` ## over..."`）

**原因**：
- `sleep_level=2` 会完全丢弃 vLLM 的模型权重
- `wake_up` 时需要重新初始化模型，此时使用 dummy 权重
- 虽然 `update_params` 会用训练后的权重覆盖，但 vLLM 内部实现有 bug，部分权重未被正确同步
- 结果是模型部分权重是随机值，输出乱码

**解决方案**：
```yaml
actor_rollout_ref:
  rollout:
    sleep_level: 1  # 权重 offload 到 CPU，wake_up 时从 CPU 恢复（正确）
    # sleep_level: 2  # 完全丢弃权重，wake_up 有 bug（错误）
```

**显存影响**：`sleep_level=1` 只比 `sleep_level=2` 多占用约 1-2GB GPU 显存，在 80GB GPU 上完全足够。
