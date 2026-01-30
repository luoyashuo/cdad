# Reward Model Load/Offload 快速上手

## 快速开始

### 1. 使用预配置启动（最简单）

```bash
# 直接使用 offload 启动脚本
bash scripts/grpo/start_with_offload.sh
```

这个脚本会：
- ✅ 使用所有可用 GPU 进行 GRPO 训练
- ✅ Reward models 自动在 CPU 和 GPU 之间切换
- ✅ 无需独立的 Reward Server
- ✅ 自动清理残留进程

### 2. 或者使用配置文件启动

```bash
# 使用 offload 配置
bash scripts/grpo/run_train.sh qwen3_8b_grpo_with_offload 1,2,3,5
```

## 核心优势

### 对比传统方式

| 特性 | 传统方式 | Load/Offload 方式 |
|-----|---------|------------------|
| GPU 数量要求 | N+1 张 (N 用于训练，1 用于 Reward) | N 张（全部用于训练） |
| 启动复杂度 | 需要先启动 Reward Server | 直接启动训练 |
| GPU 利用率 | Reward GPU 长期占用 | 动态共享，按需使用 |
| 显存占用 | Reward 模型常驻显存 | 仅计算时占用 |

### 实际效果

**场景**: 4 张 GPU (1,2,3,5)

**传统方式**:
```
GPU 5: Reward Server (Student Model + FSF Judge) - 常驻
GPU 1,2,3: GRPO 训练
```

**Offload 方式**:
```
GPU 1,2,3,5: 全部用于 GRPO 训练
CPU: Reward Models 待命
计算 reward 时: 临时使用 GPU 1 → 计算完成 → 释放
```

## 核心配置

### 配置文件: `examples/qwen3_8b_grpo_4gpu_with_offload.yaml`

关键配置:
```yaml
reward_model:
  custom_reward_function:
    module: src.reward.combined_reward_with_offload
    function: compute_cot_rewriter_reward_with_offload

  reward_kwargs:
    # Load/Offload 配置
    enable_offload: true        # 启用 offload 机制
    device: cuda:0              # 计算时使用的 GPU
    offload_device: cpu         # 不用时保存位置

    # 模型路径
    student_model_path: /data-share/LLM/Qwen3-1.7B
    fsf_model_path: /data-share/LLM/Qwen3-8B-Instruct

    # 其他配置
    use_logic_reward: true
    lambda_logic: 0.5
    allow_cpu: true
```

## 工作流程

### 训练流程图

```
初始状态:
├── GPU 1,2,3,5: GRPO 训练 (Actor + vLLM)
└── CPU: Reward Models (待命)

Rollout 完成，需要计算 Reward:
├── Step 1: 加载 Student Model 到 GPU 1
├── Step 2: 计算 Cognitive Reward
├── Step 3: 卸载 Student Model 到 CPU
├── Step 4: 加载 FSF Judge 到 GPU 1
├── Step 5: 计算 Logic Reward
├── Step 6: 卸载 FSF Judge 到 CPU
└── Step 7: 返回 Combined Reward

继续训练:
└── GPU 1,2,3,5: GRPO 训练继续
```

## 核心代码实现

### 1. Student Reward Client (自动 load/offload)

文件: `src/reward/student_reward_with_offload.py`

```python
class HFStudentRewardClientWithOffload:
    def __init__(self, enable_offload=True, device="cuda:0"):
        # 模型初始化在 CPU
        self.enable_offload = enable_offload
        self.compute_device = device
        self.offload_device = torch.device("cpu")

    def compute_cognitive_reward_batch(self, ...):
        # 自动 load → compute → offload
        if self.enable_offload:
            self.load_to_gpu()
        try:
            # 计算 reward
            result = ...
        finally:
            if self.enable_offload:
                self.offload_to_cpu()
        return result
```

### 2. Combined Reward (协调多个模型)

文件: `src/reward/combined_reward_with_offload.py`

```python
def compute_reward(self, ...):
    # Step 1: Student Model (auto load/offload)
    cog_result = self.student_client.compute_cognitive_reward_batch(...)

    # Step 2: FSF Judge (auto load/offload)
    logic_result = self.fsf_judge.compute_logic_reward(...)

    # Step 3: Combine
    total_rewards = [c + λ * l for c, l in zip(r_cog, r_logic)]
    return total_rewards
```

## 性能调优

### 优化 1: 选择合适的 GPU

```yaml
# 方案 A: 使用第一张 GPU（推荐）
device: cuda:0

# 方案 B: 使用最后一张 GPU
device: cuda:3

# 选择原则: 选择负载相对较低的 GPU
```

### 优化 2: 调整 Batch Size

```yaml
# 增加 batch size 可以减少 reward 计算频率
rollout_batch_size: 512  # 增大 → 计算频率降低
```

### 优化 3: 禁用 Logic Reward（可选）

```yaml
# 只使用 Student Model，不使用 FSF Judge
reward_kwargs:
  use_logic_reward: false
  # fsf_model_path: ...  # 不需要
```

## 故障排查

### 问题 1: GPU OOM

```yaml
# 解决方案: 减少 reward workers
reward_worker:
  num_workers: 1  # 降低到 1
```

### 问题 2: 训练速度变慢

```yaml
# 解决方案: 增加 rollout batch size
rollout_batch_size: 1024  # 增大
```

### 问题 3: 模型加载失败

```yaml
# 解决方案: 启用 CPU fallback
reward_kwargs:
  allow_cpu: true
```

## 监控和验证

### 1. 查看日志

```bash
# 查看训练日志
tail -f logs/grpo/qwen3_8b_grpo_with_offload.log

# 查看 load/offload 日志
grep -i "loading\|offload" logs/grpo/qwen3_8b_grpo_with_offload.log
```

### 2. 监控 GPU

```bash
# 实时监控 GPU 使用
watch -n 1 nvidia-smi

# 你应该看到：
# - GRPO 训练时: 所有 GPU 都在使用
# - 计算 Reward 时: 某个 GPU 显存临时增加
```

### 3. 验证日志输出

正常日志应该包含:
```
[HFStudentRewardClient] Loading model to cuda:0...
[HFStudentRewardClient] Model loaded to cuda:0
[CombinedRewardComputer] Computing cognitive rewards...
[HFStudentRewardClient] Offloading model to cpu...
[HFStudentRewardClient] Model offloaded to cpu
```

## 完整命令参考

### 启动训练

```bash
# 方式 1: 使用 offload 脚本（推荐）
bash scripts/grpo/start_with_offload.sh

# 方式 2: 直接运行训练
bash scripts/grpo/run_train.sh qwen3_8b_grpo_with_offload 1,2,3,5

# 方式 3: Python 直接启动
python src/trainer/train.py --config-name=qwen3_8b_grpo_with_offload
```

### 停止训练

```bash
# 停止当前训练
pkill -f "train.py"

# 清理 Ray
ray stop --force
```

## 更多文档

- 📖 详细文档: `docs/REWARD_MODEL_OFFLOAD.md`
- ⚙️ 配置文件: `examples/qwen3_8b_grpo_4gpu_with_offload.yaml`
- 🚀 启动脚本: `scripts/grpo/start_with_offload.sh`
- 💻 源代码: `src/reward/*_with_offload.py`

## 总结

使用 Load/Offload 机制，你可以：

✅ **节省 GPU**: 所有 GPU 都用于训练，无需独立 Reward Server
✅ **简化部署**: 一条命令启动，无需复杂的 GPU 分配策略
✅ **灵活调度**: 自动管理模型加载和卸载，无需手动干预
✅ **易于调试**: 所有进程在一起，日志更清晰

开始使用:
```bash
bash scripts/grpo/start_with_offload.sh
```
