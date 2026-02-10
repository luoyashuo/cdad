# Code Surf 笔记

## TaskRunnerRemote

**位置**: `verl/verl/trainer/main_ppo.py:74-93`

**作用**: TaskRunnerRemote 是将 `TaskRunner` 类动态包装成 Ray Remote Actor 的结果。

### 代码逻辑

```python
task_runner_num_gpus = config.reward_model.get("task_runner_num_gpus", 0)

if task_runner_num_gpus > 0:
    TaskRunnerRemote = ray.remote(num_cpus=1, num_gpus=task_runner_num_gpus)(TaskRunner)
else:
    TaskRunnerRemote = ray.remote(num_cpus=1)(TaskRunner)

runner = TaskRunnerRemote.remote()
ray.get(runner.run.remote(config))
```

### 核心功能

1. **动态资源分配**: 根据 `task_runner_num_gpus` 配置决定是否为 TaskRunner 分配 GPU 资源
   - 当 `task_runner_num_gpus > 0` 时，TaskRunner 会占用指定数量的 GPU（用于 reward 计算）
   - 当 `task_runner_num_gpus = 0` 时，TaskRunner 只使用 CPU

2. **进程隔离**: 通过 `ray.remote()` 装饰器，TaskRunner 在独立的 Ray Actor 进程中运行，与主进程隔离

3. **训练入口**: `TaskRunner.run()` 方法是整个 GRPO 训练的实际执行入口，负责:
   - 加载模型和 tokenizer
   - 初始化 FSDP workers
   - 配置资源池
   - 创建数据集
   - 启动 RayPPOTrainer

### 为什么不直接用 @ray.remote 装饰器

代码注释说明:
> Note: Do NOT add @ray.remote decorator here - it's applied dynamically in run_ppo() to support configurable GPU resources for reward computation

因为需要根据配置文件动态决定 GPU 资源数量，所以不能在类定义时使用静态的 `@ray.remote` 装饰器，而是在运行时动态应用。

### 调用链

```
main()
  → run_ppo(config)
    → TaskRunnerRemote = ray.remote(...)(TaskRunner)
    → runner = TaskRunnerRemote.remote()
    → ray.get(runner.run.remote(config))
        → TaskRunner.run()
            → RayPPOTrainer.fit()
```
