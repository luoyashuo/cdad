# VERL 修改总结 - CoT Rewriter GRPO

本文档详细记录了从 verl 0.6.1 修改的功能，便于迁移到 verl 0.5.0。

---

## 零、项目背景与修改目标

### 0.1 项目概述

**CoT Rewriter GRPO** 是基于 VERL 框架的 Chain-of-Thought (CoT) 改写模型训练系统，使用 GRPO (Group Relative Policy Optimization) 算法。

**核心目标**: 训练一个 **CoT Rewriter** 模型，能够将原始的 Chain-of-Thought 推理过程改写得更加清晰、高效，减少失败分支，帮助下游模型（Student Model）更好地理解和利用推理过程。

### 0.2 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training System                               │
├─────────────────────────────────────────────────────────────────────┤
│  GPU 0-2: Actor Model (Qwen3-8B)                                    │
│           + Rollout Engine (vLLM/SGLang)                            │
│           + GRPO Training                                            │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP API (Reward Computation)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 3: Reward Models                                                │
│    ├── Student Model (Qwen3-1.7B) → R_cog (Cognitive Reward)        │
│    └── FSF Judge (Qwen2.5-7B)     → R_logic (Logic Reward)          │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.3 奖励函数设计

**总奖励公式**: `R = R_cog + λ * R_logic`

#### 认知奖励 (Cognitive Reward, R_cog)

```
R_cog = answer_log_prob(rewritten) - answer_log_prob(original)

其中: answer_log_prob = avg(token_log_probs) + α * min(token_log_probs)

AND R_cog = -1 if rewritten被截断
```

- 使用冻结的 Student Model 计算改写后 CoT 对答案预测的概率
- **新设计**: 使用连续值奖励（而非二元0/1），并引入 min 项惩罚不确定性
- 目的：激励改写的 CoT 能让 Student Model 更容易预测出正确答案

#### 逻辑奖励 (Logic Reward, R_logic)

```
R_logic = 1 if FSF(Rewrite_CoT) < FSF(Original_CoT) else 0
```

- **FSF (Failed Step Fraction)**: 失败分支占比
- 使用 LLM-as-a-Judge 分析推理过程中的失败分支
- 目的：鼓励减少无效的推理分支

### 0.4 为什么需要这些修改？

#### 修改动机

1. **自定义奖励函数**: VERL 原生不支持复杂的奖励计算逻辑（需要调用 Student Model 和 FSF Judge），需要新增 `cot_rewriter` 奖励模块

2. **GPU 内存管理**: 
   - 4 GPU 环境下，Actor/Rollout 占用 3 个 GPU，第 4 个 GPU 需要运行 Reward Models
   - 需要支持 vLLM 卸载（offload）以释放 GPU 内存给 Reward Models 使用
   - 支持模型在 CPU/GPU 之间动态切换

3. **多模式奖励计算**:
   - 支持本地 HuggingFace 模型
   - 支持多 GPU 并行
   - 支持 vLLM 加速
   - 支持远程 HTTP 服务

4. **数据格式兼容**:
   - 原始数据使用 JSONL 格式，包含纯文本 prompt
   - 需要转换为 VERL 所需的 chat message 格式
   - 需要传递 `original_cot`、`ground_truth` 等字段给 reward function

5. **调试与监控**:
   - 需要详细的日志追踪训练和奖励计算过程
   - 需要监控 GPU 内存使用情况
   - 需要计时分析各阶段性能

#### 关键技术挑战

| 挑战 | 解决方案 |
|------|----------|
| vLLM 无法正确释放 GPU 内存 | 销毁+重建机制，绕过 allocator 检查 |
| Ray 的 OpenTelemetry 冲突 | 禁用 Ray Dashboard Agent |
| TaskRunner 无法访问 GPU | 配置 `task_runner_num_gpus` |
| 纯文本 prompt 格式不兼容 | `RLHFDatasetJSONL` 自动转换 |
| 奖励模型与训练竞争 GPU | `offload_rollout_for_reward` 机制 |

---

## 一、新增文件

### 1. `verl/experimental/reward/cot_rewriter/` 目录

#### 1.1 `__init__.py`
**功能**: 导出 CoT Rewriter 奖励模块的所有公共接口

**导出内容**:
- `compute_cot_rewriter_reward` - 主奖励计算函数
- `CombinedRewardComputer` - 组合奖励计算器
- `CoTRewriterRewardManager` - 奖励管理器
- `HFStudentRewardClient` - HuggingFace Student 模型客户端
- `RemoteStudentRewardClient` - 远程 Student 模型客户端
- `HFFSFJudge` - FSF 评判模型

#### 1.2 `combined_reward.py`
**功能**: 组合奖励计算的核心实现

**主要类**:
```python
class CombinedRewardComputer:
    """
    计算 R = R_cog + lambda * R_logic
    
    支持模式:
    1. 本地常驻模式: 模型加载到GPU并保持
    2. 本地卸载模式: 模型在CPU，计算时临时移到GPU
    3. 远程模式: 通过HTTP调用单独的GPU服务器
    4. 多GPU模式: 使用多个GPU并行计算
    5. vLLM模式: 使用vLLM加速log prob计算
    """
```

**Reward 计算逻辑**:
```
新 Reward 公式:
- answer_log_prob(q, cot, a) = avg(token_log_probs) + alpha_cog * min(token_log_probs)
- R_cog = answer_log_prob(rewritten) - answer_log_prob(original)
- (可选) R_logic = FSF based reward
- Total = R_cog + lambda * R_logic
```

**关键参数**:
- `student_model_path`: Student 模型路径
- `fsf_model_path`: FSF Judge 模型路径
- `device`: 运行设备
- `lambda_logic`: 逻辑奖励权重
- `alpha_cog`: min log prob 项权重（默认 0.01）
- `enable_offload`: 是否启用CPU卸载
- `use_multi_gpu`: 是否使用多GPU
- `use_vllm`: 是否使用vLLM

#### 1.3 `fsf_reward.py`
**功能**: FSF (Failure-Success-Failure) Judge 实现

**主要类**:
```python
class HFFSFJudge:
    """使用HuggingFace Transformers的FSF评判模型"""
    
    def compute_logic_reward(self, original_cots, rewrite_cots):
        """计算逻辑奖励，基于FSF比较"""
```

#### 1.4 `student_reward.py`
**功能**: Student 模型客户端实现

**主要类**:
```python
class HFStudentRewardClient:
    """本地HuggingFace Student模型客户端"""
    
class MultiGPUStudentRewardClient:
    """多GPU并行Student模型客户端"""
    
class VLLMStudentRewardClient:
    """vLLM加速的Student模型客户端"""
    
class RemoteStudentRewardClient:
    """远程HTTP Student模型客户端"""
```

**核心方法**:
```python
def compute_cognitive_reward_batch(self, questions, rewrite_cots, answers, original_cots=None):
    """
    计算认知奖励
    
    返回:
    - rewards: rewrite_log_prob - original_log_prob
    - rewrite_mean_log_probs: rewrite的平均log prob
    - original_mean_log_probs: original的平均log prob
    - rewrite_token_log_probs: 每个token的log prob
    """
```

#### 1.5 `reward_manager.py`
**功能**: VERL 兼容的奖励管理器

**主要类**:
```python
class CoTRewriterRewardManager(AbstractRewardManager):
    """
    VERL兼容的CoT Rewriter奖励管理器
    
    职责:
    1. 管理奖励计算的生命周期
    2. 处理模型加载/卸载
    3. 与VERL训练流程集成
    """
```

#### 1.6 `logging_config.py`
**功能**: 日志配置

### 2. `verl/utils/dataset/rl_dataset_jsonl.py`
**功能**: 支持 JSONL 格式的数据集

**主要类**:
```python
class RLHFDatasetJSONL(RLHFDataset):
    """
    扩展的RLHFDataset，支持:
    1. JSONL文件格式
    2. 纯文本prompt转换为chat格式
    3. 自定义字段映射
    """
    
    def _build_messages(self, example: dict):
        """将纯文本prompt转换为chat格式"""
        # 如果prompt是字符串，转换为: [{"role": "user", "content": prompt}]
        
    def __getitem__(self, item):
        """
        生成必要的tensor字段:
        - input_ids
        - attention_mask
        - position_ids
        - raw_prompt_ids
        """
```

**数据字段映射**:
```python
# 从数据文件读取的字段 -> reward_model字段
"solution" -> reward_model["ground_truth"]
"thought" -> reward_model["original_cot"]
"question" -> reward_model["question"]
"fsf" -> reward_model["fsf"]
```

---

## 二、修改的文件

### 1. `verl/experimental/reward/__init__.py`

**修改内容**:
```python
# 新增导入
from .cot_rewriter import (
    compute_cot_rewriter_reward,
    CombinedRewardComputer,
    CoTRewriterRewardManager,
    HFStudentRewardClient,
    RemoteStudentRewardClient,
    HFFSFJudge,
)

# 更新 __all__
__all__ = [
    "RewardModelManager",
    "RewardManagerWorker",
    # CoT Rewriter exports
    "compute_cot_rewriter_reward",
    "CombinedRewardComputer",
    "CoTRewriterRewardManager",
    "HFStudentRewardClient",
    "RemoteStudentRewardClient",
    "HFFSFJudge",
]
```

### 2. `verl/utils/dataset/__init__.py`

**修改内容**:
```python
from .rl_dataset_jsonl import RLHFDatasetJSONL

__all__ = ["RLHFDataset", "RLHFDatasetJSONL", "RMDataset", "SFTDataset"]
```

### 3. `verl/trainer/main_ppo.py`

**修改内容**:

#### 3.1 Ray Dashboard 禁用（避免 OpenTelemetry 冲突）
```python
# 文件开头添加
os.environ.setdefault("RAY_DISABLE_DASHBOARD_AGENT", "1")
```

#### 3.2 Ray 初始化改进
```python
# 原代码
ray_init_kwargs = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
ray.init(**ray_init_kwargs)

# 修改为
ray_init_dict = OmegaConf.to_container(ray_init_kwargs, resolve=True)
ray_init_dict["runtime_env"] = OmegaConf.to_container(runtime_env, resolve=True)
ray_init_dict.setdefault("include_dashboard", False)  # 禁用dashboard
ray_init_dict = {k: v for k, v in ray_init_dict.items() if v is not None}
ray.init(**ray_init_dict)
```

#### 3.3 TaskRunner GPU 资源配置
```python
# 新增: TaskRunner可以配置GPU资源用于奖励计算
task_runner_num_gpus = config.reward_model.get("task_runner_num_gpus", 0)
if task_runner_num_gpus > 0:
    task_runner_class = ray.remote(num_cpus=1, num_gpus=task_runner_num_gpus)(TaskRunner)
else:
    task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
```

#### 3.4 ResourcePoolManager GPU 资源调整
```python
# 创建ResourcePoolManager时传递task_runner_num_gpus
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec=resource_pool_spec,
    mapping=self.mapping,
    task_runner_num_gpus=task_runner_num_gpus  # 新增参数
)
```

### 4. `verl/trainer/ppo/ray_trainer.py`

**修改内容**:

#### 4.1 ResourcePoolManager 类添加 task_runner_num_gpus 字段
```python
@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    task_runner_num_gpus: float = 0.0  # 新增: TaskRunner GPU资源
```

#### 4.2 GPU 资源检查调整
```python
def create_resource_pool(self):
    # ... 原有代码 ...
    total_available_gpus = sum(node_available_gpus.values())
    
    # 新增: 加回TaskRunner GPU资源（因为与训练GPU时间共享）
    total_available_gpus += self.task_runner_num_gpus
    
    # ... 原有代码 ...
```

#### 4.3 验证流程增强
```python
def _validate(self):
    print("[VALIDATE] Starting validation process...")
    
    for batch_idx, test_data in enumerate(self.val_dataloader):
        print(f"[VALIDATE] Processing batch {batch_idx + 1}/{len(self.val_dataloader)}")
        
        # 生成序列
        print(f"[VALIDATE] Batch {batch_idx + 1}: Starting sequence generation...")
        test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(...)
        print(f"[VALIDATE] Batch {batch_idx + 1}: Sequence generation completed")
        
        # 计算奖励
        print(f"[VALIDATE] Batch {batch_idx + 1}: Starting reward computation...")
        result = self.val_reward_fn(test_batch, return_dict=True)
        print(f"[VALIDATE] Batch {batch_idx + 1}: Reward computation completed")
```

#### 4.4 训练流程增强 - Rollout
```python
# 生成阶段
rollout_n = self.config.actor_rollout_ref.rollout.n
num_prompts = gen_batch_output.batch["input_ids"].shape[0] // rollout_n
print(f"\n[Step {self.global_steps}] Starting rollout: {num_prompts} prompts × {rollout_n} samples = {total_sequences} sequences")

rollout_start = time.time()
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
rollout_time = time.time() - rollout_start
print(f"[Step {self.global_steps}] Rollout completed in {rollout_time:.2f}s ({total_sequences / rollout_time:.1f} seq/s)")
```

#### 4.5 vLLM 卸载用于奖励计算
```python
# 奖励计算前卸载vLLM
offload_for_reward = self.config.reward_model.get("offload_rollout_for_reward", False)
if offload_for_reward and hasattr(self, 'actor_rollout_wg'):
    print(f"[Step {self.global_steps}] Calling offload_for_reward...")
    self.actor_rollout_wg.offload_for_reward(offload_actor=True)
    print(f"[Step {self.global_steps}] offload_for_reward completed")

# 计算奖励
print(f"[Step {self.global_steps}] Computing rewards...")
reward_start = time.time()
reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
reward_time = time.time() - reward_start
print(f"[Step {self.global_steps}] Reward computation completed in {reward_time:.2f}s")

# 重新加载Actor模型用于计算old_log_prob
if offload_for_reward and hasattr(self, 'actor_rollout_wg'):
    print(f"[Step {self.global_steps}] Reloading Actor model...")
    self.actor_rollout_wg.reload_after_reward(reload_actor=True)
```

#### 4.6 Actor 更新前确保 vLLM 卸载
```python
if self.config.trainer.critic_warmup <= self.global_steps:
    # 确保vLLM已卸载
    if offload_for_reward and hasattr(self, 'actor_rollout_wg'):
        try:
            self.actor_rollout_wg.trainer_mode()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"trainer_mode call failed: {e}")
    
    # 更新Actor
    actor_start = time.time()
    actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_time = time.time() - actor_start
    print(f"[Step {self.global_steps}] Actor update completed in {actor_time:.2f}s")
```

### 5. `verl/trainer/ppo/reward.py`

**修改内容**:

#### 5.1 自定义奖励管理器自动导入
```python
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

# 自动导入自定义奖励管理器模块
reward_manager_name = config.reward_model.get("reward_manager", "naive")

# 如果reward_manager未注册，尝试从配置文件自动导入
if reward_manager_name not in REWARD_MANAGER_REGISTRY:
    reward_manager_config = config.get("reward_manager")
    if reward_manager_config is not None:
        reward_manager_module = reward_manager_config.get("module")
        if reward_manager_module:
            module_path = reward_manager_module.get("path")
            if module_path and os.path.exists(module_path):
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
```

#### 5.2 compute_reward 增强日志
```python
def compute_reward(data, reward_fn):
    print("[compute_reward] Starting reward_fn call...", flush=True)
    try:
        reward_result = reward_fn(data, return_dict=True)
        print(f"[compute_reward] reward_fn returned...", flush=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        print(f"[compute_reward] reward_tensor shape: {reward_tensor.shape}...", flush=True)
    except Exception as e:
        print(f"Error in reward_fn: {e}", flush=True)
        ...
    return reward_tensor, reward_extra_infos_dict
```

### 6. `verl/workers/fsdp_workers.py`

**修改内容**:

#### 6.1 update_actor 增强日志和GPU内存监控
```python
def update_actor(self, data: DataProto):
    logger.info(f"[update_actor] Starting...")
    log_gpu_memory_usage("update_actor start", logger=logger)
    
    torch.cuda.empty_cache()
    log_gpu_memory_usage("update_actor after cache clear", logger=logger)
    
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("update_actor after load model", logger=logger)
    
    # ... 原有代码 ...
    
    logger.info(f"[update_actor] update_policy completed")
    log_gpu_memory_usage("update_actor after update_policy", logger=logger)
```

#### 6.2 compute_log_prob 增强日志
```python
def compute_log_prob(self, data: DataProto):
    logger.info(f"[compute_log_prob] Starting, _is_actor={self._is_actor}...")
    log_gpu_memory_usage("compute_log_prob start", logger=logger)
    
    if self._is_offload_param:
        logger.info(f"[compute_log_prob] Loading FSDP model to GPU...")
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    
    # ... 原有代码 ...
    
    logger.info(f"[compute_log_prob] Forward pass completed")
    logger.info(f"[compute_log_prob] Completed successfully")
```

#### 6.3 新增 offload_for_reward 方法
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def offload_for_reward(self, offload_actor: bool = False):
    """
    在奖励计算前卸载vLLM以释放GPU内存
    
    注意: vLLM 0.8.5的sleep(level=2)有问题，所以我们直接销毁engine
    
    Args:
        offload_actor: 是否同时卸载Actor模型到CPU（默认False）
    """
    if self._is_rollout and hasattr(self, 'rollout') and self.config.rollout.free_cache_engine:
        # 销毁vLLM inference engine
        rollout = self.rollout
        if hasattr(rollout, 'inference_engine') and rollout.inference_engine is not None:
            # 存储配置用于后续重建
            self._vllm_destroyed = True
            
            # 删除inference engine
            del rollout.inference_engine
            rollout.inference_engine = None
            
            # 强制垃圾回收和CUDA缓存清理
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # 可选: 卸载Actor模型
    if offload_actor and self._is_actor:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
```

#### 6.4 新增 reload_after_reward 方法
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def reload_after_reward(self, reload_actor: bool = False):
    """
    奖励计算后重新加载模型
    
    注意: vLLM会在下一次rollout_mode()调用时自动重建
    
    Args:
        reload_actor: 是否重新加载Actor模型到GPU（默认False）
    """
    if reload_actor and self._is_actor:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    
    # vLLM会在下一次rollout_mode()调用时自动重建
```

### 7. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**修改内容**:

#### 7.1 日志级别调整
```python
# 原代码
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 修改为
logger.setLevel(os.getenv("VERL_ROLLOUT_LOGGING_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "INFO")))
```

#### 7.2 sleep_level 配置支持
```python
# 原代码
self.sleep_level = VLLM_SLEEP_LEVEL

# 修改为
self.sleep_level = config.get("sleep_level", VLLM_SLEEP_LEVEL)
# 如果free_cache_engine=True，强制使用sleep_level=2
if config.free_cache_engine and self.sleep_level < 2:
    logger.warning(f"[vLLM] free_cache_engine=True but sleep_level={self.sleep_level}, forcing sleep_level=2")
    self.sleep_level = 2
```

#### 7.3 保存 engine 创建参数用于重建
```python
# 在 __init__ 中保存参数
self._engine_create_params = {
    "model": model_path,
    "enable_sleep_mode": config.free_cache_engine,
    "tensor_parallel_size": tensor_parallel_size,
    # ... 其他参数
}

self.inference_engine = LLM(**self._engine_create_params)
```

#### 7.4 新增 _destroy_inference_engine 方法
```python
def _destroy_inference_engine(self):
    """
    销毁 inference engine 以释放 GPU 内存
    
    这是vLLM 0.8.x - 0.10.x中sleep(level=2)的替代方案
    """
    import gc
    import time
    import torch
    from vllm.distributed.parallel_state import destroy_model_parallel
    from vllm.device_allocator import cumem
    
    # 1. 销毁 model parallel state
    destroy_model_parallel()
    
    # 2. 删除 driver_worker
    if hasattr(self.inference_engine, 'llm_engine'):
        llm_engine = self.inference_engine.llm_engine
        if hasattr(llm_engine, 'model_executor'):
            executor = llm_engine.model_executor
            if hasattr(executor, 'driver_worker'):
                del executor.driver_worker
            del llm_engine.model_executor
    
    # 3. 释放 cumem allocator 内存
    # ... 详细实现见源码 ...
    
    # 4. 删除 inference engine
    del self.inference_engine
    self.inference_engine = None
    
    # 5. CUDA 清理
    for i in range(5):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(0.5)
```

#### 7.5 新增 _reset_vllm_allocator 方法
```python
def _reset_vllm_allocator(self):
    """
    重置 vLLM 的 cumem allocator 以允许创建新实例
    
    vLLM 0.8.x 使用 cumem allocator 跟踪使用情况，阻止每进程多个实例
    """
    from vllm.device_allocator import cumem
    
    # 重置 CuMemAllocator._instance
    if hasattr(cumem.CuMemAllocator, '_instance'):
        cumem.CuMemAllocator._instance = None
    
    # 重置 cumem._allocator
    if hasattr(cumem, '_allocator'):
        cumem._allocator = None
```

#### 7.6 新增 _patch_vllm_allocator_check 方法
```python
def _patch_vllm_allocator_check(self):
    """
    Monkey patch vLLM 的 allocator 检查以允许实例重建
    
    让 get_current_usage() 始终返回 0，绕过 sleep mode 检查
    """
    from vllm.device_allocator import cumem
    
    def patched_get_current_usage(self):
        return 0
    
    cumem.CuMemAllocator.get_current_usage = patched_get_current_usage
```

#### 7.7 新增 _recreate_inference_engine 方法
```python
def _recreate_inference_engine(self):
    """重建 inference engine"""
    import gc
    import time
    import torch
    
    # CUDA 清理
    for i in range(5):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(0.5)
    
    # 重置 allocator
    self._reset_vllm_allocator()
    
    # Monkey patch
    self._patch_vllm_allocator_check()
    
    # 使用保存的参数重建
    self.inference_engine = LLM(**self._engine_create_params)
```

#### 7.8 修改 generate_sequences 方法
```python
def generate_sequences(self, prompts: DataProto, **kwargs):
    # 添加日志
    n_samples = self.sampling_params.n if hasattr(self.sampling_params, 'n') else 1
    total_sequences = len(vllm_inputs) * n_samples
    logger.info(f"[vLLM Rollout] Starting generation: {len(vllm_inputs)} prompts × {n_samples} samples = {total_sequences} sequences")
    
    gen_start_time = time.time()
    outputs = self.inference_engine.generate(...)
    gen_time = time.time() - gen_start_time
    
    tokens_generated = sum(len(out.outputs[0].token_ids) for out in outputs)
    logger.info(f"[vLLM Rollout] Generation completed in {gen_time:.2f}s | {tokens_generated} tokens | {tokens_per_sec:.1f} tokens/s")
```

#### 7.9 修改 resume 方法
```python
async def resume(self, tags: Optional[list[str]] = None):
    """Resume from sleep mode or recreate if destroyed"""
    
    # 检查是否需要重建
    if self.inference_engine is None:
        logger.info("[vLLM resume] inference_engine is None, recreating...")
        self._recreate_inference_engine()
        return
    
    # 原有代码...
    if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
        self.inference_engine.wake_up(tags=tags)
    else:
        self.inference_engine.wake_up()
```

#### 7.10 修改 release 方法
```python
async def release(self):
    """
    释放 GPU 内存
    
    由于 vLLM 的 sleep(level=2) 无法正确释放内存，我们直接销毁 engine
    """
    if not self.config.free_cache_engine:
        return
    
    if self.inference_engine is not None:
        self.inference_engine.reset_prefix_cache()
    
    # 销毁 inference engine
    self._destroy_inference_engine()
```

---

## 三、配置参数说明

### 新增配置参数 (YAML)

```yaml
reward_model:
  # ===== CoT Rewriter 相关配置 =====
  
  # 奖励管理器名称
  reward_manager: "cot_rewriter"
  
  # Student 模型路径
  student_model_path: "./models/Qwen3-1.7B"
  
  # FSF Judge 模型路径（可选）
  fsf_model_path: "./models/FSF-Judge"
  
  # 逻辑奖励权重
  lambda_logic: 0.5
  
  # 是否使用逻辑奖励
  use_logic_reward: false
  
  # min log prob 权重
  alpha_cog: 0.01
  
  # 运行设备
  device: "cuda:3"
  
  # 是否启用CPU卸载（节省GPU内存）
  enable_offload: true
  
  # 卸载目标设备
  offload_device: "cpu"
  
  # 是否使用多GPU
  use_multi_gpu: false
  
  # 多GPU设备列表
  student_devices: ["cuda:0", "cuda:1", "cuda:2"]
  
  # 是否使用vLLM
  use_vllm: false
  
  # vLLM tensor parallel size
  vllm_tensor_parallel_size: 1
  
  # vLLM GPU内存利用率
  vllm_gpu_memory_utilization: 0.3
  
  # 奖励微批次大小（防止OOM）
  reward_micro_batch_size: 64
  
  # GPU批次大小
  gpu_batch_size: 8
  
  # ===== TaskRunner 相关配置 =====
  
  # TaskRunner GPU资源数（用于奖励计算）
  task_runner_num_gpus: 1
  
  # 是否在奖励计算前卸载vLLM
  offload_rollout_for_reward: true

data:
  # 使用自定义数据集类
  dataset_cls: "verl.utils.dataset.rl_dataset_jsonl:RLHFDatasetJSONL"
  
  # 字段映射
  prompt_key: "problem"
  
  # 自定义字段映射（用于reward function）
  field_mapping:
    question_key: "question"
    original_cot_key: "thought"
    ground_truth_key: "solution"

actor_rollout_ref:
  rollout:
    # vLLM sleep level (0: 不sleep, 1: 释放KV cache, 2: 释放模型权重)
    sleep_level: 2
    
    # 是否启用cache engine释放
    free_cache_engine: true
```

---

## 四、迁移到 verl 0.5.0 的步骤

### 步骤 1: 复制新增文件
```bash
# 创建目录结构
mkdir -p verl-0.5.0/verl/experimental/reward/cot_rewriter
mkdir -p verl-0.5.0/verl/utils/dataset

# 复制新增文件
cp verl/verl/experimental/reward/cot_rewriter/*.py verl-0.5.0/verl/experimental/reward/cot_rewriter/
cp verl/verl/utils/dataset/rl_dataset_jsonl.py verl-0.5.0/verl/utils/dataset/
```

### 步骤 2: 修改 `verl/experimental/reward/__init__.py`
添加 CoT Rewriter 模块的导入和导出。

### 步骤 3: 修改 `verl/utils/dataset/__init__.py`
添加 `RLHFDatasetJSONL` 的导入和导出。

### 步骤 4: 修改 `verl/trainer/main_ppo.py`
- 添加 Ray Dashboard 禁用
- 修改 Ray 初始化
- 添加 TaskRunner GPU 配置
- 修改 ResourcePoolManager 创建

### 步骤 5: 修改 `verl/trainer/ppo/ray_trainer.py`
- 添加 `task_runner_num_gpus` 字段到 ResourcePoolManager
- 修改 GPU 资源检查逻辑
- 添加训练流程中的日志和计时
- 添加 vLLM 卸载/重载逻辑

### 步骤 6: 修改 `verl/trainer/ppo/reward.py`
- 添加自定义奖励管理器自动导入
- 增强 compute_reward 日志

### 步骤 7: 修改 `verl/workers/fsdp_workers.py`
- 增强 update_actor 和 compute_log_prob 的日志
- 添加 `offload_for_reward` 方法
- 添加 `reload_after_reward` 方法

### 步骤 8: 修改 `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- 调整日志级别
- 添加 sleep_level 配置支持
- 保存 engine 创建参数
- 添加 `_destroy_inference_engine` 方法
- 添加 `_reset_vllm_allocator` 方法
- 添加 `_patch_vllm_allocator_check` 方法
- 添加 `_recreate_inference_engine` 方法
- 修改 `generate_sequences` 添加日志
- 修改 `resume` 支持 engine 重建
- 修改 `release` 使用销毁而不是 sleep

### 步骤 9: 应用 diff 文件
```bash
cd /Users/moonshot/Desktop/cot_rewrite_grpo
# 生成针对 verl 0.5.0 的 patch
diff -ruN verl-0.5.0 verl > verl_to_0.5.0.patch

# 或者在 verl 0.5.0 目录应用修改
patch -p1 < verl_modifications.diff
```

---

## 五、注意事项

### 1. vLLM 版本兼容性
- vLLM 0.8.x - 0.10.x 的 `sleep(level=2)` 无法正确释放 GPU 内存
- 修改使用销毁+重建的方式作为替代方案
- 需要 monkey patch 绕过 vLLM 的 allocator 检查

### 2. GPU 内存管理
- `offload_rollout_for_reward` 选项允许在奖励计算前释放 vLLM 内存
- TaskRunner 可以配置 GPU 资源用于奖励计算
- 注意 GPU 资源的时间共享机制

### 3. 数据格式
- JSONL 格式需要 `datasets` 库支持
- 纯文本 prompt 会自动转换为 chat 格式
- 确保数据字段与 `field_mapping` 配置匹配

### 4. Ray 配置
- 必须设置 `RAY_DISABLE_DASHBOARD_AGENT=1` 避免 OpenTelemetry 冲突
- TaskRunner 的 GPU 资源需要从总 GPU 数中扣除用于资源检查

### 5. 日志级别
- 设置 `VERL_ROLLOUT_LOGGING_LEVEL=INFO` 查看 vLLM 详细日志
- 设置 `VERL_LOGGING_LEVEL=INFO` 查看其他模块日志

---

## 六、调试信息

修改后的代码添加了详细的日志输出，包括：
- `[REWARD]` - 奖励计算相关日志
- `[VALIDATE]` - 验证流程日志
- `[Step N]` - 训练步骤日志
- `[update_actor]` - Actor 更新日志
- `[compute_log_prob]` - Log prob 计算日志
- `[offload_for_reward]` - vLLM 卸载日志
- `[vLLM]` - vLLM rollout 相关日志

所有日志都会输出到控制台，便于排查问题。
