# 格式问题调查报告

**日期**: 2026-02-10
**问题**: 训练时rollout经常出现整组样本不满足format（缺少 `<rewrite_reasoning>` 标签）

## 问题描述

### 训练时的现象
```
[REWARD] Sample 0: No valid <rewrite_reasoning> tags found, will use original text for cognitive reward and apply format penalty
[REWARD] Sample 1: No valid <rewrite_reasoning> tags found, will use original text for cognitive reward and apply format penalty
...
[REWARD] Sample 7: No valid <rewrite_reasoning> tags found, will use original text for cognitive reward and apply format penalty
```

- **训练时**: 8/8样本格式错误（0%正确率）
- **离线测试**: 基本满足格式要求

## 调查过程

### 1. 数据集验证

首先确认训练数据中包含完整的格式指令：

```python
# 数据中的problem字段包含完整的格式指令
problem = """
...
### Output Format
Put your rewritten reasoning process inside <rewrite_reasoning> tags. Do not output the final answer again outside the tags, just the reasoning.

<rewrite_reasoning>
[Your rewritten reasoning process here]
</rewrite_reasoning>
"""
```

**结论**: 数据集格式正确，包含完整的 `<rewrite_reasoning>` 标签指令。

### 2. Prompt长度验证

```python
# Prompt token length: 2652
# Max prompt length in config: 5500
# 结论: Prompt没有被截断
```

### 3. 离线测试对比

创建了测试脚本 `scripts/grpo/test_format_comparison.py` 进行对比测试：

| 测试方式 | temperature | 格式正确率 |
|---------|-------------|-----------|
| 字符串prompt | 1.0 | 75% (6/8) |
| prompt_token_ids | 1.0 | 50% (4/8) |
| prompt_token_ids | 0.1 | 75% (3/4) |

**关键发现**:
1. 使用 `prompt_token_ids` 方式会降低格式正确率
2. temperature越低，格式正确率越高
3. 即使使用字符串prompt，temperature=1.0时也只有75%正确率

### 4. 与FSDP的关系分析

#### FSDP-vLLM权重同步流程

训练时使用Hybrid Engine架构：
- FSDP Actor: 用于梯度计算和参数更新
- vLLM Engine: 用于高效生成

配置：
```yaml
actor_rollout_ref:
  rollout:
    load_format: dummy_dtensor  # vLLM使用随机权重初始化，然后从FSDP同步
```

#### 权重同步代码

```python
# verl/verl/workers/sharding_manager/fsdp_vllm.py:122
self.base_sync_done: bool = "dummy" not in load_format
```

当 `load_format="dummy_dtensor"` 时：
- `base_sync_done=False`
- vLLM初始化时使用**随机权重**
- 第一次进入 `sharding_manager.__enter__()` 时从FSDP同步权重

#### 生成流程

```python
# verl/verl/workers/fsdp_workers.py:741-750
def generate_sequences(self, prompts: DataProto):
    with self.rollout_sharding_manager:  # 这里触发权重同步
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
```

## 可能的问题原因

### 1. FSDP权重同步时序问题

第一次rollout时，FSDP可能还在初始化或权重未完全加载，导致同步到vLLM的权重不正确。

### 2. 随机种子问题

多GPU环境下（8x80G），vLLM的随机种子可能没有正确设置，导致生成结果不一致。

### 3. prompt_token_ids处理差异

vLLM处理 `prompt_token_ids` 和字符串prompt的方式可能存在差异：
- 字符串prompt: vLLM内部进行tokenization
- prompt_token_ids: 直接使用token ids，可能跳过某些预处理

### 4. Temperature影响

temperature=1.0时随机性较高，模型可能"忘记"格式指令。

## 相关代码位置

| 文件 | 功能 |
|------|------|
| `verl/verl/workers/sharding_manager/fsdp_vllm.py` | FSDP-vLLM权重同步 |
| `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | vLLM rollout实现 |
| `verl/verl/workers/fsdp_workers.py` | FSDP worker，generate_sequences入口 |
| `verl/verl/utils/dataset/rl_dataset_jsonl.py` | 数据集处理，left_pad配置 |
| `scripts/grpo/test_format_comparison.py` | 格式测试脚本 |

## 建议的解决方法

### 1. 降低rollout temperature
```yaml
actor_rollout_ref:
  rollout:
    temperature: 0.7  # 从1.0降低到0.7
```

### 2. 启用INFO级别日志查看权重同步
```bash
export VERL_LOGGING_LEVEL=INFO
```

### 3. 检查是否有更好的load_format选项
- `safetensors`: vLLM直接加载checkpoint（需要更多显存）
- `dummy_hf`: 使用HF tokenizer而不是vLLM的

### 4. 检查随机种子设置
确保训练时的随机种子正确设置，特别是多GPU环境。

## 待验证

1. [ ] 设置VERL_LOGGING_LEVEL=INFO后运行训练，查看权重同步日志
2. [ ] 尝试降低temperature到0.7，观察格式正确率变化
3. [ ] 检查多GPU环境下的随机种子设置
4. [ ] 对比使用不同load_format的效果

## 测试脚本

- `scripts/grpo/test_model_format.py`: 原始离线测试脚本
- `scripts/grpo/test_rollout_format.py`: 模拟训练rollout测试
- `scripts/grpo/test_format_comparison.py`: 对比不同prompt方式
