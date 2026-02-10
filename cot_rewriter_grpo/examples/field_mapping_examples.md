# 字段映射配置示例

VERL训练支持自定义字段名，通过 `field_mapping` 配置可以适配不同的数据格式。

## 配置方式

在配置文件的 `reward_model.custom_reward_function.reward_kwargs.field_mapping` 中设置：

```yaml
reward_model:
  custom_reward_function:
    reward_kwargs:
      field_mapping:
        prompt_key: "prompt"              # 输入prompt字段名
        reward_model_key: "reward_model"  # reward信息包裹字段名 (设为null表示平铺)
        question_key: "question"          # 问题字段名
        original_cot_key: "original_cot"  # 原始CoT字段名
        ground_truth_key: "ground_truth"  # 答案字段名
```

## 示例1: 默认格式（已转换的数据）

数据格式：
```json
{
  "prompt": "Please rewrite...",
  "reward_model": {
    "question": "What is 2+2?",
    "original_cot": "Let me think...",
    "ground_truth": "4"
  }
}
```

配置：
```yaml
field_mapping: {}  # 使用默认值
```

## 示例2: 原始数据格式（使用 problem/thought/solution）

数据格式：
```json
{
  "problem": "Please rewrite...",
  "question": "What is 2+2?",
  "thought": "Let me think...",
  "solution": "4"
}
```

配置：
```yaml
field_mapping:
  prompt_key: "problem"
  question_key: "question"
  original_cot_key: "thought"
  ground_truth_key: "solution"
```

## 示例3: 平铺结构（无 reward_model 包裹）

数据格式：
```json
{
  "prompt": "Please rewrite...",
  "question": "What is 2+2?",
  "cot": "Let me think...",
  "answer": "4"
}
```

配置：
```yaml
field_mapping:
  reward_model_key: null      # null表示不使用reward_model包裹
  question_key: "question"
  original_cot_key: "cot"
  ground_truth_key: "answer"
```

## 示例4: 完全自定义字段

数据格式：
```json
{
  "input_text": "Please rewrite...",
  "metadata": {
    "q": "What is 2+2?",
    "reasoning": "Let me think...",
    "target": "4"
  }
}
```

配置：
```yaml
field_mapping:
  prompt_key: "input_text"
  reward_model_key: "metadata"
  question_key: "q"
  original_cot_key: "reasoning"
  ground_truth_key: "target"
```

## 完整配置示例

```yaml
# examples/custom_field_mapping.yaml

reward_model:
  enable: true
  custom_reward_function:
    path: src/reward/combined_reward.py
    name: compute_cot_rewriter_reward
    reward_kwargs:
      student_model_path: /path/to/student
      fsf_model_path: /path/to/fsf_judge
      device: cuda:5
      lambda_logic: 0.5
      use_logic_reward: true
      # 字段映射配置
      field_mapping:
        prompt_key: "problem"
        reward_model_key: null
        question_key: "question"
        original_cot_key: "thought"
        ground_truth_key: "solution"
```

## 注意事项

1. `prompt_key` 只影响数据处理，不影响模型输入（模型输入始终是 `prompt`）
2. `reward_model_key` 设为 `null` 时表示字段平铺在顶层
3. 如果某个字段映射后找不到，会自动尝试默认字段名作为fallback
4. 字段名必须是字符串，不能使用数字或特殊字符
