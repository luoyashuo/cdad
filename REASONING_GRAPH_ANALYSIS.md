# Reasoning Graph Analysis Implementation

## 概述

本项目实现了论文《What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT》中提出的**Reasoning Graph**和**Failed-Step Fraction (FSF)**分析方法。

## 核心概念

### 1. Reasoning Graph (推理图)
将链式思维(CoT)转换为有向图结构，其中：
- 节点表示推理步骤
- 边表示步骤间的逻辑关系
- 红色节点表示失败的尝试
- 蓝色节点表示成功的步骤
- 虚线边表示回溯操作

### 2. Failed-Step Fraction (FSF)
定义：失败/放弃的步骤占总步骤的比例
```
FSF = (失败步骤数) / (总步骤数)
```

根据论文发现：
- **FSF与推理准确性呈负相关**
- **FSF是比CoT长度和回顾比率更强的预测指标**
- **较低的FSF表示更有效的推理过程**

## 实现功能

### 1. 推理步骤提取
- 基于模式的步骤分割
- `<think>`标签内容提取
- 失败尝试检测（关键词模式）
- 回顾行为识别

### 2. 图结构构建
- 构建有向图表示推理流程
- 识别步骤间的顺序关系
- 标记回溯和分支
- 支持深度层次结构

### 3. 指标计算
- 总步骤数
- 失败步骤数
- 失败步骤分数（FSF）
- 回顾比率
- 最大深度
- 分支数量

### 4. 可视化
- 使用Graphviz生成图形
- 颜色编码：红色=失败，蓝色=成功
- 形状编码：矩形=回顾，椭圆=前进
- 虚线表示回溯边

## 使用方法

```python
from reasoning_graph_analysis import analyze_dataset

# 分析数据集
metrics = analyze_dataset(
    input_file="/path/to/omnithought_dataset.jsonl",
    output_dir="./plots/reasoning_graphs",
    max_samples=1000  # 处理的样本数
)
```

## 输出结果

### 1. 推理图可视化
- 每个样本生成一个PNG图像
- 显示步骤间的逻辑关系
- 突出显示失败尝试

### 2. 指标统计
- FSF分布直方图
- 步骤数分布
- FSF与步骤数相关性

### 3. 分析报告
- 汇总JSON文件
- 统计摘要
- 正确性相关性

## 扩展功能

### 1. 正确性检查
虽然简化实现，但支持：
- 最终答案匹配检查
- 置信度指标分析
- 不确定性检测

### 2. 难度分层
根据论文方法，支持按难度分析：
- HARP数据集的难度等级（1-6）
- GPQA-Diamond的难度分层
- 相关性强度变化

### 3. 模型比较
支持多模型分析：
- Claude 3.7 Sonnet
- Deepseek R1系列
- Qwen系列
- GPT系列

## 关键发现

基于目前处理的结果显示：
- FSF范围：0.0 - 1.0
- 平均FSF：待计算
- FSF与推理长度的相关性：待分析
- 失败步骤对后续推理的影响明显

## 下一步计划

1. **大规模分析**：处理完整Omnithought数据集
2. **正确性标注**：获取地面真值进行有效性分析
3. **干预实验**：验证移除失败步骤的效果
4. **时序分析**：研究FSF在推理过程中的变化
5. **模型适应性**：测试不同LRM的表现模式

## 相关文件

- `reasoning_graph_analysis.py`：主要分析脚本
- `plots/reasoning_graphs/`：生成的图表目录
- `metrics_summary.json`：汇总结果

## 参考文献

Feng, Y., Kempe, J., Zhang, C., Jain, P., & Hartshorn, A. (2025). What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT. arXiv:2509.19284