
# 任务
帮我基于verl框架，实现以下项目：

构建一个基于 **GRPO (Group Relative Policy Optimization)** 算法的训练系统，目标是训练一个 **CoT Rewriter (改写模型)**。
- **Actor Model**: Qwen-2.5-7b-instruct (用于改写 CoT)。
- **Student Model**: 一个被冻结的 Reference Model (用于评估改写后的 CoT 对 Answer 预测概率的增益)。
- **硬件环境**: 单机 4 * A100 (80G)。

# Reward Logic Definition (Critical)
你需要实现一个复杂的组合奖励函数 $R = R_{cog} + \lambda R_{logic}$：
1.  **认知奖励 ($R_{cog}$)**: 
    -   需要加载一个冻结的 `Student_Model`。
    -   计算公式: $R_{cog} = \log P_{student}(\text{Answer} | \text{Question}, \text{Rewrite\_CoT})$。
    -   *目的*: 激励 Rewrite_CoT 能让 Student Model 以更高的置信度输出正确答案。
2.  **逻辑奖励 ($R_{logic}$)**:
    -   利用 **LLM-as-a-Judge**  分析 CoT 的 "失败分支占比" (FSF)。
    -   计算公式: 若 $\text{FSF}(\text{Rewrite\_CoT}) < \text{FSF}(\text{Original\_CoT})$ 则为 1，否则为 0。
# 要求
1. 默认参与GRPO的模型是Qwen3-8b
2. 提供输入输出的数据示例，展示数据集格式
3. 计算reward的方式是，把student_model和fsf_judge_model单独部署在一个vllm服务中，通过访问vllm服务计算奖励
4. 支持手动开启PEFT，因为可能只有2*A100-80G
5. 项目的配置风格如下：
examples/xxxx.yaml : 训练的配置文件
scripts/run_xxxx.sh : 启动训练的脚本（同时启动vllm服务，ready后启动verl训练）
src/xxx :项目的代码逻辑
6. 运行环境：
torch2.6.0+cuda12.4
7. 使用uv配置运行环境
8. 重复运行实验，会把ckpt保存在 output_dir/version_{timestamp}/xxx路径下，避免覆盖