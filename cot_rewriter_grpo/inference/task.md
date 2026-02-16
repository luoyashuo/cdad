# 任务
我需要调用改写模型对原有的数据集进行改造

# 原始数据格式
参考：```head -n 1 /data/luoys/data/cdad/omnithought_rv_optimal/train.jsonl```
内容如下：
```
{
  "conversations": [
    {
      "role": "user",
      "content": "xxx"
    },
    {
      "role": "assistant",
      "content": "xxx"
    }
  ]
}
```
# 改写步骤
0. 通过yaml文件，读取本次运行的配置，包括rewrite_model_path, student_model_path, fsf_judge_model_path，数据路径，输出路径，n次调用改写模型，similarity_penalty阈值，ppl_relative_improvement阈值，fsf阈值等必要信息
1. 转换rewrite_model的模型格式：
- 改写模型的格式如：/data/luoys/checkpoints/cdad/cot_rewriter_grpo/llama3_8b_grpo_8x80g_v8/global_step_100
- 你需要在原地转化为huggingface的格式，例如/data/luoys/checkpoints/cdad/cot_rewriter_grpo/llama3_8b_grpo_8x80g_v8/global_step_100的转换结果为/data/luoys/checkpoints/cdad/cot_rewriter_grpo/llama3_8b_grpo_8x80g_v8/global_step_100/hf
需要包含模型权重，tokenizer等文件
2. 读取，解析原始数据集
- 参考/data/luoys/projects/cdad/reasoning_graph/build_cot_rewrite_from_sharegpt.py的做法，
- 提取question和thought：
```
# question 来自 conversation[0]
question = conversations[0].get("content", "")

# think 来自 conversation[1] 的 content
assistant_content = conversations[1].get("content", "")
try:
think = extract_think_from_rv(assistant_content)
except IndexError:
continue
```
3. 构建改写模型的请求prompt
```
def build_prompt(question: str, thought: str) -> Tuple[str, str]:
    """
    构建 system_prompt 和 user_prompt

    Args:
        question: 问题文本
        thought: 推理过程

    Returns:
        (system_prompt, user_prompt) 元组
    """
    system_prompt =  """\
You are an expert in math and code. The 'Original Reasoning Process' is a verified and valid derivation. Using it as a reference, rewrite the reasoning with a better logical structure and a more clear delivery.

Follow these guidelines:
1. **Logical Optimization**: Keep logic direct and coherent; eliminate redundancy without losing necessary complexity.
2. **Step-by-Step Delivery**: Present the reasoning in a clear, sequential, step-by-step manner.
3. **Avoid Literal Repetition**: Rephrase the content in your own words rather than simply duplicating the text of the original process.

Directly output the rewritten reasoning strictly enclosed within <rewrite_reasoning> </rewrite_reasoning> tags. Output nothing else."""
    user_prompt = f"""\
## Problem
{question}

## Original Reasoning Process
{thought}"""

    return system_prompt, user_prompt

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
```
4. 调用改写模型, 每个输入调用n次, 提取n个结果（有些结果可能后续处理会失败）
output = rewrite_model.generate(
    messages,
    max_new_tokens=15000,
    temperature=0.6,
    top_p=0.95,
    presence_penalty=1.2,
    frequency_penalty=1.2,
)，这一步的结果保存到{output_dir}/ckpt/ckpt_1_rewrite.jsonl中；
支持断点续传，每次重启，从上次中断的地方继续（如果一个prompt没有生成够n次，则从上一个样本开始）
5. 从ckpt中加载、提取改写结果
```
rewrite_reasoning = output.split("<rewrite_reasoning>")[1].split("</rewrite_reasoning>")[0].strip()
```
6. 计算similarity_penalty，如果similarity_penalty大于0.9，则抛弃这条改写结果. 采用多进程处理加速
7. 加载student_model, 计算ppl_relative_improvement的值(逻辑参考/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/sglang_student_reward.py)，低于一个设定的阈值，则抛弃。 需要输出ppl_relative_improvement分布，方便参考. 本次过滤的结果保存到{output_dir}/ckpt/ckpt_2_ppl_filter.jsonl中. 本功能需要支持断点续传
8. 计算r_logic：计算逻辑和训练的逻辑相同(/data/luoys/projects/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/sglang_fsf_reward.py)，如果fsf>0，则抛弃该结果. 本次过滤的结果保存到{output_dir}/ckpt/ckpt_3_fsf_filter.jsonl中, 本功能需要支持断点续传
9. 选择ppl_relative_improvement最高，fsf最低，similarity_penalty最低的结果作为改写的结果(按照顺序过滤);多进程加速
10. 用指定的改写结果替换原始的thought，拼接新的数据集，并且标记哪些数据使用的是改写结果，如果上述做法没有符合要求的结果，
    则直接用原始的thought拼接新的数据集，并且标记哪些数据使用的是原始的thought
    拼接方式为：
    "conversations" = {
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": f"<think>\n{rewrite_cot.strip()}\n</think>\n\n{answer.strip()}"
        },
    }
    多进程加速

# 文件实现
实现在/data/luoys/projects/cdad/cot_rewriter_grpo/inference/rewrite_dataset.py路径下
