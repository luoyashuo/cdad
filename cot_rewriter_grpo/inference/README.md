# 加载&改写流程管线
1. 转换为HF格式：读取模型路径(例如/data/cdad/checkpoints/cot_rewriter/cot_rewriter_grpo/qwen2_5_7b_grpo_8x40g_v2/global_step_30)，自动检测，如果是lora格式需要帮我merge，merge方式参考verl.model_merger
2. Rollout：读入sharegpt格式的数据，例如/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl格式
提取think的方式是：
question = conversations[0]['content'].strip()
original_cot = conversations[1]['content'].split("<think>")[-1].split("</think>")[0].strip()
solution = conversations[1]['content'].split("</think>")[-1].strip()
然后参考/home/luoys/cdad/reasoning_graph/build_cot_rewrite_from_deepmath.py build_prompt的方式进行rollout，每个样本可以rollout多次，每次rollout的结果保存到rewrite_cots中，这个地方需要一方面支持断点续传（自动检测），另一方面支持多卡数据并行rollout，还有tqdm显示总进度
输出保存为到jsonl，格式为：
{
    "question": question,
    "original_cot": original_cot,
    "solution": solution,
    "rewrite_cots": [
        {"rewrite_cot": rewrite_cot},
        {"rewrite_cot": rewrite_cot},
    ],
}
3. Correct Tag:
对rollout的结果，按照/home/luoys/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/sglang_correct_judge.py的方式,判断每个rewrite_cot是否正确，保存格式为jsonl：
{
    "question": question,
    "original_cot": original_cot,
    "solution": solution,
    "rewrite_cots": [
        {"rewrite_cot": rewrite_cot,'is_correct':true or false},
        {"rewrite_cot": rewrite_cot,'is_correct':true or false},
    ],
}
4. FSF TAG:
对每个样本，按照/home/luoys/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/sglang_fsf_reward.py的方式，计算fsf，判断fsf是否大于0，输出格式
{
    "question": question,
    "original_cot": original_cot,
    "solution": solution,
    "rewrite_cots": [
        {"rewrite_cot": rewrite_cot,'is_correct':true or false, 'fsf':fsf, 'fsf_model_output':fsf_model_output},
        {"rewrite_cot": rewrite_cot,'is_correct':true or false, 'fsf':fsf, 'fsf_model_output':fsf_model_output},
    ],
    "fsf_tag": true or false,
}
5. 根据rollout的结果，重新构建sharegpt格式的数据集，每个question选择1个rewrite_cot，支持多种筛选条件：
- 是否过滤fsf>0的样本
- 是否过滤is_correct=False的样本
- 是否随机筛选
- 是否优先筛选fsf最大的样本
- 是否优先筛选is_correct=True的样本
sharegpt构建的方式为
{
    'conversations': [
        {'role': 'user', 'value': question},
        {'role': 'assistant', 'value': f"<think>\n{rewrite_cot.strip()}\n</think>\n\n{solution.strip()}"},
    ]
}