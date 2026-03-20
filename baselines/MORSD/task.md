# 任务
帮我复现论文：/home/luoys/papers/MORSD
按照论文中的要求，实现Diffculty selection的数据筛选流程, 不需要Diversity Selection了！
输入是/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl的格式: 
{
    'question': '...',
    'reasoning': [
        {
            'thought': '...',
            'solution': '...'
        },
        ...
    ]
}
每个question，对应3个不同的reasoning，把thought视作Rationale，solution视作answer
拼接response的方式为:
messages = [
    {
        'role': 'user',
        'content': question
    },
    {
        'role': 'assistant',
        'content': f"<think>\n{rationale}\n</think>\n\n{answer}"
    },
    ...
]
输出格式为：
/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train_morsd.jsonl,
{
    'question': '...',
    'reasoning': [
        {
            'thought': '...',
            'solution': '...',
            'rds': [
                {
                    "model_name": "Llama-3.2-3B-Instruct",（自动从model_path中提取）
                    "model_path": "/data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct",
                    "ppl_with_rationale": xxx,   # PPL(answer | rationale, question)                                                                                                                                    
                    "ppl_without_rationale": xxx, # PPL(answer | question) 
                    "rd": xxx, # rd = ppl_with_rationale / ppl_without_rationale
                }
            ]

        },
        ...
    ]
}
# 实现细节
1. 实现rd计算脚本：/home/luoys/cdad/baselines/MORSD/rd_calc.py
2. 实现数据集构建脚本：/home/luoys/cdad/baselines/MORSD/build_dataset_morsd.py, 参考/home/luoys/cdad/reasoning_graph/build_sharegpt_dataset.py，构建符合LlamaFactory格式的数据集
3. 环境：基于/home/luoys/cdad/.venv运行，计算rd可以用vllm
4. 断点续传，不会中断后不会重复计算
5. 多卡DP，尽量不要TP
6. 基于vllm 0.8.5.post1
7. 用bash命令启动，帮我写好启动脚本：
/home/luoys/cdad/baselines/MORSD/scripts/run.sh
针对以下3个模型，帮我写好启动脚本：
- /data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct
- /data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct
- /data/modelscope_cache/models/Qwen/Qwen2.5-7B-Instruct
