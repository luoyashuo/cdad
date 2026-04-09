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
对每个样本，按照/home/luoys/cdad/cot_rewriter_grpo/verl/verl/experimental/reward/cot_rewriter/sglang_fsf_reward.py的方式，计算fsf。

**同时计算 original CoT 的 FSF + DOT 图**（用于后续 GED 计算和分布分析）。

输出格式：
```json
{
    "question": question,
    "original_cot": original_cot,
    "solution": solution,
    "original_fsf": 0.25,
    "original_fsf_model_output": "<DOT图原始输出>",
    "rewrite_cots": [
        {
            "rewrite_cot": rewrite_cot,
            "is_correct": true,
            "fsf": 0.1,
            "fsf_model_output": "<DOT图原始输出>",
            "ged": 3.0
        }
    ],
    "fsf_tag": true
}
```

- `original_fsf` / `original_fsf_model_output`：原始 CoT 的 FSF 值和 DOT 图输出
- `fsf` / `fsf_model_output`：每个 rewrite_cot 的 FSF 值和 DOT 图输出
- `ged`：原始 CoT 与 rewrite_cot 的 DOT 图编辑距离（Graph Edit Distance），CPU 并行计算
- `fsf_tag`：该样本是否存在至少一个 fsf > 0 的 rewrite

**断点续传与 original FSF 补算（topup）**：

如果 `fsf_tag_output.jsonl` 中存在缺少 `original_fsf_model_output` 的记录（由旧版 pipeline 产生），重新运行时会自动检测并用 FSF judge 补算，然后重新计算 GED。

```bash
# 续跑：自动检测缺失的 original_fsf 并补算
bash inference/run_pipeline.sh --config inference/config_example.yaml --resume
```

5. 分布分析（Distribution Analysis）:
Step 4 完成后自动运行，也可单独触发（不需要 GPU）。

分析内容：
- **FSF 分布**：original CoT 和 rewrite CoT 的 FSF 均值/标准差/分位数 + 直方图
- **失败节点数分布**：解析 DOT 图（multiprocessing 并行），统计 `failed_nodes` 和 `total_nodes` 的分布
- **改写效果对比**：每条记录 `original_fsf - min(rewrite_fsf)`，统计改善/持平/变差比例

报告保存到 `{output_dir}/analysis/fsf_distributions.txt`，同时打印到日志。

```bash
# 仅重跑分布分析（CPU only）
bash inference/run_pipeline.sh --config inference/config_example.yaml --step analyze
```

6. 根据rollout的结果，重新构建sharegpt格式的数据集，每个question选择1个rewrite_cot，支持多种筛选条件：
- 是否过滤fsf>0的样本
- 是否过滤is_correct=False的样本
- 是否随机筛选
- 是否优先筛选fsf最大的样本
- 是否优先筛选is_correct=True的样本
sharegpt构建的方式为
```json
{
    "conversations": [
        {"role": "user", "value": question},
        {"role": "assistant", "value": "<think>\n{rewrite_cot}\n</think>\n\n{solution}"}
    ]
}
```