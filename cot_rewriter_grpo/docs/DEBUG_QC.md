# 任务
训练任务: bash scripts/grpo/start_qc.sh
运行环境(uv)：
```
source /data/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是8*A100-80G
4. 你不能修改数据的长度， response长度
# LOG
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/llama3_8b_grpo_8x80g.log
# 问题
1. 我希望计算R_cog的时候，增加一个正确性的检测（在配置中开启），用一个模型判断rewrite_cot和solution相比较，是否有逻辑错误，实现方式参考fsf_judge_model，在Student Model计算结束，释放显存后，再加载该judge Model，判断结束后再释放显存, 如果判断结果为rewrite_cot和solution有正确性问题，则强制把R_cog变成一个配置中预设的值（ppl_delta模式下为0）；
判断的prompt类似：
```
Determine if the [Reasoning Process] leads to the conclusion in the [Solution].

## Question
{question}

## Reasoning Process
{rewrite_cot}

## Solution
{solution}

## Output
Output <judge>Yes</judge> if the reasoning results in the given solution; otherwise, <judge>No</judge>.
```
2. 默认 correct judge model使用/data/luoys/modelscope_cache/models/Qwen/Qwen3-4B