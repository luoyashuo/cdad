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
system\n\nCutting Knowledge Date: December 2023\
我看目前rollout_data中
/data/luoys/checkpoints/cdad/rollout_data/llama3_8b_grpo_8x80g_top2000/45.jsonl
input中包含了除了数据集（/data/luoys/data/cdad/omnithought_rv_optimal_cot_rewrite_top2000/train.jsonl）中的system prompt,还多了一段话system\n\nCutting Knowledge Date: December 2023\； 
这是为什么？怎么把这段话丢弃？另外fsf model是不是也有类似的问题？