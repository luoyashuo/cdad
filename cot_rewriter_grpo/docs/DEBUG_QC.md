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
为什么保存的数据/data/luoys/checkpoints/cdad/rollout_data/llama3_8b_grpo_8x80g_v7/1.jsonl 第三行
  "too_short_penalties": 0.0,
  "too_long_penalties": 0.0,
  但是  "length_ratio": 0.3328100470957614,
  按理说应该吃到惩罚才对啊

