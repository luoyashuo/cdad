# 任务
训练任务: bash scripts/grpo/start_qc.sh
运行环境(uv)：
```
source /root/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是8*A100-80G
4. 你不能修改数据的长度， response长度
5. 参考原版verl逻辑：/root/luoys/projects/verl-0.5.0

# LOG
```
/root/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_dapo_8x80g_vllm_rollout_sglang_reward.log
```

为什么log有：
[36m(TaskRunner pid=3866367)[0m 2026-02-09 16:52:28 | INFO     | reward | [REWARD] Sample 15: No valid <rewrite_reasoning> tags found, using original text for reward computation
这种格式不对的话，应该直接给format惩罚了才对吧
另外/root/luoys/checkpoints/cdad/qwen3_4b_dapo_8x80g/rollout_data/1.jsonl里面保存的数据，"output"字段是rollout的结果还是经过处理，提取了<rewrite_reasoning>内部数据的字段？