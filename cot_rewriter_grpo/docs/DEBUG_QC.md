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
5. 参考原版verl逻辑：/data/luoys/projects/verl-0.5.0
# LOG
```
/data/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_dapo_8x80g_vllm_rollout_sglang_reward.log
```
# 问题
我想要修改这个参数too_short_threshold
就是设置一个上下界：
too_short_threshold_high: 0.8
too_short_threshold_low: 0.4
在上下界内部，惩罚线性增长，超过上下界，惩罚固定为最大值
