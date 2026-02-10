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
为什么log中打印的样本完全一样？
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   QUESTION: Evaluate the limit \( \lim_{x \to \infty} \frac{\left(\frac{6}{e}\right)^x}{\Gamma(x+1)} \).
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   ORIGINAL_CoT: Okay, so I need to find the limit as x approaches infinity of (6/e)^x divided by the gamma function ...
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   REWRITE_CoT: To evaluate the limit $ \lim_{x \to \infty} \frac{\left(\frac{6}{e}\right)^x}{\Gamma(x+1)} $, we beg...
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE] SAMPLE[1] reward=0.0000
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   QUESTION: Evaluate the limit \( \lim_{x \to \infty} \frac{\left(\frac{6}{e}\right)^x}{\Gamma(x+1)} \).
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   ORIGINAL_CoT: Okay, so I need to find the limit as x approaches infinity of (6/e)^x divided by the gamma function ...
[36m(TaskRunner pid=760424)[0m 2026-02-11 00:25:06 | INFO     | reward | [REWARD_PRE]   REWRITE_CoT: To evaluate the limit $ \lim_{x \to \infty} \frac{\left(\frac{6}{e}\right)^x}{\Gamma(x+1)} $, we beg...