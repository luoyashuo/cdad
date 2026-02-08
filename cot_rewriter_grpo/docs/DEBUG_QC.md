# 任务
训练任务: bash scripts/grpo/start.sh
运行环境(uv)：
```
source /root/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是2*A100-80G
4. 你不能修改数据的长度， response长度
5. 参考原版verl逻辑：/root/luoys/projects/verl-0.5.0
# LOG
```
/root/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_grpo_2x80g_full_sglang.log
```
# 问题
1. 从log中看，每次打印的结果reward都是0“[REWARD_POST] SAMPLE[3] reward=0.0000”，这不对，你检查目前reward计算方式对不对
2. 第二次rollout之后输出乱码，第一次是正常的：
(TaskRunner pid=634413) 2026-02-08 13:56:46 | INFO     | reward | [REWARD_POST] SAMPLE[3] reward=0.0000
(TaskRunner pid=634413) 2026-02-08 13:56:46 | INFO     | reward | [REWARD_POST]   QUESTION: Develop a program that takes a single lowercase word as input and determines the total count of lett...
(TaskRunner pid=634413) 2026-02-08 13:56:46 | INFO     | reward | [REWARD_POST]   ORIGINAL_CoT: Okay, let's see. I need to write a program that takes a lowercase word and counts the number of lett...
(TaskRunner pid=634413) 2026-02-08 13:56:46 | INFO     | reward | [REWARD_POST]   REWRITE_CoT: \brief aけれど .... ################༻  mozilla c20 theorem ........