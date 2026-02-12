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
logs/grpo/gemma3_4b_gspo_8x80g.log
```
# 问题
检查目前实现逻辑和上次commit的区别，上一次不会OOM，这次反而OOM，仔细检查OOM发生的时机，为什么这次就OOM了？是不是Reward模型没释放显存