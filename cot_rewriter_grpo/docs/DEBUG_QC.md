# 任务
我运行训练任务bash scripts/grpo/start.sh的报错
运行环境(uv)：
```
source /root/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是2*A100-80G
4. 你不能修改数据的长度， response长度
5. 参考原版verl逻辑：/root/luoys/projects/verl-0.5.0
# 报错信息如下
```
tail -n 200 /root/luoys/projects/cdad/cot_rewriter_grpo/logs/grpo/qwen3_4b_grpo_2x80g_full_sglang.log
```



# 任务
训练任务：scripts/grpo/start_qc.sh
运行环境(uv): source /root/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate

1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是2*A100-80G
4. 你不能修改数据的长度， response长度
5. 参考原版verl逻辑：/root/luoys/projects/verl-0.5.0

我希望在rollout阶段，log打印进度，和每个prompt len（token）, 以及前200个token
每个数据得到response后，也打印response的前200个token和len
Reward阶段也是这样，每个prompt前后打印，以及reward值，同时显示进度