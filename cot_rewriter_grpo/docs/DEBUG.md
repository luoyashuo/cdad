# 任务
我运行训练任务bash scripts/grpo/start.sh的报错
运行环境(uv)：
```
source /data-share/yeesuanAI08/lihaitao/luoys/projects/cdad/cot_rewriter_grpo/.venv/bin/activate
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. 运行配置是2*A100-40G
4. 你不能修改数据的长度， response长度
# 报错信息如下
```
tail -n 200 logs/grpo/qwen3_4b_grpo_2x40g_lora.log
```
