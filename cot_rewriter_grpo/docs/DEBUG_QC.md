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

# 问题
我希望correct_judge_model的功能多一个可以在config中配置的功能，用来控制是否过滤judges结果是No的样本（不参与GRPO优势计算）； 另外可以对correct_judge_model的输出长度限制一下，最多10个token，加快速度
