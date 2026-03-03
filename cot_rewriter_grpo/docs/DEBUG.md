# 任务
训练任务: bash scripts/grpo/start_ruc.sh
运行环境(uv)：
```
/home/u20240075/luoyashuo/projects/cdad/cot_rewriter_grpo/.venv
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
# 报错
我希望你彻底地重构所有penalty的实现方式，一旦有penalty出现：
1. correct_judge
2. overlong
3. format
等出现了penalty
就直接把reward变成0，而不是在原有的reward上进行减
然后对应的swanlab的统计也要更新