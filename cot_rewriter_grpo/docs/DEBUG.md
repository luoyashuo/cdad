# 任务
训练任务: bash scripts/grpo/start_a100.sh
运行环境(uv)：
```
/home/luoys/cdad/cot_rewriter_grpo/.venv
```
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
# 报错
我需要修改reward mode=kl下的loss计算
1. 目前kl的loss是计算actor的response上（也就是rewrite_cot上）(r_kl_cot)，D_KL((actor(rewrite_cot|sp,q,original_cot)) || student(rewrite_cot|q)) -> 0,1归一化
2. 我需要添加另一个kl项(r_kl_answer)，也就是D_KL((student(answer|q,rewrite_cot)) || student(answer|q,original_cot)) -> 0,1归一化
3. 我需要把这两个归一化后的0，1之间的值的加权和作为reward，然后用reward去更新actor，默认第1项的权重为3/4，第二项的权重为1/4
你需要在swanlab，rollout_data,val_data中记录第二个kl的原始值