实现fsf_judge模型的测试脚本
1. 基于vllm加载模型（例如/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000）
2. 在验证集（/data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl）上进行推理，max_token=3, 并且记录第一个token中N,Y之间的softmax概率值，作为包含”失败分支“的概率，把推理细节保存到ckpt路径下，例如/data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val/xxx.jsonl
3. 根据softmax概率值计算precision下的recall和threshold，绘制PR曲线(为了更好的模块化，该功能的输入为jsonl文件，输出为对应路径下，即fsf_judge_val文件夹下的一个csv文件，表示不同precision下的recall和threshold，还有pr曲线)