我想要参考：
/home/luoys/cdad/cot_rewriter_grpo/inference/configs/deepmath_plain_30k_train_llama3_8b_1b/run.sh
的逻辑，基于token skip，以一定比例对原始cot进行裁剪
输入文件格式为：
- /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_plain_30k/train.jsonl
其中thought = conversations[-1]['content'].split('<think>\n')[-1].split('\n</think>')[0].strip()
solution = conversations[-1]['content'].split('<think>\n')[-1].split('\n</think>')[1].strip()
我想要构建token skip的数据集，一部分数据用原始的CoT，另一部分用token skip后的cot，按照我们自定义的比例
lingua模型为：/data/hf_home/models/microsoft/llmlingua-2-xlm-roberta-large-meetingbank
配置文件为：/home/luoys/cdad/baselines/TokenSkip/dataset_build/config/xxxx.yaml
启动脚本为: /home/luoys/cdad/baselines/TokenSkip/dataset_build/run.sh