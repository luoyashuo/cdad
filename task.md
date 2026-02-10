# 我需要启动4个sft训练(2GPU)
# step1 构建数据集
# 1. CD Optimal 策略（7B 模型配置：CD 范围 0-6）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s cd_optimal \
    --cd_min 0 --cd_max 6 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_cd_optimal \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 2. RV Optimal 策略（7B 模型配置：RV 范围 3-5）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s rv_optimal \
    --rv_min 3 --rv_max 5 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_rv_optimal \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 3. Combined 双规则概率采样（7B 模型配置）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --mu_cd 6 \
    --beta 0.5 \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_combined \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# 4. Combined + FSF 最低（在候选范围内选择 FSF 最低的）
"""
python build_sharegpt_dataset.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
       /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    -s combined \
    --cd_min 0 --cd_max 6 \
    --rv_min 3 --rv_max 5 \
    --fsf_lowest \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_combined_fsf \
    --train-ratio 0.99 \
    -r ../LlamaFactory/data/dataset_info.json
"""

# step2：编写llamafactory的config
参考：
/mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/LlamaFactory/examples/baseline

# step3: 编写训练脚本
参考：
/mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/LlamaFactory/examples/baseline/run.sh
但是顺序启动4个训练

# step4：编写eval脚本
参考：
/mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/eval/eval_configs

# step5：编写eval启动脚本
参考：/mnt/moonfs/luoyashuo-ksyun/code/graphkd/cdad/eval/launch_scripts/run.sh
顺序启动eval