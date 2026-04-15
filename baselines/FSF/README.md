# FSF Baseline

从 `deepmath_sft_30k/train.jsonl` 的 `reasoning[].thought` 逐条计算 FSF，并构造两个数据集：

- `10k` 纯 `fsf = 0`
- `10k` 纯 `fsf > 0`

约束与原仓库保持一致：

- FSF prompt 与 `verl/verl/experimental/reward/cot_rewriter/sglang_fsf_reward.py` 完全复用
- FSF 解析方式一致：DOT 图中 `fillcolor=lightpink` 的节点数 / 总节点数
- question 级筛选方式一致：只从“至少包含一个 `fsf > 0` thought”的 question 中取样

## 文件

- `build_fsf_datasets.py`
  端到端主脚本。支持打分缓存、断点续跑、构建 raw + ShareGPT 两种输出。
- `run_full.sh`
  默认 8 卡启动脚本。

## 默认输出

输出目录默认是：

`/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits`

其中包含：

- `scored_records.jsonl`
- `fsf_zero_10000/raw_samples.jsonl`
- `fsf_zero_10000/sharegpt.jsonl`
- `fsf_positive_10000/raw_samples.jsonl`
- `fsf_positive_10000/sharegpt.jsonl`
- `summary.json`

## 直接运行

```bash
bash /home/luoys/cdad/baselines/FSF/run_full.sh
```

## V2 数据集（不绑定 question）

第二版数据集的核心变化：zero 与 positive 的 question 池彻底解绑。
zero 候选可从**所有**包含 `fsf=0` thought 的 question 中抽取，不再强制要求该 question 同时也包含 `fsf>0`。

构建方式（直接复用已有的 `scored_records.jsonl`，无需重新跑 FSF 推理）：

```bash
bash /home/luoys/cdad/baselines/FSF/run_v2.sh
```

或手动：

```bash
python /home/luoys/cdad/baselines/FSF/build_fsf_datasets.py \
  --phase build \
  --scored-input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits/scored_records.jsonl \
  --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits_v2 \
  --target-size 10000 \
  --selection random \
  --no-question-binding
```

输出目录默认是 `/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits_v2`。

## 常用参数

```bash
python /home/luoys/cdad/baselines/FSF/build_fsf_datasets.py \
  --phase all \
  --input /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/train.jsonl \
  --output-dir /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_sft_30k/fsf_10k_splits \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --batch-size 256 \
  --target-size 10000 \
  --selection random \
  --resume
```

说明：

- `--resume`
  从已有 `scored_records.jsonl` 继续。
- `--phase score`
  只做 FSF 打分。
- `--phase build`
  只从已有 `scored_records.jsonl` 构造数据集。
- `--pair-on-question`
  强制两个数据集使用同一批 question，此时要求 question 同时有 `fsf=0` 和 `fsf>0` thought。
- `--ray-temp-dir`
  指定 Ray 临时目录；默认是 `/home/luoys/.ray_tmp_fsf`，路径更短，避免 Ray 的 socket 路径过长，同时不挤占 `/tmp`。

## 输出样本格式

raw 格式：

```json
{
  "question_idx": 123,
  "reasoning_idx": 1,
  "question_id": "...",
  "question": "...",
  "thought": "...",
  "solution": "...",
  "fsf": 0.25,
  "failed_nodes": 2,
  "total_nodes": 8
}
```

ShareGPT 格式：

```json
{
  "id": "...",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n..."}
  ]
}
```
