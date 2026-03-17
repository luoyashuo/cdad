# Eval V2 使用说明

## 简介

`eval_v2.py` 实现了跨 Benchmark 流式推理，将所有 benchmark 的 prompts 混合进一个全局队列，实现：
- GPU 持续满载推理（通过 vLLM continuous batching）
- Benchmark 推理完成后立即在 CPU 线程触发评分，不阻塞其他 benchmark 的推理
- 支持 Resume 断点续传

## 使用方法

### 基本用法

```bash
python eval_v2.py eval_configs/eval_all_sft_v2.yaml
```

### 断点续传

```bash
# 自动检测已完成的 benchmark 并跳过
python eval_v2.py eval_configs/eval_all_sft_v2.yaml --resume

# 指定续传的输出目录
python eval_v2.py eval_configs/eval_all_sft_v2.yaml --resume-dir /path/to/version_xxx
```

### 指定输出目录

```bash
python eval_v2.py eval_configs/eval_all_sft_v2.yaml --output-dir /custom/output/path
```

## 输出目录结构

```
{model_path}/evalscope/{version}/
├── config.yaml              # 配置文件副本
├── progress_v2.json         # 进度文件（用于 resume）
├── unified_summary.xlsx     # 汇总表格（Excel）
├── unified_summary.json     # 汇总表格（JSON）
├── AIME25/                  # 各 benchmark 的详细输出
│   └── {timestamp}/
│       ├── predictions/{model_name}/aime25_default.jsonl
│       ├── repeat_0/predictions/...   # 各 repeat 的 predictions
│       ├── repeat_1/predictions/...
│       └── ...
├── AIME25_result.json       # AIME25 的评分结果
├── MATH500_result.json      # MATH500 的评分结果
└── ...
```

## 结果文件格式

### {benchmark}_result.json

```json
{
  "benchmark": "AIME25",
  "dataset": "aime25",
  "repeats": 8,
  "per_repeat_scores": [0.23, 0.27, 0.20, 0.23, 0.30, 0.23, 0.27, 0.20],
  "mean": 0.2412,
  "std": 0.0306,
  "min": 0.20,
  "max": 0.30
}
```

### unified_summary.xlsx

| benchmark | dataset | repeats | mean | std | min | max | run_1 | run_2 | ... |
|-----------|---------|---------|------|-----|-----|-----|-------|-------|-----|
| AIME25 | aime25 | 8 | 0.2412 | 0.0306 | 0.20 | 0.30 | 0.23 | 0.27 | ... |
| MATH500 | math_500 | 1 | 0.82 | 0.0 | 0.82 | 0.82 | 0.82 | | |

## 与 eval.py 的区别

| 特性 | eval.py | eval_v2.py |
|------|---------|------------|
| 推理方式 | 串行执行每个 benchmark | 混合所有 prompts 全局推理 |
| GPU 利用率 | benchmark 间有间隙 | 持续满载 |
| 评分方式 | 推理完成后串行评分 | 异步并行评分 |
| 适用场景 | 单 benchmark 或资源受限 | 多 benchmark 大规模评测 |

## 注意事项

1. **配置文件格式**：与 `eval.py` 使用相同的 YAML 配置文件格式
2. **依赖要求**：需要与 `eval.py` 相同的依赖（vllm, evalscope, pandas 等）
3. **内存使用**：由于混合了所有 prompts，内存使用会比 `eval.py` 略高
4. **评分并发**：默认使用 4 个线程进行并发评分，可通过修改代码调整

## 技术实现

1. **Prompt 提取**：通过 evalscope API 加载所有 benchmark 的 dataset，提取 prompts
2. **全局推理**：使用 VLLMOfflineEngine 的 `generate_single()` 方法，支持 per-request SamplingParams
3. **结果收集**：按 benchmark 路由结果，写入 evalscope 兼容的 predictions JSONL
4. **异步评分**：每个 repeat 独立评分，使用 `run_task(use_cache=True)` 跳过推理
