# FSF Judge Evaluation Scripts

This directory contains scripts for evaluating FSF (Failed Step Fraction) Judge models.

**Location**: `fsf_judge/` directory

## Scripts Overview

### 1. `fsf_judge_inference.py`

Runs inference on a validation dataset using vLLM and records the softmax probabilities for the first token (N/Y).

**Usage:**
```bash
python fsf_judge/fsf_judge_inference.py \
    --model_path /path/to/checkpoint-12000 \
    --data_path /path/to/val.jsonl \
    --output_dir /path/to/checkpoint-12000/fsf_judge_val \
    --max_tokens 5 \
    --batch_size 32 \
    --tensor_parallel_size 1 \
    --dtype bfloat16
```

**Arguments:**
- `--model_path`: Path to the trained FSF Judge model checkpoint
- `--data_path`: Path to the validation dataset (jsonl file)
- `--output_dir`: Output directory for inference results (default: `{model_path}/fsf_judge_val`)
- `--max_tokens`: Maximum number of tokens to generate (default: 5)
- `--batch_size`: Batch size for inference (default: 32)
- `--tensor_parallel_size`: Tensor parallelism size for vLLM (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization for vLLM (default: 0.9)
- `--dtype`: Data type for inference: float16, bfloat16, float32 (default: bfloat16)

**Output:**
- `inference_results.jsonl`: Inference results with softmax probabilities
- `metrics.json`: Basic metrics (accuracy, etc.)

**Output Format (inference_results.jsonl):**
```json
{
  "index": 0,
  "messages": [...],
  "target_label": "Y",
  "generated_text": "Y\n...",
  "generated_first_token": "Y",
  "predicted_label": "Y",
  "prob_N": 0.1234,
  "prob_Y": 0.8766,
  "failed_branch_prob": 0.8766
}
```

### 2. `fsf_judge_analyze_pr.py`

Analyzes inference results and computes Precision-Recall curves at different thresholds.

**Usage:**
```bash
python fsf_judge/fsf_judge_analyze_pr.py \
    --input_path /path/to/checkpoint-12000/fsf_judge_val/inference_results.jsonl \
    --output_dir /path/to/checkpoint-12000/fsf_judge_val \
    --num_thresholds 100
```

**Arguments:**
- `--input_path`: Path to the inference results jsonl file
- `--output_dir`: Output directory for PR curve data and plots
- `--num_thresholds`: Number of threshold points to compute (default: 100)
- `--positive_label`: Label considered as positive class (default: Y)

**Output:**
- `pr_curve_data.csv`: Full PR curve data with all metrics at each threshold
- `precision_recall_threshold.csv`: Summary table of precision levels and corresponding recall/threshold
- `pr_curve.png`: PR curve visualization
- `score_distribution.png`: Distribution of predicted probabilities by class

**CSV Format (precision_recall_threshold.csv):**
```csv
target_precision,achieved_precision,recall,threshold,f1_score
0.50,0.5123,0.9876,0.1000,0.6754
0.55,0.5532,0.9654,0.1500,0.7123
...
```

### 3. `fsf_judge_evaluate.py` (Pipeline)

Combines inference and PR curve analysis into a single pipeline for convenience.

**Usage:**
```bash
python fsf_judge/fsf_judge_evaluate.py \
    --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
    --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
    --max_tokens 5 \
    --batch_size 32 \
    --tensor_parallel_size 1
```

**Arguments:**
All arguments from both inference and analysis scripts, plus:
- `--skip_inference`: Skip inference if results already exist
- `--skip_analysis`: Skip PR curve analysis

## Example Workflow

### Quick Start

```bash
# Run full evaluation pipeline
python fsf_judge/fsf_judge_evaluate.py \
    --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
    --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
    --batch_size 32
```

### Multi-GPU Inference

```bash
# Use 2 GPUs for tensor parallelism
python fsf_judge/fsf_judge_evaluate.py \
    --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
    --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
    --tensor_parallel_size 2 \
    --batch_size 64
```

### Step-by-Step

```bash
# Step 1: Run inference only
python fsf_judge/fsf_judge_inference.py \
    --model_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000 \
    --data_path /data-share/yeesuanAI08/lihaitao/luoys/data/cdad/fsf_judge/val.jsonl \
    --output_dir /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val

# Step 2: Analyze results and generate PR curve
python fsf_judge/fsf_judge_analyze_pr.py \
    --input_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val/inference_results.jsonl
```

## Requirements

```bash
# Core dependencies
pip install vllm transformers torch numpy tqdm

# For plotting (optional)
pip install matplotlib
```

## Understanding the Output

### Probability Interpretation

- `prob_N`: Softmax probability of token "N" (No failed branches)
- `prob_Y`: Softmax probability of token "Y" (Yes, contains failed branches)
- `failed_branch_prob`: Same as `prob_Y` - the probability that the reasoning trace contains failed branches

### PR Curve Interpretation

- **Positive class**: "Y" (contains failed branches)
- **Precision**: Of all predictions marked as containing failed branches, what fraction actually do?
- **Recall**: Of all reasoning traces that actually contain failed branches, what fraction were detected?
- **Threshold**: The probability cutoff used to make the binary decision

### Selecting a Threshold

Use `precision_recall_threshold.csv` to find the optimal threshold for your use case:

- If you need high precision (few false positives), use a higher threshold
- If you need high recall (few false negatives), use a lower threshold
- For balanced performance, look for the threshold with the highest F1 score
