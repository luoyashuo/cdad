# Data Directory

This directory contains the training and validation data for CoT Rewriter GRPO.

## Data Format

### Input Format (JSON/JSONL)

Each sample should contain:

```json
{
  "question": "The question or problem to solve",
  "original_cot": "The original Chain-of-Thought reasoning (may contain errors, backtracking)",
  "answer": "The correct answer"
}
```

### Output Format (Parquet)

After processing with `src/data/dataset.py`, the data is converted to Parquet format with:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | string | Formatted rewrite prompt for the model |
| `question` | string | Original question |
| `original_cot` | string | Original CoT reasoning |
| `ground_truth` | string | Correct answer |

## Creating Training Data

### From Sample Data

```bash
# Create sample data and split into train/val
python src/data/dataset.py
```

### From Custom Data

```python
from src.data.dataset import prepare_dataset

prepare_dataset(
    input_path="your_data.json",  # or .jsonl
    output_dir="data",
    train_ratio=0.9,
    seed=42,
)
```

## Directory Structure

```
data/
├── README.md           # This file
├── examples/           # Sample data
│   └── sample_data.json
├── train.parquet       # Training data (generated)
└── val.parquet         # Validation data (generated)
```

## Data Quality Guidelines

For best training results:

1. **Original CoT** should contain realistic reasoning with:
   - Some failed attempts or wrong directions
   - Backtracking and corrections
   - Eventually arriving at the correct answer

2. **Questions** should be diverse:
   - Math problems
   - Logic puzzles
   - Code understanding
   - General reasoning tasks

3. **Answers** should be clear and unambiguous

## Example Dataset Generation

If you have a dataset of math problems with solutions, you can generate CoT data:

```python
import json

# Example: Convert GSM8K-style data
def convert_gsm8k_to_cot_format(gsm8k_data):
    converted = []
    for item in gsm8k_data:
        converted.append({
            "question": item["question"],
            "original_cot": item["reasoning"],  # GSM8K has reasoning
            "answer": item["answer"],
        })
    return converted
```
