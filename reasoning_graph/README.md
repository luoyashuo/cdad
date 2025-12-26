# Reasoning Graph Builder

This tool builds reasoning graphs from Chain-of-Thought (COT) traces and computes the Failed-Step Fraction (FSF) metric as described in the paper "What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT".

## Overview

The tool:
1. Reads a dataset in JSONL format containing reasoning traces
2. Uses an LLM to convert each reasoning trace into a Graphviz DOT representation
3. Computes FSF (Failed-Step Fraction) from the graph
4. Saves the results back to the dataset

FSF is calculated as:
```
FSF = (number of failed nodes) / (total number of nodes)
```

Where failed nodes are those marked with `fillcolor=lightpink` in the graph.

## Requirements

- Python 3.10+
- Access to LLM API (configured via AsyncConsulOpenAIProxy)
- asyncio, aiohttp

## Usage

### Basic Usage

Process a dataset:

```bash
python3 build_reasoning_graph.py \
    --input /path/to/input.jsonl \
    --output /path/to/output.jsonl \
    --model gpt-oss-120b-judge \
    --max-concurrent 10
```

### Parameters

- `--input`: Path to input JSONL file
- `--output`: Path to output JSONL file
- `--model`: Model name for LLM API (default: gpt-oss-120b-judge)
- `--start-idx`: Starting index for processing (default: 0)
- `--end-idx`: Ending index for processing (default: process all)
- `--max-concurrent`: Maximum concurrent requests (default: 10)
- `--temperature`: Temperature for LLM generation (default: 0.1)
- `--max-tokens`: Max tokens for LLM generation (default: 4096)

### Batch Processing

Use the shell script to process the entire OmniThought Math dataset:

```bash
./run_build_graph.sh
```

This will:
- Process `train.jsonl` and `test.jsonl`
- Save results to a new directory with `_with_graphs` suffix
- Use default model and concurrency settings

## Data Format

### Input Format

Each line in the input JSONL file should be a JSON object with:

```json
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "solution": "...",
            ...
        }
    ]
}
```

### Output Format

The output adds `reasoning_graph` field to each reasoning item:

```json
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "solution": "...",
            "reasoning_graph": {
                "reasoning_graph": "digraph G { ... }",
                "fsf": 0.25
            }
        }
    ]
}
```

## Testing

Test with a single sample:

```bash
python3 test_build_reasoning_graph.py
```

This will:
1. Test DOT parsing with a sample graph
2. Process a single sample from the dataset
3. Generate visualization files in `test_output/`

## Implementation Details

### LLM Prompt

The prompt asks the LLM to:
- Create one node per distinct reasoning step
- Color successful steps light blue, failed attempts light pink
- Connect nodes based on information flow
- Include all steps without reordering
- Always start with "problem statement" and end with "final answer"
- Generate valid Graphviz DOT code

### FSF Calculation

FSF is computed by:
1. Counting all nodes in the DOT code (nodes with `[...]`)
2. Counting nodes with `fillcolor=lightpink`
3. Dividing failed nodes by total nodes

### Concurrency

The tool uses asyncio with a semaphore to limit concurrent LLM requests, preventing overload.

## Output Structure

Results are saved to:
```
/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k_with_graphs/
├── train.jsonl
├── test.jsonl
└── dataset_info.json
```

The FSF values range from 0 (no failed steps) to 1 (all steps failed), with lower values indicating more efficient reasoning according to the paper's findings.
