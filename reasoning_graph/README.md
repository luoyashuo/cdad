# Reasoning Graph Generation

This directory contains tools for generating reasoning graphs from thought traces.

## 📋 Important: Understanding Formats

This pipeline uses **three different formats** at different stages:

1. **Input Format** - Original dataset structure
2. **Build Output Format** - Flat, thought-level results (for generation)
3. **Merged Format** - Nested, convenient structure (for analysis)

**Please read [FORMAT_GUIDE.md](FORMAT_GUIDE.md) first** to understand which format to use when.

## Files

- `build_reasoning_graph.py` - Generate reasoning graphs (creates flat format)
- `merge_rg_info.py` - Merge into convenient nested format
- `validate_rg_parsing.py` - Validate flat format (pydot parsing)
- `validate_merged_format.py` - Validate merged format (structure + parsing)
- `verify_output_format.py` - Verify output format
- `preview_output.py` - Preview output files
- `FORMAT_GUIDE.md` - Essential guide to understanding formats

## Quick Start

### 1. Generate Reasoning Graphs

```bash
python3 reasoning_graph/build_reasoning_graph.py \
    --input /path/to/input.jsonl \
    --output /path/to/output.jsonl \
    --models gpt-oss-120b-judge claude-3.7-sonnet Qwen2-5-72B-Instruct \
    --runs-per-model 3 \
    --start-idx 0 \
    --end-idx 100 \
    --max-concurrent-tasks 256
```

### 2. Validate Build Output (Recommended)

```bash
python3 reasoning_graph/validate_rg_parsing.py \
    --file /path/to/output.jsonl
```

### 3. Merge into Convenient Format

```bash
python3 reasoning_graph/merge_rg_info.py \
    --input /path/to/input.jsonl \
    --output /path/to/merged.jsonl \
    --rg-files /path/to/output.jsonl
```

### 4. Validate Merged Output (Recommended)

```bash
python3 reasoning_graph/validate_merged_format.py \
    --file /path/to/merged.jsonl
```

## Output Formats

### Build Output Format (Flat)

Each record represents one (thought, model, run) combination:
```json
{
    "question": "What is 2+2?",
    "thought": "Let me add 2 and 2. 2+2=4.",
    "q_id": 0,
    "t_id": 0,
    "reasoning_graph": "digraph G { ... }",
    "model": "gpt-oss-120b-judge",
    "run_id": 1
}
```

**Use when:** Generating graphs, checkpointing, parallel processing

### Merged Format (Nested)

Maintains original structure with reasoning_graph arrays:
```json
{
    "question": "What is 2+2?",
    "reasoning": [
        {
            "thought": "Let me add 2 and 2. 2+2=4.",
            "reasoning_graph": [
                {"reasoning_graph": "...", "model": "gpt-oss-120b-judge", "run_id": 1},
                {"reasoning_graph": "...", "model": "gpt-oss-120b-judge", "run_id": 2}
            ]
        }
    ]
}
```

**Use when:** Analysis, training, visualization

## Parallel Processing Example

To process a large dataset in parallel chunks:

```bash
# Terminal 1 - Process first 1000 entries
python3 reasoning_graph/build_reasoning_graph.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_part1.jsonl \
    --models gpt-oss-120b-judge \
    --runs-per-model 3 \
    --start-idx 0 \
    --end-idx 1000 \
    --max-concurrent-tasks 256

# Terminal 2 - Process next 1000 entries
python3 reasoning_graph/build_reasoning_graph.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_part2.jsonl \
    --models gpt-oss-120b-judge \
    --runs-per-model 3 \
    --start-idx 1000 \
    --end-idx 2000 \
    --max-concurrent-tasks 256

# Merge all parts
python3 reasoning_graph/merge_rg_info.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_merged.jsonl \
    --rg-files /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_part*.jsonl
```

## Validation

### Why Validate?

All saved reasoning_graph entries are **guaranteed** to be parseable by pydot because:
1. Each graph is validated during generation
2. Invalid graphs are filtered out
3. Only parseable graphs are saved

But validation tools provide additional confidence and help catch any issues.

### Tools

**For build output (flat format):**
```bash
python3 reasoning_graph/validate_rg_parsing.py --file rg_output.jsonl
```

**For merged output (nested format):**
```bash
python3 reasoning_graph/validate_merged_format.py --file rg_merged.jsonl
```

## Quick Test

Run the complete pipeline on a small test dataset:

```bash
cd /home/luoyashuo/research/graph_kd/cdad/reasoning_graph
./USAGE_EXAMPLE.sh
```

Or run automated tests:

```bash
python3 test_merge_format.py
python3 test_validation.py
```

## Requirements

```bash
pip install pydot tqdm
```

## Need Help?

- Read [FORMAT_GUIDE.md](FORMAT_GUIDE.md) for detailed format explanation
- Read [TESTING.md](TESTING.md) for validation details
- Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for command reference
- Run `./USAGE_EXAMPLE.sh` for a live demonstration
