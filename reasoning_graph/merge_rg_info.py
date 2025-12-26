#!/usr/bin/env python3
"""
Merge reasoning graph output files with input data.

This script:
1. Loads input JSONL file (original dataset)
2. Loads one or more output JSONL files (reasoning graph results)
3. Groups output entries by (q_id, t_id)
4. Merges reasoning_graph data into the original input structure
5. Adds reasoning_graph field (list) to each thought in the reasoning array

The output format maintains the original input structure with added reasoning_graph:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "reasoning_graph": [
                {"reasoning_graph": "...", "model": "...", "run_id": 1},
                {"reasoning_graph": "...", "model": "...", "run_id": 2},
                ...
            ]
        },
        ...
    ]
}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_input_entries(input_path: str) -> List[Dict[str, Any]]:
    """Load all entries from input JSONL file."""
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_output_entries(output_paths: List[str]) -> List[Dict[str, Any]]:
    """Load all entries from one or more output JSONL files."""
    entries = []
    for output_path in output_paths:
        path = Path(output_path)
        if not path.exists():
            print(f"Warning: Output file {output_path} does not exist, skipping...")
            continue

        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Ensure required fields exist
                    if all(field in entry for field in ['q_id', 't_id', 'reasoning_graph', 'model', 'run_id']):
                        entries.append(entry)
    return entries


def group_output_by_thought(output_entries: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Group output entries by (q_id, t_id) for efficient lookup.

    Returns:
        Dictionary mapping (q_id, t_id) to list of reasoning_graph results
    """
    grouped = defaultdict(list)

    for entry in output_entries:
        q_id = entry['q_id']
        t_id = entry['t_id']
        key = (q_id, t_id)

        # Extract the reasoning graph result
        result = {
            'reasoning_graph': entry['reasoning_graph'],
            'model': entry['model'],
            'run_id': entry['run_id']
        }
        grouped[key].append(result)

    return grouped


def merge_entries(input_entries: List[Dict[str, Any]],
                  output_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge reasoning graph data into original input structure.

    Args:
        input_entries: Original input entries
        output_entries: Generated reasoning graph results

    Returns:
        Merged entries with reasoning_graph added to each thought
    """
    # Group output entries by (q_id, t_id)
    output_by_thought = group_output_by_thought(output_entries)

    # Create merged entries
    merged_entries = []

    for q_id, input_entry in enumerate(input_entries):
        # Check if this question has any reasoning graphs
        has_reasoning_graph = False

        # Create a copy of the input entry
        merged_entry = input_entry.copy()

        # Get reasoning array (if it exists)
        if 'reasoning' not in merged_entry:
            merged_entry['reasoning'] = []

        # Ensure reasoning is a list
        if not isinstance(merged_entry['reasoning'], list):
            merged_entry['reasoning'] = []

        # Process each thought in the reasoning array
        for t_id, thought_item in enumerate(merged_entry['reasoning']):
            # Ensure thought_item is a dict
            if not isinstance(thought_item, dict):
                thought_item = {'thought': str(thought_item)}

            # Get reasoning graphs for this thought
            key = (q_id, t_id)
            if key in output_by_thought:
                # Add reasoning_graph array to the thought
                thought_item['reasoning_graph'] = output_by_thought[key]
                has_reasoning_graph = True

        # Only include entry if it has at least one reasoning graph
        if has_reasoning_graph:
            merged_entries.append(merged_entry)

    return merged_entries


def save_merged_entries(output_path: str, entries: List[Dict[str, Any]]):
    """Save merged entries to output JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Merge reasoning graph output files with input data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge single output file
  python3 reasoning_graph/merge_rg_info.py \
      --input /path/to/input.jsonl \
      --output /path/to/merged.jsonl \
      --rg-files /path/to/rg_info.jsonl

  # Merge multiple output files
  python3 reasoning_graph/merge_rg_info.py \
      --input /path/to/input.jsonl \
      --output /path/to/merged.jsonl \
      --rg-files /path/to/rg_part1.jsonl /path/to/rg_part2.jsonl

  # Use pattern to match multiple files
  python3 reasoning_graph/merge_rg_info.py \
      --input /path/to/input.jsonl \
      --output /path/to/merged.jsonl \
      --rg-files "/path/to/rg_info_*.jsonl"
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSONL file (original dataset)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSONL file (merged results)')
    parser.add_argument('--rg-files', type=str, nargs='+', required=True,
                        help='One or more reasoning graph output JSONL files')

    args = parser.parse_args()

    print(f"Loading input from: {args.input}")
    input_entries = load_input_entries(args.input)
    print(f"Loaded {len(input_entries)} input entries")

    print(f"Loading output from: {args.rg_files}")
    output_entries = load_output_entries(args.rg_files)
    print(f"Loaded {len(output_entries)} output entries")

    if not output_entries:
        print("Error: No output entries found!")
        return

    print("Merging entries...")
    merged_entries = merge_entries(input_entries, output_entries)
    print(f"Merged {len(merged_entries)} entries with reasoning graph information")

    print(f"Saving to: {args.output}")
    save_merged_entries(args.output, merged_entries)

    print("\n" + "="*80)
    print("✅ Merge completed successfully!")
    print("="*80)
    print(f"Total questions with reasoning graphs: {len(merged_entries)}")
    print(f"Output format: Original structure + reasoning_graph array per thought")
    print("="*80)

    # Show a sample entry
    if merged_entries:
        print("\n📊 SAMPLE ENTRY:")
        sample = merged_entries[0]
        print(f"Question: {sample.get('question', '')[:80]}...")
        print(f"Number of reasoning steps: {len(sample.get('reasoning', []))}")

        if 'reasoning' in sample and len(sample['reasoning']) > 0:
            first_thought = sample['reasoning'][0]
            print(f"\nFirst thought:")
            print(f"  Thought: {first_thought.get('thought', '')[:60]}...")

            if 'reasoning_graph' in first_thought:
                rg_list = first_thought['reasoning_graph']
                print(f"  Number of reasoning graphs: {len(rg_list)}")
                print(f"  Models: {', '.join(set(r['model'] for r in rg_list))}")
                print(f"  Runs: {min(r['run_id'] for r in rg_list)}-{max(r['run_id'] for r in rg_list)}")

                # Show first reasoning graph preview
                if rg_list:
                    first_rg = rg_list[0]
                    print(f"\n  First reasoning graph (model: {first_rg['model']}, run: {first_rg['run_id']}):")
                    preview = first_rg['reasoning_graph'][:100] + '...' if len(first_rg['reasoning_graph']) > 100 else first_rg['reasoning_graph']
                    print(f"    {preview}")
                    print(f"    Length: {len(first_rg['reasoning_graph'])} chars")

        print("\n" + "="*80)


if __name__ == '__main__':
    main()
