#!/usr/bin/env python3
"""
Merge reasoning graph output files with input data.

This script:
1. Loads input JSONL file (original dataset)
2. Loads one or more output JSONL files (reasoning graph results)
3. Groups output entries by (q_id, t_id)
4. Computes FSF metrics for each reasoning graph (node_count, failed_node_count, edge_count, fsf)
5. Merges reasoning_graph data into the original input structure
6. Adds reasoning_graph field (list) to each thought in the reasoning array

The output format maintains the original input structure with added reasoning_graph:
{
    "question": "...",
    "reasoning": [
        {
            "thought": "...",
            "reasoning_graph": [
                {
                    "reasoning_graph": "...",
                    "model": "...",
                    "run_id": 1,
                    "node_count": 10,
                    "failed_node_count": 2,
                    "edge_count": 9,
                    "fsf": 0.2
                },
                ...
            ]
        },
        ...
    ]
}

FSF (Failed-Step Fraction) is calculated as: failed_node_count / node_count
where failed nodes are those with fillcolor=lightpink in the DOT graph.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import pydot
from multiprocessing import Pool
from tqdm import tqdm


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


def parse_dot_graph_parallel(args: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse DOT graph and compute FSF metrics."""
    dot_code, entry = args
    try:
        graphs = pydot.graph_from_dot_data(dot_code)
        if not graphs or len(graphs) == 0:
            return None
        
        graph = graphs[0]
        if graph is None:
            return None
        
        nodes = graph.get_nodes() or []
        edges = graph.get_edges() or []
        
        # Count total nodes
        node_count = len(nodes)
        
        # Count failed nodes (nodes with fillcolor=lightpink)
        failed_nodes = 0
        for node in nodes:
            if node is None:
                continue
            try:
                fillcolor = node.get('fillcolor')
                if fillcolor is not None:
                    fillcolor = str(fillcolor).strip("'\"").lower()
                    if fillcolor == 'lightpink':
                        failed_nodes += 1
            except:
                continue
        
        edge_count = len(edges)
        
        # Compute FSF: Failed-Step Fraction = failed_nodes / total_nodes
        fsf = failed_nodes / max(node_count, 1) if node_count > 0 else 0.0
        
        return {
            'node_count': node_count,
            'failed_node_count': failed_nodes,
            'edge_count': edge_count,
            'fsf': fsf
        }
    except Exception as e:
        return None


def group_output_by_thought(output_entries: List[Dict[str, Any]], num_workers: int = 8) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Group output entries by (q_id, t_id) for efficient lookup.
    Also computes FSF metrics for each reasoning graph.

    Args:
        output_entries: List of output entries to process
        num_workers: Number of worker processes for parallel processing

    Returns:
        Dictionary mapping (q_id, t_id) to list of reasoning_graph results with FSF metrics
    """
    if len(output_entries) == 0:
        return defaultdict(list)
    
    print(f"Computing FSF metrics for {len(output_entries)} reasoning graphs...")
    print(f"Using {num_workers} workers for parallel processing...")
    
    # Prepare arguments for parallel processing
    process_args = [(entry['reasoning_graph'], entry) for entry in output_entries]
    
    # Process entries in parallel with progress bar
    fsf_metrics_list = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(process_args), desc="Parsing DOT graphs") as pbar:
            for result in pool.imap_unordered(parse_dot_graph_parallel, process_args):
                fsf_metrics_list.append(result)
                pbar.update(1)
    
    # Group processed entries by (q_id, t_id)
    grouped = defaultdict(list)
    success_count = 0
    fail_count = 0
    
    for i, entry in enumerate(output_entries):
        q_id = entry['q_id']
        t_id = entry['t_id']
        key = (q_id, t_id)
        
        # Get the pre-computed FSF metrics
        fsf_metrics = fsf_metrics_list[i]
        
        # Extract the reasoning graph result with FSF metrics
        result = {
            'reasoning_graph': entry['reasoning_graph'],
            'model': entry['model'],
            'run_id': entry['run_id']
        }
        
        # Add FSF metrics and track parsing success/failure
        if fsf_metrics is not None:
            result.update(fsf_metrics)
            success_count += 1
        else:
            # Set default values if parsing failed
            result.update({
                'node_count': 0,
                'failed_node_count': 0,
                'edge_count': 0,
                'fsf': 0.0
            })
            fail_count += 1
        
        grouped[key].append(result)
    
    print(f"  ✅ Successfully parsed: {success_count}/{len(output_entries)}")
    if fail_count > 0:
        print(f"  ⚠️  Failed to parse: {fail_count}/{len(output_entries)}")
    return grouped


def merge_entries(input_entries: List[Dict[str, Any]],
                  output_entries: List[Dict[str, Any]], 
                  num_workers: int = 8) -> List[Dict[str, Any]]:
    """
    Merge reasoning graph data into original input structure.

    Args:
        input_entries: Original input entries
        output_entries: Generated reasoning graph results
        num_workers: Number of worker processes for parallel processing

    Returns:
        Merged entries with reasoning_graph added to each thought
    """
    # Group output entries by (q_id, t_id)
    output_by_thought = group_output_by_thought(output_entries, num_workers)

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
        description='Merge reasoning graph output files with input data and compute FSF metrics',
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

Note: This script will parse each DOT graph to compute FSF (Failed-Step Fraction)
and other metrics (node_count, failed_node_count, edge_count).
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSONL file (original dataset)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSONL file (merged results)')
    parser.add_argument('--rg-files', type=str, nargs='+', required=True,
                        help='One or more reasoning graph output JSONL files')
    parser.add_argument('--num-workers','-w', type=int, default=8,
                        help='Number of worker processes for parallel processing (default: 8)')

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
    merged_entries = merge_entries(input_entries, output_entries, args.num_workers)
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
                    print(f"    Node count: {first_rg.get('node_count', 0)}")
                    print(f"    Failed nodes: {first_rg.get('failed_node_count', 0)}")
                    print(f"    Edge count: {first_rg.get('edge_count', 0)}")
                    print(f"    FSF: {first_rg.get('fsf', 0.0):.6f}")

        print("\n" + "="*80)


if __name__ == '__main__':
    main()
"""
python3 reasoning_graph/merge_rg_info.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/train_merge.jsonl \
    --rg-files /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_all/rg_info.jsonl \
    -w 60
python3 reasoning_graph/merge_rg_info.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/train_merge.jsonl \
    --rg-files /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all/rg_info.jsonl \
    -w 60


"""