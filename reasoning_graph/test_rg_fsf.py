#!/usr/bin/env python3
"""
Test script for FSF analysis - samples a small subset for quick verification.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pydot


def parse_dot_graph(dot_code: str) -> Dict[str, int]:
    """Parse DOT graph and extract metrics."""
    try:
        graphs = pydot.graph_from_dot_data(dot_code)
        if not graphs or len(graphs) == 0:
            return None
        
        graph = graphs[0]
        if graph is None:
            return None
        
        nodes = graph.get_nodes()
        edges = graph.get_edges()
        
        node_count = len(nodes) if nodes else 0
        edge_count = len(edges) if edges else 0
        
        fsf = edge_count / max(node_count, 1)
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'fsf': fsf
        }
    except Exception as e:
        return None


def test_analysis():
    """Test with first 1000 entries."""
    file_path = "/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_info.jsonl"
    output_dir = Path("rg_plot_test")
    output_dir.mkdir(exist_ok=True)
    
    print("Testing with first 1000 entries...")
    
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            entry = json.loads(line)
            metrics = parse_dot_graph(entry['reasoning_graph'])
            if metrics:
                data.append({
                    'q_id': entry['q_id'],
                    't_id': entry['t_id'],
                    'model': entry['model'],
                    'run_id': entry['run_id'],
                    **metrics
                })
    
    print(f"Successfully parsed {len(data)} entries")
    
    # Calculate stats
    fsf_values = [d['fsf'] for d in data]
    print(f"\nFSF Statistics (sample):")
    print(f"Mean: {np.mean(fsf_values):.4f}")
    print(f"Std: {np.std(fsf_values):.4f}")
    print(f"Min: {np.min(fsf_values):.4f}")
    print(f"Max: {np.max(fsf_values):.4f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(fsf_values, bins=30, alpha=0.7, edgecolor='black')
    plt.title('FSF Distribution (Sample, n=1000)')
    plt.xlabel('FSF')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'fsf_test_distribution.png')
    plt.close()
    
    print(f"\nTest plot saved to {output_dir}/fsf_test_distribution.png")
    
    return data


if __name__ == '__main__':
    test_analysis()
