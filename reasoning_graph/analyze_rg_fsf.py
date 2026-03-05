#!/usr/bin/env python3
"""
Fast FSF analysis script with multiprocessing support.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pydot
from multiprocessing import Pool, cpu_count
import tqdm
import os


def parse_dot_graph_parallel(args: Tuple[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse DOT graph in parallel."""
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
            # Get the node's fillcolor attribute
            try:
                fillcolor = node.get('fillcolor')
                if fillcolor is not None:
                    fillcolor = str(fillcolor).strip("'\"").lower()
                    if fillcolor == 'lightpink':
                        failed_nodes += 1
            except:
                # If get fails, node might not have fillcolor attribute
                continue
        
        edge_count = len(edges)
        
        # Compute FSF: Failed-Step Fraction = failed_nodes / total_nodes
        fsf = failed_nodes / max(node_count, 1) if node_count > 0 else 0.0
        
        return {
            'q_id': entry['q_id'],
            't_id': entry['t_id'],
            'question': entry.get('question', ''),
            'thought': entry.get('thought', ''),
            'reasoning_graph': dot_code,  # 保留原始的DOT代码
            'model': entry['model'],
            'run_id': entry['run_id'],
            'node_count': node_count,
            'failed_node_count': failed_nodes,
            'edge_count': edge_count,
            'fsf': fsf
        }
    except Exception as e:
        return None


def load_and_parse_rg_data_fast(file_path: str, max_workers: int = None) -> List[Dict[str, Any]]:
    """Load and parse reasoning graph data using multiprocessing."""
    if max_workers is None:
        max_workers = min(cpu_count(), 16)  # Limit to 16 workers to avoid memory issues
    
    print(f"Loading and parsing data using {max_workers} workers...")
    
    # Load all entries first
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append((entry['reasoning_graph'], entry))
    
    print(f"Processing {len(entries)} entries...")
    
    # Process in parallel
    parsed_data = []
    with Pool(max_workers) as pool:
        results = list(tqdm.tqdm(
            pool.imap(parse_dot_graph_parallel, entries),
            total=len(entries),
            desc="Parsing graphs"
        ))
        
        for result in results:
            if result is not None:
                parsed_data.append(result)
    
    print(f"Successfully parsed {len(parsed_data)} entries")
    print(f"Skipped {len(entries) - len(parsed_data)} entries due to errors")
    
    return parsed_data


def calculate_fsf_statistics(data: List[Dict[str, Any]]) -> Tuple[Dict, List[Dict]]:
    """Calculate FSF statistics."""
    # Group by (q_id, t_id, model)
    thought_model_groups = defaultdict(list)
    for entry in data:
        key = (entry['q_id'], entry['t_id'], entry['model'])
        thought_model_groups[key].append(entry)
    
    # Group by (q_id, t_id) for non-zero FSF analysis (ignoring model)
    thought_groups = defaultdict(list)
    for entry in data:
        key = (entry['q_id'], entry['t_id'])
        thought_groups[key].append(entry)
    
    # Group by q_id for question-level non-zero FSF analysis
    question_groups = defaultdict(list)
    for entry in data:
        key = entry['q_id']
        question_groups[key].append(entry)
    
    # Calculate std for groups with >= 3 runs
    thought_model_stats = []
    for key, entries in thought_model_groups.items():
        if len(entries) >= 3:
            fsf_values = [e['fsf'] for e in entries]
            std = np.std(fsf_values)
            mean = np.mean(fsf_values)
            
            # Also collect node and edge statistics
            node_counts = [e['node_count'] for e in entries]
            failed_node_counts = [e['failed_node_count'] for e in entries]
            edge_counts = [e['edge_count'] for e in entries]
            
            thought_model_stats.append({
                'q_id': key[0],
                't_id': key[1],
                'model': key[2],
                'run_count': len(entries),
                'fsf_mean': mean,
                'fsf_std': std,
                'fsf_values': fsf_values,
                'node_count_mean': np.mean(node_counts),
                'failed_node_count_mean': np.mean(failed_node_counts),
                'edge_count_mean': np.mean(edge_counts)
            })
    
    # Overall statistics
    all_fsf = [e['fsf'] for e in data]
    all_nodes = [e['node_count'] for e in data]
    all_failed_nodes = [e['failed_node_count'] for e in data]
    all_edges = [e['edge_count'] for e in data]
    
    # Count thoughts with non-zero FSF (across all runs/models for same thought)
    thoughts_with_nonzero_fsf = 0
    total_thoughts = len(thought_groups)
    
    for key, entries in thought_groups.items():
        # Check if any entry in this thought group has FSF > 0
        if any(e['fsf'] > 0 for e in entries):
            thoughts_with_nonzero_fsf += 1
    
    thoughts_with_nonzero_fsf_percentage = (thoughts_with_nonzero_fsf / total_thoughts * 100) if total_thoughts > 0 else 0
    
    # Count questions with non-zero FSF (across all thoughts/runs for same question)
    questions_with_nonzero_fsf = 0
    total_questions = len(question_groups)
    
    for key, entries in question_groups.items():
        # Check if any entry in this question group has FSF > 0
        if any(e['fsf'] > 0 for e in entries):
            questions_with_nonzero_fsf += 1
    
    questions_with_nonzero_fsf_percentage = (questions_with_nonzero_fsf / total_questions * 100) if total_questions > 0 else 0
    
    overall_stats = {
        'total_entries': len(data),
        'unique_thoughts': len(thought_model_groups),
        'thoughts_with_3plus_runs': len(thought_model_stats),
        'thoughts_with_nonzero_fsf': thoughts_with_nonzero_fsf,
        'thoughts_with_nonzero_fsf_percentage': thoughts_with_nonzero_fsf_percentage,
        'total_unique_thoughts': total_thoughts,
        'questions_with_nonzero_fsf': questions_with_nonzero_fsf,
        'questions_with_nonzero_fsf_percentage': questions_with_nonzero_fsf_percentage,
        'total_unique_questions': total_questions,
        'fsf_mean': np.mean(all_fsf),
        'fsf_std': np.std(all_fsf),
        'fsf_min': np.min(all_fsf),
        'fsf_max': np.max(all_fsf),
        'fsf_median': np.median(all_fsf),
        'node_count_mean': np.mean(all_nodes),
        'failed_node_count_mean': np.mean(all_failed_nodes),
        'edge_count_mean': np.mean(all_edges)
    }
    
    return overall_stats, thought_model_stats


def plot_fsf_distribution(data: List[Dict[str, Any]], output_dir: Path):
    """Plot overall FSF distribution."""
    fsf_values = [e['fsf'] for e in data]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    axes[0].hist(fsf_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Flow Structure Factor (FSF)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('FSF Distribution (Histogram)')
    
    # Box plot
    axes[1].boxplot(fsf_values, vert=True)
    axes[1].set_ylabel('FSF')
    axes[1].set_title('FSF Distribution (Box Plot)')
    
    # Statistics text
    mean_val = np.mean(fsf_values)
    median_val = np.median(fsf_values)
    std_val = np.std(fsf_values)
    
    stats_text = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
    fig.text(0.15, 0.85, stats_text, transform=fig.transFigure,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'fsf_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FSF distribution plot to {output_path}")


def plot_fsf_std_distribution(thought_model_stats: List[Dict[str, Any]], output_dir: Path):
    """Plot distribution of FSF std for thoughts with >=3 runs."""
    if not thought_model_stats:
        print("No thoughts with >=3 runs found, skipping std distribution plot")
        return
    
    std_values = [s['fsf_std'] for s in thought_model_stats]
    
    plt.figure(figsize=(12, 6))
    plt.hist(std_values, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Standard Deviation of FSF')
    plt.ylabel('Count')
    plt.title(f'Distribution of FSF Std (Thoughts with ≥3 runs, n={len(thought_model_stats)})')
    
    # Add statistics text
    mean_std = np.mean(std_values)
    median_std = np.median(std_values)
    
    stats_text = f'Mean Std: {mean_std:.4f}\nMedian Std: {median_std:.4f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'fsf_std_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved FSF std distribution plot to {output_path}")


def plot_model_comparison(data: List[Dict[str, Any]], output_dir: Path):
    """Plot FSF distribution by model."""
    # Group by model
    model_fsf = defaultdict(list)
    for entry in data:
        model_fsf[entry['model']].append(entry['fsf'])
    
    models = list(model_fsf.keys())
    
    plt.figure(figsize=(12, 8))
    
    # Box plot by model
    fsf_by_model = [model_fsf[model] for model in models]
    bp = plt.boxplot(fsf_by_model, labels=models, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('FSF')
    plt.title('FSF Distribution by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / 'fsf_by_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison plot to {output_path}")


def generate_summary_report(
    overall_stats: Dict[str, Any],
    thought_model_stats: List[Dict[str, Any]],
    data: List[Dict[str, Any]],
    output_dir: Path
):
    """Generate a summary report of the analysis."""
    report_path = output_dir / 'fsf_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REASONING GRAPH FSF ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        for key, value in overall_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nFSF DEFINITION\n")
        f.write("-" * 40 + "\n")
        f.write("FSF (Failed-Step Fraction) = failed_nodes / total_nodes\n")
        f.write("where failed_nodes have fillcolor=lightpink (unsuccessful attempts)\n\n")
        
        f.write(f"\nTHOUGHTS WITH NON-ZERO FSF\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total unique thoughts: {overall_stats['total_unique_thoughts']}\n")
        f.write(f"Thoughts with non-zero FSF (at least one run): {overall_stats['thoughts_with_nonzero_fsf']}\n")
        f.write(f"Percentage of thoughts with non-zero FSF: {overall_stats['thoughts_with_nonzero_fsf_percentage']:.2f}%\n")
        f.write(f"Thoughts with all-zero FSF: {overall_stats['total_unique_thoughts'] - overall_stats['thoughts_with_nonzero_fsf']}\n")
        f.write(f"Percentage of thoughts with all-zero FSF: {100 - overall_stats['thoughts_with_nonzero_fsf_percentage']:.2f}%\n\n")
        
        f.write(f"\nQUESTIONS WITH NON-ZERO FSF\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total unique questions: {overall_stats['total_unique_questions']}\n")
        f.write(f"Questions with non-zero FSF (at least one thought has FSF > 0): {overall_stats['questions_with_nonzero_fsf']}\n")
        f.write(f"Percentage of questions with non-zero FSF: {overall_stats['questions_with_nonzero_fsf_percentage']:.2f}%\n")
        f.write(f"Questions with all-zero FSF: {overall_stats['total_unique_questions'] - overall_stats['questions_with_nonzero_fsf']}\n")
        f.write(f"Percentage of questions with all-zero FSF: {100 - overall_stats['questions_with_nonzero_fsf_percentage']:.2f}%\n\n")
        
        f.write(f"\nTHOUGHTS WITH ≥3 RUNS (STD ANALYSIS)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total thoughts with ≥3 runs: {len(thought_model_stats)}\n")
        
        if thought_model_stats:
            # Sort by std (descending) to show most variable
            sorted_by_std = sorted(thought_model_stats, key=lambda x: x['fsf_std'], reverse=True)
            
            f.write("\nTop 10 thoughts with highest FSF std:\n")
            for i, stat in enumerate(sorted_by_std[:10], 1):
                f.write(f"{i}. q_id={stat['q_id']}, t_id={stat['t_id']}, model={stat['model']}\n")
                f.write(f"   Runs: {stat['run_count']}, Mean FSF: {stat['fsf_mean']:.6f}, Std: {stat['fsf_std']:.6f}\n")
                f.write(f"   Avg Nodes: {stat['node_count_mean']:.1f}, Avg Failed Nodes: {stat['failed_node_count_mean']:.1f}\n")
            
            # Show model-wise breakdown
            model_stats = {}
            for stat in thought_model_stats:
                model = stat['model']
                if model not in model_stats:
                    model_stats[model] = []
                model_stats[model].append(stat['fsf_std'])
            
            f.write(f"\nMODEL-WISE STD STATISTICS\n")
            f.write("-" * 40 + "\n")
            for model, stds in model_stats.items():
                f.write(f"{model}:\n")
                f.write(f"  Count: {len(stds)}\n")
                f.write(f"  Mean Std: {np.mean(stds):.6f}\n")
                f.write(f"  Median Std: {np.median(stds):.6f}\n")
                f.write(f"  Max Std: {np.max(stds):.6f}\n\n")
        
        # Model distribution
        model_counts = {}
        model_fsf_means = defaultdict(list)
        for entry in data:
            model = entry['model']
            model_counts[model] = model_counts.get(model, 0) + 1
            model_fsf_means[model].append(entry['fsf'])
        
        f.write(f"MODEL DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            mean_fsf = np.mean(model_fsf_means[model])
            f.write(f"{model}: {count} entries (mean FSF: {mean_fsf:.6f})\n")
    
    print(f"Saved summary report to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze reasoning graphs FSF metrics (Fast version)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to rg_info.jsonl file')
    parser.add_argument('--output-dir', type=str, default='rg_plot',
                        help='Output directory for plots (default: rg_plot)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    print(f"Using {cpu_count()} CPU cores")
    
    # Parse DOT graphs using multiprocessing
    data = load_and_parse_rg_data_fast(args.input, args.workers)
    
    if not data:
        print("No valid data found!")
        return
    
    print("Calculating FSF statistics...")
    overall_stats, thought_model_stats = calculate_fsf_statistics(data)
    
    print("\n" + "=" * 60)
    print("OVERALL FSF STATISTICS")
    print("=" * 60)
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n📊 THOUGHTS WITH NON-ZERO FSF STATISTICS")
    print("-" * 60)
    print(f"Total unique thoughts: {overall_stats['total_unique_thoughts']}")
    print(f"Thoughts with non-zero FSF (at least one run): {overall_stats['thoughts_with_nonzero_fsf']}")
    print(f"Percentage: {overall_stats['thoughts_with_nonzero_fsf_percentage']:.2f}%")
    print(f"Thoughts with all-zero FSF: {overall_stats['total_unique_thoughts'] - overall_stats['thoughts_with_nonzero_fsf']}")
    print(f"All-zero percentage: {100 - overall_stats['thoughts_with_nonzero_fsf_percentage']:.2f}%")
    
    print(f"\n📊 QUESTIONS WITH NON-ZERO FSF STATISTICS")
    print("-" * 60)
    print(f"Total unique questions: {overall_stats['total_unique_questions']}")
    print(f"Questions with non-zero FSF (at least one thought has FSF > 0): {overall_stats['questions_with_nonzero_fsf']}")
    print(f"Percentage: {overall_stats['questions_with_nonzero_fsf_percentage']:.2f}%")
    print(f"Questions with all-zero FSF: {overall_stats['total_unique_questions'] - overall_stats['questions_with_nonzero_fsf']}")
    print(f"All-zero percentage: {100 - overall_stats['questions_with_nonzero_fsf_percentage']:.2f}%")
    
    print(f"\nFound {len(thought_model_stats)} thoughts with ≥3 runs")
    
    print("\nGenerating visualizations...")
    plot_fsf_distribution(data, output_dir)
    plot_fsf_std_distribution(thought_model_stats, output_dir)
    plot_model_comparison(data, output_dir)
    
    print("Generating summary report...")
    generate_summary_report(overall_stats, thought_model_stats, data, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print(f"📊 Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
"""
python reasoning_graph/analyze_rg_fsf.py \
    --input /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/rg_info.jsonl \
    --output-dir  /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/rg_plot \
    --workers 64
"""

"""
python reasoning_graph/analyze_rg_fsf.py \
    --input /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/rg_info.jsonl \
    --output-dir  /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/rg_plot \
    --workers 64
"""