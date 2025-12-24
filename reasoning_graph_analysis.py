#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Graph Analysis for Omnithought Math Dataset
Extract reasoning graphs from chain-of-thought data and calculate Failed-Step Fraction (FSF)
Based on the paper: "What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT"
"""

import json
import re
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tqdm import tqdm
import graphviz


class ReasoningStep:
    """Represents a single reasoning step extracted from CoT"""
    def __init__(self, content: str, step_id: int, is_failed: bool = False,
                 is_review: bool = False, depth: int = 0):
        self.content = content.strip()
        self.step_id = step_id
        self.is_failed = is_failed
        self.is_review = is_review
        self.depth = depth
        self.children = []
        self.parent = None

    def __repr__(self):
        status = "FAILED" if self.is_failed else "SUCCESS"
        review = " REVIEW" if self.is_review else ""
        return f"Step {self.step_id}: [{status}{review}] {self.content[:50]}..."



class ReasoningGraphExtractor:
    """Extract reasoning graphs from CoT text using pattern-based heuristics"""

    def __init__(self):
        # Patterns for detecting reasoning structure
        self.step_patterns = [
            r'(?:Step|step)\s*(\d+)[.:]\s*(.+)',
            r'(?:First|Second|Third|Fourth|Fifth|Next|Then|Finally)[,:]\s*(.+)',
            r'(?:Let\'s|Let us|We\s+(?:can|should|need\s+to))\s*(.+)',
            r'(?:Alternatively|Another\s+approach|Or)\s*(.+)',
            r'(?:However|But|Wait|Hmm)\s*(.+)',
            r'(?:Check|Verify|Test)\s*(.+)',
            r'<think>(.*?)</think>',
        ]

        # Patterns for detecting failed attempts
        self.failure_patterns = [
            r'(?:This\s+(?:doesn\'t|does\s+not)\s+work)',
            r'(?:This\s+is\s+(?:wrong|incorrect))',
            r'(?:Let\'s\s+try\s+(?:a\s+)?different)',
            r'(?:I\s+(?:made\s+an?|need\s+to\s+correct))',
            r'(?:Go\s+back|Rewind|Start\s+over)',
            r'(?:Abandon|Discard|Forget\s+this)',
            r'(?:Contradiction|Contradicts)',
        ]

        # Patterns for detecting review behaviors
        self.review_patterns = [
            r'(?:Check|Verify|Test|Review|Re-examine)',
            r'(?:Let\'s\s+double-check)',
            r'(?:Confirm|Make\s+sure)',
            r'(?:As\s+we\s+saw|As\s+mentioned)',
        ]

    def extract_steps(self, text: str) -> List[ReasoningStep]:
        """Extract reasoning steps from CoT text"""
        steps = []
        current_depth = 0

        # Clean the text
        text = re.sub(r'\n+', '\n', text.strip())

        # Try to extract from <think> tags first
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            reasoning_text = think_match.group(1)
        else:
            reasoning_text = text

        # Split into paragraphs
        paragraphs = [p.strip() for p in reasoning_text.split('\n') if p.strip()]

        step_id = 0
        for para in paragraphs:
            # Check if this paragraph represents a step
            is_step = False
            step_content = para

            for pattern in self.step_patterns:
                match = re.match(pattern, para, re.IGNORECASE | re.DOTALL)
                if match:
                    is_step = True
                    if len(match.groups()) > 1:
                        step_content = match.group(2)
                    else:
                        step_content = match.group(1)
                    break

            # If no explicit step marker, but paragraph is substantial, treat as step
            if not is_step and len(para.split()) > 5:
                is_step = True

            if is_step:
                # Detect if this is a failed step
                is_failed = any(re.search(pattern, para, re.IGNORECASE)
                               for pattern in self.failure_patterns)

                # Detect if this is review behavior
                is_review = any(re.search(pattern, para, re.IGNORECASE)
                               for pattern in self.review_patterns)

                # Adjust depth based on indentation or markers
                if re.match(r'.*[>:]', para[:20]):
                    current_depth += 1
                elif re.match(r'.*<', para[:20]):
                    current_depth = max(0, current_depth - 1)

                step = ReasoningStep(
                    content=step_content,
                    step_id=step_id,
                    is_failed=is_failed,
                    is_review=is_review,
                    depth=current_depth
                )
                steps.append(step)
                step_id += 1

        return steps

    def build_graph(self, steps: List[ReasoningStep]) -> nx.DiGraph:
        """Build a directed graph from reasoning steps"""
        G = nx.DiGraph()

        # Add nodes
        for step in steps:
            G.add_node(step.step_id,
                      content=step.content[:100],
                      is_failed=step.is_failed,
                      is_review=step.is_review,
                      depth=step.depth)

        # Add edges based on sequential flow and depth changes
        for i, step in enumerate(steps):
            if i < len(steps) - 1:
                next_step = steps[i + 1]
                # Sequential flow
                G.add_edge(step.step_id, next_step.step_id, type='sequential')

                # Backtracking edges for failed steps
                if step.is_failed and next_step.depth < step.depth:
                    G.add_edge(step.step_id, next_step.step_id, type='backtrack')

        return G

    def calculate_metrics(self, steps: List[ReasoningStep], G: nx.DiGraph) -> Dict:
        """Calculate reasoning metrics including FSF"""
        if not steps:
            return {
                'total_steps': 0,
                'failed_steps': 0,
                'fsf': 0.0,
                'review_ratio': 0.0,
                'max_depth': 0,
                'branches': 0
            }

        total_steps = len(steps)
        failed_steps = sum(1 for step in steps if step.is_failed)
        review_steps = sum(1 for step in steps if step.is_review)

        # Failed-Step Fraction (FSF)
        fsf = failed_steps / total_steps if total_steps > 0 else 0.0

        # Review ratio
        review_ratio = review_steps / total_steps if total_steps > 0 else 0.0

        # Graph metrics
        max_depth = max((step.depth for step in steps), default=0)
        branches = len([n for n in G.nodes() if G.in_degree(n) > 1])

        return {
            'total_steps': total_steps,
            'failed_steps': failed_steps,
            'fsf': fsf,
            'review_ratio': review_ratio,
            'max_depth': max_depth,
            'branches': branches,
            'is_solution_correct': self.check_solution_correctness(steps)
        }

    def check_solution_correctness(self, steps: List[ReasoningStep]) -> Optional[bool]:
        """Check if the final solution is correct (placeholder)"""
        # This is a simplified check - in practice, you'd compare with ground truth
        if not steps:
            return None

        final_step = steps[-1]
        content = final_step.content.lower()

        # Look for confidence indicators
        if any(word in content for word in ['\boxed', 'final answer', 'solution is']):
            # Check for uncertainty or contradiction
            if any(word in content for word in ['unsure', 'uncertain', 'contradiction']):
                return False
            return True

        return None


class ReasoningGraphVisualizer:
    """Visualize reasoning graphs with Graphviz"""

    def __init__(self, output_dir: str = './plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_graph_viz(self, G: nx.DiGraph, sample_id: str, metrics: Dict) -> str:
        """Create a Graphviz visualization of the reasoning graph"""
        dot = graphviz.Digraph(comment=f'Reasoning Graph for {sample_id}')
        dot.attr(rankdir='TB', size='12,8')

        # Add nodes with color coding
        for node_id, node_data in G.nodes(data=True):
            color = 'red' if node_data.get('is_failed', False) else 'lightblue'
            shape = 'box' if node_data.get('is_review', False) else 'ellipse'

            # Truncate content for display
            content = node_data.get('content', '')[:50] + '...' if len(str(node_data.get('content', ''))) > 50 else str(node_data.get('content', ''))

            dot.node(str(node_id), f"{node_id}\n{content}",
                    color=color, shape=shape, style='filled')

        # Add edges
        for u, v, edge_data in G.edges(data=True):
            edge_type = edge_data.get('type', 'sequential')
            if edge_type == 'backtrack':
                dot.edge(str(u), str(v), color='red', style='dashed')
            else:
                dot.edge(str(u), str(v))

        # Add metrics as comment
        metrics_text = f"FSF: {metrics['fsf']:.3f} | "
        metrics_text += f"Steps: {metrics['total_steps']} | "
        metrics_text += f"Failed: {metrics['failed_steps']} | "
        metrics_text += f"Max Depth: {metrics['max_depth']}"

        output_path = os.path.join(self.output_dir, f'reasoning_graph_{sample_id}')
        dot.render(output_path, format='png', cleanup=True)

        return f"{output_path}.png"

    def plot_metrics_distribution(self, all_metrics: List[Dict], output_file: str):
        """Plot distribution of metrics across all samples"""
        if not all_metrics:
            print("No metrics to plot")
            return

        fsf_values = [m['fsf'] for m in all_metrics]
        total_steps = [m['total_steps'] for m in all_metrics]
        review_ratios = [m['review_ratio'] for m in all_metrics]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Reasoning Graph Metrics Distribution', fontsize=16)

        # FSF distribution
        axes[0, 0].hist(fsf_values, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Failed-Step Fraction (FSF)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'FSF Distribution\nMean: {np.mean(fsf_values):.3f}')
        axes[0, 0].axvline(np.mean(fsf_values), color='darkred', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # Total steps distribution
        axes[0, 1].hist(total_steps, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Total Steps')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Steps Distribution\nMean: {np.mean(total_steps):.1f}')
        axes[0, 1].axvline(np.mean(total_steps), color='darkblue', linestyle='--', label='Mean')
        axes[0, 1].legend()

        # Review ratio distribution
        axes[1, 0].hist(review_ratios, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Review Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Review Ratio Distribution\nMean: {np.mean(review_ratios):.3f}')
        axes[1, 0].axvline(np.mean(review_ratios), color='darkgreen', linestyle='--', label='Mean')
        axes[1, 0].legend()

        # FSF vs Total Steps scatter plot
        axes[1, 1].scatter(total_steps, fsf_values, alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Total Steps')
        axes[1, 1].set_ylabel('Failed-Step Fraction')
        axes[1, 1].set_title('FSF vs Total Steps')

        # Add correlation coefficient
        if len(fsf_values) > 1 and len(total_steps) > 1:
            corr = np.corrcoef(total_steps, fsf_values)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                           transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics distribution plot saved to {output_file}")


def analyze_dataset(input_file: str, output_dir: str = './plots',
                   max_samples: int = 1000):
    """Main function to analyze the dataset"""

    extractor = ReasoningGraphExtractor()
    visualizer = ReasoningGraphVisualizer(output_dir)

    all_metrics = []
    graph_files = []

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing samples")):
            if i >= max_samples:
                break

            try:
                data = json.loads(line)

                # Extract reasoning data
                reasoning_data = data.get('reasoning', [{}])[0]
                cot_text = reasoning_data.get('full_response', '')

                if not cot_text:
                    continue

                # Extract steps
                steps = extractor.extract_steps(cot_text)

                # Build graph
                G = extractor.build_graph(steps)

                # Calculate metrics
                metrics = extractor.calculate_metrics(steps, G)
                metrics['sample_id'] = i
                all_metrics.append(metrics)

                # Create visualization for every 50th sample
                if i % 50 == 0 and i < 500:  # Limited to avoid too many files
                    graph_file = visualizer.create_graph_viz(G, f"sample_{i}", metrics)
                    graph_files.append(graph_file)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    print(f"\nProcessed {len(all_metrics)} samples")

    # Plot overall metrics distribution
    metrics_plot_file = os.path.join(output_dir, 'metrics_distribution.png')
    visualizer.plot_metrics_distribution(all_metrics, metrics_plot_file)

    # Save metrics summary
    summary_file = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    fsf_values = [m['fsf'] for m in all_metrics]
    print(f"Failed-Step Fraction (FSF):")
    print(f"  Mean: {np.mean(fsf_values):.3f}")
    print(f"  Median: {np.median(fsf_values):.3f}")
    print(f"  Std: {np.std(fsf_values):.3f}")

    total_steps = [m['total_steps'] for m in all_metrics]
    print(f"\nTotal Steps:")
    print(f"  Mean: {np.mean(total_steps):.1f}")
    print(f"  Median: {np.median(total_steps):.1f}")

    # Check correlation with correctness if available
    correct_mask = [m.get('is_solution_correct', None) for m in all_metrics if m.get('is_solution_correct') is not None]
    if correct_mask:
        correct_fsf = [m['fsf'] for m in all_metrics if m.get('is_solution_correct') is True]
        incorrect_fsf = [m['fsf'] for m in all_metrics if m.get('is_solution_correct') is False]

        if correct_fsf and incorrect_fsf:
            print(f"\nFSF by Correctness:")
            print(f"  Correct solutions mean FSF: {np.mean(correct_fsf):.3f}")
            print(f"  Incorrect solutions mean FSF: {np.mean(incorrect_fsf):.3f}")

    print(f"\nGenerated graphs: {len(graph_files)}")
    print(f"Metrics saved to: {summary_file}")

    return all_metrics


if __name__ == "__main__":
    # Set up paths
    input_file = "/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl"
    output_dir = "/home/luoyashuo/research/graph_kd/cdad/plots/reasoning_graphs"

    # Analyze dataset
    print("Starting Reasoning Graph Analysis...")
    metrics = analyze_dataset(input_file, output_dir, max_samples=1000)

    print("\nAnalysis complete!")