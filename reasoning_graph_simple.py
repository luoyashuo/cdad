#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Reasoning Graph Analysis for Omnithought Math Dataset
Extract reasoning graphs from chain-of-thought data and calculate Failed-Step Fraction (FSF)
Based on the paper: "What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT"
"""

import json
import re
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm


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

        if not steps and reasoning_text:
            # If no steps extracted, treat entire text as one step
            is_failed = any(re.search(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
                           for pattern in self.failure_patterns)
            is_review = any(re.search(pattern, reasoning_text, re.IGNORECASE | re.DOTALL)
                           for pattern in self.review_patterns)

            steps.append(ReasoningStep(
                content=reasoning_text,
                step_id=0,
                is_failed=is_failed,
                is_review=is_review,
                depth=0
            ))

        return steps

    def build_simple_graph(self, steps: List[ReasoningStep]) -> Dict:
        """Build a simple graph representation from reasoning steps"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'node_positions': {},
            'failed_steps': [],
            'review_steps': []
        }

        # Create nodes
        for step in steps:
            node_data = {
                'id': step.step_id,
                'content': step.content[:100],  # Truncate for display
                'is_failed': step.is_failed,
                'is_review': step.is_review,
                'depth': step.depth,
                'full_content': step.content
            }
            graph_data['nodes'].append(node_data)

            if step.is_failed:
                graph_data['failed_steps'].append(step.step_id)
            if step.is_review:
                graph_data['review_steps'].append(step.step_id)

        # Create edges (simple sequential flow + backtracking)
        for i, step in enumerate(steps):
            # Sequential flow
            if i < len(steps) - 1:
                next_step = steps[i + 1]
                if step.depth <= next_step.depth:
                    graph_data['edges'].append({
                        'from': step.step_id,
                        'to': next_step.step_id,
                        'type': 'sequential'
                    })

            # Backtracking edges for failed steps
            if step.is_failed:
                # Look for next step with lower depth
                for j in range(i + 1, len(steps)):
                    if steps[j].depth < step.depth:
                        graph_data['edges'].append({
                            'from': step.step_id,
                            'to': steps[j].step_id,
                            'type': 'backtrack'
                        })
                        break

        # Calculate positions for visualization
        self._calculate_positions(graph_data, steps)

        return graph_data

    def _calculate_positions(self, graph_data: Dict, steps: List[ReasoningStep]):
        """Calculate 2D positions for graph visualization"""
        positions = {}

        # Group by depth
        depth_groups = defaultdict(list)
        for step in steps:
            depth_groups[step.depth].append(step.step_id)

        # Position nodes
        for depth, node_ids in depth_groups.items():
            y_pos = -depth * 2  # Negative for downward layout
            for i, node_id in enumerate(node_ids):
                x_pos = i - len(node_ids) / 2  # Centered
                positions[node_id] = (x_pos, y_pos)

        graph_data['node_positions'] = positions

    def calculate_metrics(self, steps: List[ReasoningStep]) -> Dict:
        """Calculate reasoning metrics including FSF"""
        if not steps:
            return {
                'total_steps': 0,
                'failed_steps': 0,
                'fsf': 0.0,
                'review_ratio': 0.0,
                'max_depth': 0,
                'avg_step_length': 0.0
            }

        total_steps = len(steps)
        failed_steps = sum(1 for step in steps if step.is_failed)
        review_steps = sum(1 for step in steps if step.is_review)

        # Failed-Step Fraction (FSF) - the key metric from the paper
        fsf = failed_steps / total_steps if total_steps > 0 else 0.0

        # Review ratio
        review_ratio = review_steps / total_steps if total_steps > 0 else 0.0

        # Additional metrics
        max_depth = max((step.depth for step in steps), default=0)
        avg_step_length = np.mean([len(step.content.split()) for step in steps])

        # Check for solution correctness indicators
        has_boxed = any('\\boxed' in step.content for step in steps)

        return {
            'total_steps': total_steps,
            'failed_steps': failed_steps,
            'fsf': fsf,  # Key metric from the paper
            'review_ratio': review_ratio,
            'max_depth': max_depth,
            'avg_step_length': avg_step_length,
            'has_boxed_answer': has_boxed
        }


class ReasoningGraphVisualizer:
    """Visualize reasoning graphs using matplotlib"""

    def __init__(self, output_dir: str = './plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_graph_visualization(self, graph_data: Dict, sample_id: str, metrics: Dict) -> str:
        """Create a visualization of the reasoning graph"""
        plt.figure(figsize=(12, 8))

        # Extract data
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        positions = graph_data['node_positions']

        # Draw nodes
        for node in nodes:
            node_id = node['id']
            x, y = positions[node_id]

            # Color coding
            if node['is_failed']:
                color = 'red'
                alpha = 0.8
            else:
                color = 'lightblue'
                alpha = 0.7

            # Shape coding
            if node['is_review']:
                shape = 's'  # Square
                node_type = 'Review'
            else:
                shape = 'o'  # Circle
                node_type = 'Progress'

            # Draw node
            plt.scatter(x, y, c=color, marker=shape, s=500, alpha=alpha,
                       edgecolors='black', linewidth=2)

            # Add node label
            label = f"Step {node_id}:\n{node['content'][:40]}..."
            plt.text(x, y - 0.3, label, ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Draw edges
        for edge in edges:
            from_pos = positions[edge['from']]
            to_pos = positions[edge['to']]

            if edge['type'] == 'backtrack':
                plt.arrow(from_pos[0], from_pos[1],
                         to_pos[0] - from_pos[0], to_pos[1] - from_pos[1],
                         head_width=0.1, head_length=0.1, fc='red', ec='red',
                         linestyle='dashed', alpha=0.7)
            else:
                plt.arrow(from_pos[0], from_pos[1],
                         to_pos[0] - from_pos[0], to_pos[1] - from_pos[1],
                         head_width=0.08, head_length=0.08, fc='black', ec='black',
                         alpha=0.5)

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=10, label='Successful Progress'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue',
                      markersize=10, label='Successful Review'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Failed Progress'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=10, label='Failed Review'),
            plt.Line2D([0], [0], color='red', linestyle='dashed',
                      label='Backtracking'),
        ]

        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        # Add title with metrics
        title = (f"Reasoning Graph - Sample {sample_id}\n"
                f"FSF: {metrics['fsf']:.3f} | Total Steps: {metrics['total_steps']} | "
                f"Failed Steps: {metrics['failed_steps']} | Max Depth: {metrics['max_depth']}")
        plt.title(title, fontsize=14, fontweight='bold')

        plt.xlabel('Logical Progress', fontsize=12)
        plt.ylabel('Depth / Abstraction Level', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        output_file = os.path.join(self.output_dir, f'reasoning_graph_{sample_id}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

    def plot_sample_representations(self, samples_data: List[Dict], num_samples: int = 9):
        """Plot representative samples with different FSF values"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        # Select samples with varying FSF values
        samples_sorted = sorted(samples_data, key=lambda x: x['metrics']['fsf'])
        selected_indices = np.linspace(0, len(samples_sorted) - 1, num_samples, dtype=int)

        for i, idx in enumerate(selected_indices):
            sample = samples_sorted[idx]
            graph_data = sample['graph_data']
            metrics = sample['metrics']

            ax = axes[i]
            nodes = graph_data['nodes']
            positions = graph_data['node_positions']
            edges = graph_data['edges']

            # Draw nodes
            for node in nodes:
                node_id = node['id']
                x, y = positions[node_id]

                color = 'red' if node['is_failed'] else 'lightblue'
                shape = 's' if node['is_review'] else 'o'
                marker_size = 200

                ax.scatter(x, y, c=color, marker=shape, s=marker_size,
                          alpha=0.7, edgecolors='black', linewidth=1)

            # Draw edges
            for edge in edges:
                from_pos = positions[edge['from']]
                to_pos = positions[edge['to']]

                linestyle = 'dashed' if edge['type'] == 'backtrack' else 'solid'
                color = 'red' if edge['type'] == 'backtrack' else 'black'
                alpha = 0.5 if edge['type'] == 'backtrack' else 0.3

                ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                       color=color, linestyle=linestyle, alpha=alpha,
                       linewidth=1 if edge['type'] == 'backtrack' else 2)

            ax.set_title(f'Sample {i}\nFSF: {metrics["fsf"]:.3f}', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle('Representative Reasoning Graphs with Different FSF Values',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = os.path.join(self.output_dir, 'representative_samples.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return output_file

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

        # Add correlation coefficient if sufficient data
        if len(fsf_values) > 1 and len(total_steps) > 1:
            corr = np.corrcoef(total_steps, fsf_values)[0, 1]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                           transform=axes[1, 1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics distribution plot saved to {output_file}")



def analyze_dataset_simplified(input_file: str, output_dir: str = './plots/simple',
                              max_samples: int = 1000, create_graphs: bool = True):
    """Simplified main function to analyze the dataset"""

    extractor = ReasoningGraphExtractor()
    visualizer = ReasoningGraphVisualizer(output_dir)

    all_metrics = []
    samples_with_graphs = []
    processed_count = 0

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Processing samples", total=min(max_samples, 19800))):
            if processed_count >= max_samples:
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
                if not steps:
                    continue

                # Calculate metrics
                metrics = extractor.calculate_metrics(steps)
                metrics['sample_id'] = processed_count
                all_metrics.append(metrics)

                # Build simple graph
                if create_graphs and processed_count < 100:  # Limited graphs
                    graph_data = extractor.build_simple_graph(steps)
                    samples_with_graphs.append({
                        'sample_id': processed_count,
                        'graph_data': graph_data,
                        'metrics': metrics
                    })

                    # Create visualization
                    try:
                        graph_file = visualizer.create_graph_visualization(
                            graph_data, f"sample_{processed_count}", metrics)
                        # print(f"Created graph for sample {processed_count}")
                    except Exception as e:
                        print(f"Error creating graph for sample {processed_count}: {e}")

                processed_count += 1

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    print(f"\nSuccessfully processed {processed_count} samples")

    # Plot representative samples
    if samples_with_graphs:
        try:
            visualizer.plot_sample_representations(samples_with_graphs)
        except Exception as e:
            print(f"Error plotting representative samples: {e}")

    # Plot overall metrics distribution
    metrics_plot_file = os.path.join(output_dir, 'metrics_distribution.png')
    visualizer.plot_metrics_distribution(all_metrics, metrics_plot_file)

    # Save metrics summary
    summary_file = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    try:
        fsf_values = [m['fsf'] for m in all_metrics]
        print(f"Failed-Step Fraction (FSF):")
        print(f"  Mean: {np.mean(fsf_values):.3f}")
        print(f"  Median: {np.median(fsf_values):.3f}")
        print(f"  Std: {np.std(fsf_values):.3f}")
        print(f"  Min: {np.min(fsf_values):.3f}")
        print(f"  Max: {np.max(fsf_values):.3f}")

        # Analyze distribution
        low_fsf = sum(1 for f in fsf_values if f < 0.1)
        med_fsf = sum(1 for f in fsf_values if 0.1 <= f < 0.3)
        high_fsf = sum(1 for f in fsf_values if f >= 0.3)

        print(f"\nFSF Distribution:")
        print(f"  Low FSF (<0.1): {low_fsf} samples ({low_fsf/len(fsf_values)*100:.1f}%)")
        print(f"  Medium FSF (0.1-0.3): {med_fsf} samples ({med_fsf/len(fsf_values)*100:.1f}%)")
        print(f"  High FSF (≥0.3): {high_fsf} samples ({high_fsf/len(fsf_values)*100:.1f}%)")

        total_steps = [m['total_steps'] for m in all_metrics]
        print(f"\nTotal Steps:")
        print(f"  Mean: {np.mean(total_steps):.1f}")
        print(f"  Median: {np.median(total_steps):.1f}")
        print(f"  Min: {np.min(total_steps)}")
        print(f"  Max: {np.max(total_steps)}")

        # Correlation analysis
        if len(fsf_values) > 10:
            corr = np.corrcoef(total_steps, fsf_values)[0, 1]
            print(f"\nCorrelation between Total Steps and FSF: {corr:.3f}")

            # Check if longer reasoning has higher FSF (as expected from paper)
            if corr > 0:
                print("  → Longer reasoning tends to have more failed steps")
            else:
                print("  → Longer reasoning does not necessarily have more failed steps")

    except Exception as e:
        print(f"Error in statistical analysis: {e}")

    print(f"\nResults saved to: {output_dir}")
    print(f"Metrics summary: {summary_file}")

    return all_metrics


if __name__ == "__main__":
    # Set up paths
    input_file = "/mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl"
    output_dir = "/home/luoyashuo/research/graph_kd/cdad/plots/reasoning_graphs_simple"

    print("=" * 60)
    print("Reasoning Graph Analysis for Omnithought Math Dataset")
    print("Based on: 'What Characterizes Effective Reasoning? Revisiting Length, Review, and Structure of CoT'")
    print("=" * 60)

    print("\nStarting analysis...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Run smaller sample first for testing
    metrics = analyze_dataset_simplified(input_file, output_dir, max_samples=1000, create_graphs=True)

    print("\nAnalysis complete!")
    print(f"Processed {len(metrics)} reasoning traces\n")

    # Summary interpretation
    if metrics:
        print("Key Insights:")
        print("- Failed-Step Fraction (FSF) is calculated as failed_steps / total_steps")
        print("- Lower FSF indicates more effective reasoning (less failed attempts)")
        print("- Review steps are identified by keywords like 'check', 'verify', 'review'")
        print("- Failed steps are identified by keywords indicating wrong attempts or contradictions")
        print("\nGenerated visualizations show the logical flow of reasoning with:")
        print("  - Red nodes: Failed steps")
        print("  - Blue nodes: Successful steps")
        print("  - Square nodes: Review/verification steps")
        print("  - Circle nodes: Progress steps")
        print("  - Dashed arrows: Backtracking after failed attempts")