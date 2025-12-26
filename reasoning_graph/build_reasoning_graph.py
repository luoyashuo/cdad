#!/usr/bin/env python3
"""
Build reasoning graphs from COT data and compute FSF (Failed-Step Fraction).

This script reads a JSONL dataset, uses an LLM to convert each reasoning trace
into a Graphviz DOT representation, computes the FSF metric, and saves the results.
Now supports multiple models and multiple runs per model, with automatic stability analysis.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm as async_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available, progress bar disabled")

# Add parent directory to path to import rollout.py
sys.path.append(str(Path(__file__).parent.parent))

from rollout import AsyncConsulOpenAIProxy


# Prompt for generating Graphviz DOT code from reasoning trace
GRAPH_GENERATION_PROMPT = """
Parse the reasoning trace into a Graphviz diagram. Focus on these essentials:

Node Rules:
- One node per distinct reasoning step
- 'fillcolor=lightblue': Successful reasoning steps
- 'fillcolor=lightpink': Failed attempts

Edge Rules:
- Connect node A → node B if the information or insight from A is actually used to construct the reasoning in B; branch new attempts from their starting ancestor, not from dead ends.

Requirements:
- Use 'rankdir=TB'
- Include ALL attempts (including failures), do not miss any steps in the reasoning.
- ALWAYS start with a "problem statement" node
- ALWAYS end with a "final answer" node
- Do NOT reorder or reorganize the reasoning flow

Generate complete Graphviz DOT code in dot blocks.
""".strip()


class ReasoningGraphBuilder:
    def __init__(
        self,
        models: List[str] = None,
        runs_per_model: int = 1,
        timeout: float = 600.0,
        max_connections_per_service: int = 256,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_concurrent_requests: int = 10
    ):
        # If single model is provided, convert to list
        if models is None:
            models = ["gpt-oss-120b-judge"]
        elif isinstance(models, str):
            models = [models]

        self.models = models
        self.runs_per_model = runs_per_model
        self.timeout = timeout
        self.max_connections_per_service = max_connections_per_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Initialize LLM clients for each model
        self.llm_clients = {}
        for model in self.models:
            self.llm_clients[model] = AsyncConsulOpenAIProxy(
                model=model,
                timeout=timeout,
                max_connections_per_service=max_connections_per_service
            )

    async def close(self):
        """Close all LLM clients."""
        for client in self.llm_clients.values():
            await client.close()

    async def generate_reasoning_graph(self, thought: str, model: str) -> Optional[str]:
        """
        Generate Graphviz DOT code for a reasoning trace using LLM.

        Args:
            thought: The reasoning trace (COT) to convert to graph
            model: The model to use for generation

        Returns:
            Graphviz DOT code as string, or None if generation fails
        """
        if not thought or not thought.strip():
            return None

        if model not in self.llm_clients:
            print(f"Warning: Model {model} not found")
            return None

        messages_for_llm = [
            {"role": "system", "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."},
            {"role": "user", "content": f"{GRAPH_GENERATION_PROMPT}\n\nReasoning trace:\n{thought}"}
        ]

        async with self.semaphore:
            try:
                result = await self.llm_clients[model].chat_completions(
                    messages=messages_for_llm,
                    n=1,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if not result or 'choices' not in result or not result['choices']:
                    print(f"Warning: Empty result from LLM")
                    return None

                content = result['choices'][0]['message']['content']

                # Extract DOT code from the response (look for ```dot ... ``` blocks)
                dot_match = re.search(r'```(?:dot\n)?(.*?)```', content, re.DOTALL)
                if dot_match:
                    return dot_match.group(1).strip()
                else:
                    # If no code block found, return the entire content
                    return content.strip()

            except Exception as e:
                print(f"Error generating graph with model {model}: {e}")
                return None

    def compute_fsf_from_dot(self, dot_code: str) -> Optional[float]:
        """
        Compute Failed-Step Fraction (FSF) from Graphviz DOT code.

        FSF = (# failed nodes) / (# all nodes)

        Args:
            dot_code: Graphviz DOT code

        Returns:
            FSF value between 0 and 1, or None if parsing fails
        """
        if not dot_code:
            return None

        try:
            # Count total nodes
            total_nodes = len(re.findall(r'\[.*?\]', dot_code, re.DOTALL))

            # Count failed nodes (nodes with fillcolor=lightpink)
            failed_nodes = len(re.findall(r'fillcolor\s*=\s*["\']?lightpink["\']?', dot_code, re.IGNORECASE))

            if total_nodes == 0:
                return None

            return failed_nodes / total_nodes

        except Exception as e:
            print(f"Error computing FSF: {e}")
            return None


async def process_single_entry(entry: Dict[str, Any], builder: ReasoningGraphBuilder, entry_idx: int) -> Dict[str, Any]:
    """
    Process a single dataset entry by generating reasoning graphs and computing FSF for all reasoning traces
    with multiple models and multiple runs per model.

    Args:
        entry: The dataset entry
        builder: ReasoningGraphBuilder instance
        entry_idx: Index of the entry for progress tracking

    Returns:
        Modified entry with reasoning_graph and fsf fields added
    """
    if 'reasoning' not in entry or not entry['reasoning']:
        return entry

    # Process each reasoning trace
    for i, reasoning_item in enumerate(entry['reasoning']):
        thought = reasoning_item.get('thought', '')

        if not thought:
            continue

        print(f"Processing entry {entry_idx}, reasoning {i}...")

        reasoning_results = []

        # Run for each model and each run
        for model in builder.models:
            for run in range(builder.runs_per_model):
                print(f"  Using model {model}, run {run + 1}/{builder.runs_per_model}")

                # Generate reasoning graph
                graph_dot = await builder.generate_reasoning_graph(thought, model)

                if graph_dot:
                    # Compute FSF
                    fsf = builder.compute_fsf_from_dot(graph_dot)

                    # Store results
                    result = {
                        'model_name': model,
                        'run_id': run + 1,
                        'reasoning_graph': graph_dot,
                        'fsf': fsf
                    }
                    reasoning_results.append(result)
                    print(f"    Generated graph with FSF={fsf}")
                else:
                    print(f"    Failed to generate graph")
                    result = {
                        'model_name': model,
                        'run_id': run + 1,
                        'reasoning_graph': None,
                        'fsf': None
                    }
                    reasoning_results.append(result)

            await asyncio.sleep(1)  # Small delay between models

        # Add all results to the reasoning item
        reasoning_item['reasoning_graph'] = reasoning_results

    return entry


def compute_fsf_stability_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute FSF stability metrics for each data entry.

    1. Same-model FSF standard deviation (within-model consistency)
    2. Cross-model mean difference (between-model variance)

    Args:
        entries: List of processed entries with reasoning_graph results

    Returns:
        Dictionary mapping entry indices to stability metrics
    """
    stability_metrics = {}

    for entry_idx, entry in enumerate(entries):
        if 'reasoning' not in entry or not entry['reasoning']:
            continue

        entry_metrics = {}

        for reasoning_idx, reasoning_item in enumerate(entry['reasoning']):
            if ('reasoning_graph' not in reasoning_item or
                not isinstance(reasoning_item['reasoning_graph'], list)):
                continue

            all_results = reasoning_item['reasoning_graph']

            # Group results by model
            model_fsf_results = defaultdict(list)
            for result in all_results:
                if result['fsf'] is not None:
                    model_fsf_results[result['model_name']].append(result['fsf'])

            # Compute within-model standard deviation
            within_model_std = {}
            for model, fsf_values in model_fsf_results.items():
                if len(fsf_values) > 1:
                    within_model_std[model] = statistics.stdev(fsf_values)
                else:
                    within_model_std[model] = 0.0

            # Compute between-model mean difference
            model_means = {}
            for model, fsf_values in model_fsf_results.items():
                if fsf_values:
                    model_means[model] = statistics.mean(fsf_values)

            cross_model_mean_diff = None
            if len(model_means) > 1:
                all_means = list(model_means.values())
                # Compute mean of absolute differences between all pairs of models
                differences = []
                for i in range(len(all_means)):
                    for j in range(i + 1, len(all_means)):
                        differences.append(abs(all_means[i] - all_means[j]))
                cross_model_mean_diff = statistics.mean(differences) if differences else None

            # Store metrics for this reasoning
            entry_metrics[f'reasoning_{reasoning_idx}'] = {
                'within_model_std': within_model_std,
                'cross_model_mean_diff': cross_model_mean_diff,
                'model_means': model_means,
                'all_results': model_fsf_results
            }

        if entry_metrics:
            stability_metrics[f'entry_{entry_idx}'] = entry_metrics

    return stability_metrics


def plot_fsf_stability(stability_metrics: Dict[str, Any], output_dir: str = 'fsf_stability_plots'):
    """
    Generate visualization plots for FSF stability metrics.

    Args:
        stability_metrics: Stability metrics computed by compute_fsf_stability_metrics
        output_dir: Directory to save the plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    entry_names = list(stability_metrics.keys())
    reasoning_data = defaultdict(lambda: {
        'within_model_stds': defaultdict(list),
        'cross_model_diffs': []
    })

    # Extract data
    for entry_key, entry_data in stability_metrics.items():
        for reasoning_key, reasoning_data_dict in entry_data.items():
            # Collect within-model standard deviations
            for model, std in reasoning_data_dict['within_model_std'].items():
                reasoning_data[reasoning_key]['within_model_stds'][model].append(std)

            # Collect cross-model mean differences
            cross_model_diff = reasoning_data_dict['cross_model_mean_diff']
            if cross_model_diff is not None:
                reasoning_data[reasoning_key]['cross_model_diffs'].append(cross_model_diff)

    # Plot 1: Within-model FSF standard deviation
    plt.figure(figsize=(12, 8))
    for reasoning_key, data in reasoning_data.items():
        for model, stds in data['within_model_stds'].items():
            plt.subplot(2, 1, 1)
            plt.scatter([f"{reasoning_key}-{model}"] * len(stds), stds, label=f"{reasoning_key}-{model}", alpha=0.7)

    plt.title('Within-Model FSF Standard Deviation')
    plt.ylabel('FSF Std Dev')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'within_model_std.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Between-model mean differences
    plt.figure(figsize=(10, 6))
    all_cross_model_diffs = []
    labels = []
    for reasoning_key, data in reasoning_data.items():
        if data['cross_model_diffs']:
            all_cross_model_diffs.extend(data['cross_model_diffs'])
            labels.extend([reasoning_key] * len(data['cross_model_diffs']))

    if all_cross_model_diffs:
        plt.scatter(labels, all_cross_model_diffs, alpha=0.7)
        plt.title('Between-Model Mean FSF Differences')
        plt.ylabel('Mean Absolute Difference')
        plt.xlabel('Reasoning Instance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'between_model_diffs.png', dpi=300, bbox_inches='tight')
    else:
        print("Warning: No cross-model differences to plot")
    plt.close()

    # Summary statistics plot
    plt.figure(figsize=(12, 6))

    # Summary for within-model std
    plt.subplot(1, 2, 1)
    all_within_stds = []
    model_names = []
    for data in reasoning_data.values():
        for model, stds in data['within_model_stds'].items():
            if stds:
                all_within_stds.append(statistics.mean(stds))
                model_names.append(model)

    if all_within_stds:
        plt.boxplot(all_within_stds, labels=['All Models'])
        plt.title('Distribution of Within-Model\nFSF Standard Deviations')
        plt.ylabel('FSF Std Dev')

    # Summary for between-model differences
    plt.subplot(1, 2, 2)
    all_cross_diffs = []
    for data in reasoning_data.values():
        all_cross_diffs.extend(data['cross_model_diffs'])

    if all_cross_diffs:
        plt.boxplot(all_cross_diffs, labels=['Between-Model'])
        plt.title('Distribution of Between-Model\nMean FSF Differences')
        plt.ylabel('Mean Absolute Difference')

    plt.tight_layout()
    plt.savefig(output_path / 'fsf_stability_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_path.absolute()}")


def save_stability_metrics(stability_metrics: Dict[str, Any], output_file: str):
    """
    Save stability metrics to a JSON file.

    Args:
        stability_metrics: Stability metrics computed by compute_fsf_stability_metrics
        output_file: Path to output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stability_metrics, f, indent=2, ensure_ascii=False)
    print(f"Stability metrics saved to {output_file}")


def load_and_analyze_results(results_file: str, output_dir: str = 'fsf_stability_plots'):
    """
    Load processed results from a JSONL file and perform stability analysis.

    Args:
        results_file: Path to the processed JSONL file
        output_dir: Directory to save analysis plots and metrics

    Returns:
        Stability metrics dictionary
    """
    # Read the results
    entries = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {results_file}")

    # Compute stability metrics
    stability_metrics = compute_fsf_stability_metrics(entries)

    # Save stability metrics to JSON
    metrics_file = Path(output_dir) / 'stability_metrics.json'
    save_stability_metrics(stability_metrics, str(metrics_file))

    # Generate plots
    plot_fsf_stability(stability_metrics, output_dir)

    return stability_metrics


async def process_dataset(
    input_path: str,
    output_path: str,
    builder: ReasoningGraphBuilder,
    start_idx: int = 0,
    end_idx: Optional[int] = None
):
    """
    Process a dataset file and generate reasoning graphs with FSF scores.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        builder: ReasoningGraphBuilder instance
        start_idx: Starting index for processing (inclusive)
        end_idx: Ending index for processing (exclusive)
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read all entries
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    total_entries = len(entries)
    print(f"Loaded {total_entries} entries from {input_path}")

    # Apply index range
    if end_idx is None:
        end_idx = total_entries
    entries_to_process = entries[start_idx:end_idx]
    print(f"Processing entries {start_idx} to {end_idx} ({len(entries_to_process)} entries)")

    # Process entries concurrently
    tasks = []
    for idx, entry in enumerate(entries_to_process, start=start_idx):
        task = process_single_entry(entry, builder, idx)
        tasks.append(task)

    # Wait for all tasks to complete with progress bar
    if HAS_TQDM:
        processed_entries = []
        for task in async_tqdm.as_completed(tasks, total=len(tasks), desc="Processing entries"):
            processed_entries.append(await task)
    else:
        processed_entries = await asyncio.gather(*tasks)

    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nResults saved to {output_path}")
    print(f"Total entries processed: {len(processed_entries)}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Build reasoning graphs and compute FSF from COT data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output JSONL file')
    parser.add_argument('--models', type=str, nargs='+', default=['gpt-oss-120b-judge'],
                       help='Model names for LLM API (can specify multiple)')
    parser.add_argument('--runs-per-model', type=int, default=1,
                       help='Number of runs per model for each COT')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index for processing')
    parser.add_argument('--end-idx', type=int, default=None,
                       help='Ending index for processing')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum concurrent requests')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for LLM generation')
    parser.add_argument('--max-tokens', type=int, default=4096,
                       help='Max tokens for LLM generation')
    parser.add_argument('--analyze', action='store_true',
                       help='Skip processing and only analyze existing results')
    parser.add_argument('--plot-dir', type=str, default='fsf_stability_plots',
                       help='Directory to save stability plots')

    args = parser.parse_args()

    # Set default plot directory relative to output file if not specified
    if args.plot_dir == 'fsf_stability_plots' and args.output:
        output_path = Path(args.output)
        args.plot_dir = str(output_path.parent / f"{output_path.stem}_plots")

    # If analyze mode, load and analyze existing results
    if args.analyze:
        print(f"Analyzing existing results from {args.output}")
        metrics = load_and_analyze_results(args.output, args.plot_dir)
        print(f"Analysis complete. Plots saved to {args.plot_dir}")
        return

    # Create builder
    builder = ReasoningGraphBuilder(
        models=args.models,
        runs_per_model=args.runs_per_model,
        max_concurrent_requests=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    try:
        # Process dataset
        await process_dataset(
            input_path=args.input,
            output_path=args.output,
            builder=builder,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )

        # Automatically analyze results after processing
        print("\nAnalyzing results for FSF stability...")
        metrics = load_and_analyze_results(args.output, args.plot_dir)
        print(f"Analysis complete. Plots saved to {args.plot_dir}")
    finally:
        await builder.close()


if __name__ == '__main__':
    asyncio.run(main())

"""
python3 reasoning_graph/build_reasoning_graph.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train_rg.jsonl \
    --models gpt-oss-120b-judge Qwen2-5-72B-Instruct-20250314 \
    --runs-per-model 5 \
    --max-concurrent 10 \
    --temperature 0.3 \
    --max-tokens 8192 \
    --start-idx 0 \
    --end-idx 50
"""
