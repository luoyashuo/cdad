#!/usr/bin/env python3
"""
Generate reasoning graphs with thought-level concurrency.

This script:
1. Reads a JSONL dataset with reasoning traces
2. Creates fine-grained tasks: (thought_id, thought, model_name, run_id)
3. Processes tasks with high concurrency (thought-level parallelism)
4. Saves results immediately after each task completes
5. Outputs flattened format: (question, thought, q_id, t_id, reasoning_graph, model)

No analysis or plotting logic included.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm
from dataclasses import dataclass

# Try to import pydot for DOT parsing
try:
    import pydot
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False
    print("Error: pydot is required for DOT parsing")
    print("Install with: pip install pydot")
    sys.exit(1)

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


@dataclass
class Task:
    """Represents a single annotation task."""
    thought_id: str  # Format: "{q_id}_{t_id}"
    q_id: int        # Line number in input file (0-based)
    t_id: int        # Index in reasoning array
    question: str    # Original question
    thought: str     # Thought text
    model_name: str  # Model name
    run_id: int      # Run ID


class DOTValidator:
    """Utility class for validating and parsing DOT code using pydot."""

    @staticmethod
    def is_valid_dot(dot_code: str) -> Tuple[bool, Optional[str]]:
        """Validate if the given string is valid DOT code."""
        if not dot_code or not dot_code.strip():
            return False, "Empty DOT code"

        try:
            graphs = pydot.graph_from_dot_data(dot_code)
            if graphs is None or len(graphs) == 0:
                return False, "Failed to parse DOT code: no graphs returned"

            graph = graphs[0]
            if graph is None:
                return False, "Failed to parse DOT code: graph is None"

            return True, None

        except Exception as e:
            return False, f"DOT parsing error: {str(e)}"

    @staticmethod
    def parse_dot(dot_code: str) -> Optional[Any]:
        """Parse DOT code into a pydot graph object."""
        try:
            graphs = pydot.graph_from_dot_data(dot_code)
            if graphs and len(graphs) > 0 and graphs[0] is not None:
                return graphs[0]
            return None
        except Exception:
            return None
    @staticmethod
    def is_valid_dot_strict(dot_code: str) -> Tuple[bool, Optional[str]]:
        """Strict validation - checks if DOT code has actual content."""
        if not dot_code or not dot_code.strip():
            return False, "Empty DOT code"
        
        # Check for basic structure (edges -> or node definitions [])
        if '->' not in dot_code and '[' not in dot_code:
            return False, "No edges (->) or node attributes ([...]) found"
        
        try:
            graphs = pydot.graph_from_dot_data(dot_code)
            if graphs is None or len(graphs) == 0:
                return False, "No graphs returned from parser"
            
            graph = graphs[0]
            if graph is None:
                return False, "Parsed graph is None"
            
            # Check if graph has actual content
            nodes = graph.get_nodes()
            edges = graph.get_edges()
            
            # A valid reasoning graph should have at least some nodes or edges
            if not nodes and not edges:
                return False, "Graph has no nodes or edges"
            
            return True, None
        except Exception as e:
            return False, f"Parse error: {str(e)}"



class ReasoningGraphGenerator:
    def __init__(
        self,
        models: List[str] = None,
        runs_per_model: int = 1,
        max_retries: int = 0,
        timeout: float = 600.0,
        max_connections_per_service: int = 256,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_concurrent_requests: int = 100
    ):
        """Initialize the generator."""
        if models is None:
            models = ["gpt-oss-120b-judge"]
        elif isinstance(models, str):
            models = [models]

        self.models = models
        self.runs_per_model = runs_per_model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_connections_per_service = max_connections_per_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.llm_clients = {}
        for model in self.models:
            self.llm_clients[model] = AsyncConsulOpenAIProxy(
                model=model,
                timeout=timeout,
                max_connections_per_service=max_connections_per_service
            )

        self.dot_validator = DOTValidator()

    async def close(self):
        """Close all LLM clients."""
        for client in self.llm_clients.values():
            await client.close()

    async def generate_reasoning_graph(self, thought: str, model: str) -> Optional[str]:
        """Generate Graphviz DOT code for a reasoning trace using LLM."""
        if not thought or not thought.strip():
            return None

        if model not in self.llm_clients:
            return None

        messages_for_llm = [
            {"role": "system", "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."},
            {"role": "user", "content": f"{GRAPH_GENERATION_PROMPT}\n\nReasoning trace:\n{thought}"}
        ]

        # Retry logic
        max_retries = self.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async with self.semaphore:
                    result = await self.llm_clients[model].chat_completions(
                        messages=messages_for_llm,
                        n=1,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                    if not result or 'choices' not in result or not result['choices']:
                        if attempt < max_retries:
                            print(f"    Retry attempt {attempt + 1}/{max_retries}: Empty response")
                            await asyncio.sleep(0.1 * (attempt + 1))  # Small delay
                        continue

                    content = result['choices'][0]['message']['content']

                    dot_match = re.search(r'```(?:dot\n)?(.*?)```', content, re.DOTALL)
                    if dot_match:
                        dot_code = dot_match.group(1).strip()
                    else:
                        dot_code = content.strip()

                    is_valid, error_msg = self.dot_validator.is_valid_dot_strict(dot_code)
                    if not is_valid:
                        if attempt < max_retries:
                            print(f"    Retry attempt {attempt + 1}/{max_retries}: Invalid DOT - {error_msg}")
                            last_error = error_msg
                            await asyncio.sleep(0.1 * (attempt + 1))  # Small delay
                        continue

                    # Success
                    if attempt > 0:
                        print(f"    Success after {attempt} retries")
                    return dot_code

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"    Retry attempt {attempt + 1}/{max_retries}: {e} (waiting {wait_time}s)")
                    await asyncio.sleep(wait_time)
                continue

        # All retries failed
        if max_retries > 0 and last_error:
            print(f"    Failed after {max_retries} retries: {last_error}")
        return None

def load_entries(input_path: str) -> List[Dict[str, Any]]:
    """Load all entries from input JSONL file."""
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_output_entries(output_path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Load entries from output file and extract completed task keys.

    Returns:
        Tuple of (entries, completed_task_keys)
        - entries: List of output entries
        - completed_task_keys: Set of completed task keys
          Format: "{thought_id}_{model_name}_{run_id}"
    """
    output_file = Path(output_path)
    if not output_file.exists():
        return [], set()

    entries = []
    completed_keys = set()

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                entries.append(entry)

                # Extract completed task key
                key_parts = [
                    str(entry.get('q_id', '')),
                    str(entry.get('t_id', '')),
                    entry.get('model', ''),
                    str(entry.get('run_id', ''))
                ]
                if all(part for part in key_parts):
                    key = "_".join(key_parts)
                    completed_keys.add(key)

    return entries, completed_keys


def generate_tasks(
    input_entries: List[Dict[str, Any]],
    models: List[str],
    runs_per_model: int,
    completed_keys: Set[str],
    start_idx: int,
    end_idx: int
) -> List[Task]:
    """
    Generate all annotation tasks.

    Args:
        input_entries: List of entries from input file
        models: List of model names
        runs_per_model: Number of runs per model
        completed_keys: Set of completed task keys
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)

    Returns:
        List of Task objects to process
    """
    tasks = []

    for q_id in range(start_idx, end_idx):
        if q_id >= len(input_entries):
            break

        entry = input_entries[q_id]
        question = entry.get('question', '')

        if 'reasoning' not in entry:
            continue

        for t_id, reasoning_item in enumerate(entry['reasoning']):
            thought = reasoning_item.get('thought', '')
            if not thought:
                continue

            thought_id = f"{q_id}_{t_id}"

            for model_name in models:
                for run_id in range(1, runs_per_model + 1):
                    # Check if already completed
                    task_key = f"{q_id}_{t_id}_{model_name}_{run_id}"
                    if task_key in completed_keys:
                        continue

                    task = Task(
                        thought_id=thought_id,
                        q_id=q_id,
                        t_id=t_id,
                        question=question,
                        thought=thought,
                        model_name=model_name,
                        run_id=run_id
                    )
                    tasks.append(task)

    return tasks


def save_result(output_path: str, result: Dict[str, Any]):
    """Append a single result to output JSONL file."""
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


async def process_task(
    task: Task,
    generator: ReasoningGraphGenerator,
    input_entries: List[Dict[str, Any]],
    output_path: str,
    pbar: tqdm
) -> bool:
    """
    Process a single task.

    Args:
        task: Task to process
        generator: ReasoningGraphGenerator instance
        input_entries: List of input entries
        output_path: Path to output file
        pbar: Progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate graph
        graph_dot = await generator.generate_reasoning_graph(task.thought, task.model_name)

        if graph_dot is None:
            pbar.update(1)
            return False

        # Create output record
        result = {
            'question': task.question,
            'thought': task.thought,
            'q_id': task.q_id,
            't_id': task.t_id,
            'reasoning_graph': graph_dot,
            'model': task.model_name,
            'run_id': task.run_id
        }

        # Save immediately
        save_result(output_path, result)

        pbar.update(1)
        return True

    except Exception as e:
        print(f"Error processing task {task.thought_id}: {e}")
        pbar.update(1)
        return False


async def process_dataset(
    input_path: str,
    output_path: str,
    generator: ReasoningGraphGenerator,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_concurrent_tasks: int = 100
):
    """
    Process dataset with thought-level concurrency and checkpointing.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        generator: ReasoningGraphGenerator instance
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
        max_concurrent_tasks: Maximum concurrent tasks
    """
    print(f"Loading entries from {input_path}...")
    input_entries = load_entries(input_path)
    total_input_entries = len(input_entries)

    if end_idx is None or end_idx > total_input_entries:
        end_idx = total_input_entries

    print(f"Total entries in input: {total_input_entries}")
    print(f"Processing entries {start_idx} to {end_idx}")

    # Load or initialize output file
    print(f"Loading checkpoint from {output_path}...")
    output_entries, completed_keys = load_output_entries(output_path)

    if not output_entries:
        print("No checkpoint found. Starting fresh...")
    else:
        print(f"Found {len(output_entries)} existing results")
        print(f"Found {len(completed_keys)} completed tasks")

    # Generate tasks
    print("Generating tasks...")
    tasks = generate_tasks(
        input_entries,
        generator.models,
        generator.runs_per_model,
        completed_keys,
        start_idx,
        end_idx
    )

    if not tasks:
        print("All tasks already completed!")
        return

    print(f"Total tasks to process: {len(tasks)}")

    # Process tasks with progress bar
    with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
        # Create semaphore for concurrent task limit
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def process_task_with_semaphore(task):
            async with semaphore:
                return await process_task(task, generator, input_entries, output_path, pbar)

        # Process all tasks
        await asyncio.gather(*[process_task_with_semaphore(task) for task in tasks])

    print(f"\nResults saved to {output_path}")
    print(f"Total tasks processed: {len(tasks)}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate reasoning graphs with thought-level concurrency')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--models', type=str, nargs='+', default=['gpt-oss-120b-judge'], help='Model names')
    parser.add_argument('--runs-per-model', type=int, default=1, help='Number of runs per model')
    parser.add_argument('--max-retries', type=int, default=0, help='Max retries per task on failure (default: 0)')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting index')
    parser.add_argument('--end-idx', type=int, default=None, help='Ending index')
    parser.add_argument('--max-concurrent-tasks', type=int, default=100, help='Max concurrent tasks')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens for generation')

    args = parser.parse_args()

    print("Using pydot for DOT parsing")
    print(f"Max concurrent tasks: {args.max_concurrent_tasks}")

    generator = ReasoningGraphGenerator(
        models=args.models,
        runs_per_model=args.runs_per_model,
        max_retries=args.max_retries,
        max_concurrent_requests=min(args.max_concurrent_tasks, 50),  # Limit per-model concurrency
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    try:
        await process_dataset(
            input_path=args.input,
            output_path=args.output,
            generator=generator,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_concurrent_tasks=args.max_concurrent_tasks
        )
    finally:
        await generator.close()


if __name__ == '__main__':
    asyncio.run(main())

"""
Example usage:
python3 reasoning_graph/build_reasoning_graph.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/rg_info.jsonl \
    --models gpt-oss-120b-judge claude-3.7-sonnet Qwen2-5-72B-Instruct \
    --runs-per-model 3 \
    --start-idx 0 \
    --end-idx 4 \
    --max-concurrent 256

"""
