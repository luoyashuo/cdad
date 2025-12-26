#!/usr/bin/env python3
"""
Build reasoning graphs from COT data and compute FSF (Failed-Step Fraction).

This script reads a JSONL dataset, uses an LLM to convert each reasoning trace
into a Graphviz DOT representation, computes the FSF metric, and saves the results.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        model: str = "gpt-oss-120b-judge",
        timeout: float = 600.0,
        max_connections_per_service: int = 256,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_concurrent_requests: int = 10
    ):
        self.model = model
        self.timeout = timeout
        self.max_connections_per_service = max_connections_per_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Initialize LLM client
        self.llm_client = AsyncConsulOpenAIProxy(
            model=model,
            timeout=timeout,
            max_connections_per_service=max_connections_per_service
        )
    
    async def close(self):
        """Close the LLM client."""
        await self.llm_client.close()
    
    async def generate_reasoning_graph(self, thought: str) -> Optional[str]:
        """
        Generate Graphviz DOT code for a reasoning trace using LLM.
        
        Args:
            thought: The reasoning trace (COT) to convert to graph
            
        Returns:
            Graphviz DOT code as string, or None if generation fails
        """
        if not thought or not thought.strip():
            return None
        
        messages_for_llm = [
            {"role": "system", "content": "You are a helpful assistant that converts reasoning traces into Graphviz diagrams."},
            {"role": "user", "content": f"{GRAPH_GENERATION_PROMPT}\n\nReasoning trace:\n{thought}"}
        ]
        
        async with self.semaphore:
            try:
                result = await self.llm_client.chat_completions(
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
                print(f"Error generating graph: {e}")
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
            failed_nodes = len(re.findall(r'fillcolor\s*=\s*["\"]?lightpink["\"]?', dot_code, re.IGNORECASE))
            
            if total_nodes == 0:
                return None
            
            return failed_nodes / total_nodes
            
        except Exception as e:
            print(f"Error computing FSF: {e}")
            return None


async def process_single_entry(entry: Dict[str, Any], builder: ReasoningGraphBuilder, entry_idx: int) -> Dict[str, Any]:
    """
    Process a single dataset entry by generating reasoning graphs and computing FSF for all reasoning traces.
    
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
        
        # Generate reasoning graph
        graph_dot = await builder.generate_reasoning_graph(thought)
        
        if graph_dot:
            # Compute FSF
            fsf = builder.compute_fsf_from_dot(graph_dot)
            
            # Add results to the reasoning item
            reasoning_item['reasoning_graph'] = {
                'reasoning_graph': graph_dot,
                'fsf': fsf
            }
            print(f"  Generated graph with FSF={fsf}")
        else:
            print(f"  Failed to generate graph")
            reasoning_item['reasoning_graph'] = {
                'reasoning_graph': None,
                'fsf': None
            }
    
    return entry


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
    parser.add_argument('--model', type=str, default='gpt-oss-120b-judge',
                       help='Model name for LLM API')
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
    
    args = parser.parse_args()
    
    # Create builder
    builder = ReasoningGraphBuilder(
        model=args.model,
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
    finally:
        await builder.close()


if __name__ == '__main__':
    asyncio.run(main())
"""
python3 reasoning_graph/build_reasoning_graph.py \
    --input /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train.jsonl \
    --output /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k/train_rg.jsonl \
    --model gpt-oss-120b-judge \
    --max-concurrent 10 \
    --temperature 0.3 \
    --max-tokens 8192 \
    --start-idx 0 \
    --end-idx 10
"""
