#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_math_simple_multiprocess.py
Multi-process version with enhanced processing speed using ProcessPoolExecutor.
"""

import os
import json
import random
import argparse
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Event
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def is_problem_of_type(record, problem_type='math', min_reasoning_length=None, check_full_response=True):
    """
    Check if a record is a MATH or CODE problem

    Note: Actual OmniThought dataset doesn't have \boxed patterns in solution fields.
    It contains full_response with reasoning. This function adapts to the actual data format.

    Args:
        record: The data record to check
        problem_type: Type of problem to check for ('math' or 'code')
        min_reasoning_length: Minimum length of reasoning list (optional)
        check_full_response: Whether to check full_response for problem indicators (default: True)
    """
    try:
        if 'reasoning' not in record:
            return False
            
        reasoning = record['reasoning']
        
        # Handle different types of reasoning (list, ndarray, etc.)
        reasoning_length = 0
        if isinstance(reasoning, list):
            reasoning_length = len(reasoning)
        elif hasattr(reasoning, 'size') and hasattr(reasoning, '__getitem__'):
            reasoning_length = reasoning.size
        else:
            return False
        
        # Check minimum reasoning length if specified
        if min_reasoning_length is not None and reasoning_length < min_reasoning_length:
            return False
        
        if reasoning_length == 0:
            return False
        
        # Get first reasoning item
        reasoning_item = None
        if isinstance(reasoning, (list, tuple)):
            reasoning_item = reasoning[0]
        elif hasattr(reasoning, '__getitem__'):
            reasoning_item = reasoning[0]
        
        if not isinstance(reasoning_item, dict):
            return False
        
        if problem_type == 'math':
            # Check for solution field with \boxed (math problem indicator)
            if 'solution' in reasoning_item:
                solution = reasoning_item['solution']
                if solution and r'\boxed' in str(solution):
                    return True
            # If no explicit math indicators found, return False
            return False
        else:  # problem_type == 'code'
            # Check for code-related indicators in full_response or solution
            if 'full_response' in reasoning_item:
                full_response = reasoning_item['full_response']
                if full_response:
                    # Common code indicators
                    code_indicators = ['def ', 'class ', 'import ', '#include', 'public static',
                                     'function ', 'void ', 'int ', 'print(', 'printf(']
                    if any(indicator in str(full_response) for indicator in code_indicators):
                        return True

            if 'solution' in reasoning_item:
                solution = reasoning_item['solution']
                if solution:
                    code_indicators = ['def ', 'class ', 'import ', '#include', 'public static',
                                     'function ', 'void ', 'int ', 'print(', 'printf(']
                    if any(indicator in str(solution) for indicator in code_indicators):
                        return True

            # If no code indicators found, return False
            return False
            
    except Exception as e:
        # Uncomment for debugging:
        # print(f"Error checking record: {e}")
        return False


def convert_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def filter_reasoning_items_by_token_limit(record, min_reasoning_length, max_thought_tokens, min_thought_tokens, tokenizer):
    """
    Filter reasoning items by token length and check if remaining items meet min_reasoning_length.
    
    Args:
        record: The data record
        min_reasoning_length: Minimum number of reasoning items required
        max_thought_tokens: Maximum token length for each thought
        min_thought_tokens: Minimum token length for each thought
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (filtered_record, should_keep, filter_reason) where should_keep indicates if record should be kept
        and filter_reason is a string indicating why it was filtered (if applicable)
    """
    reasoning = record.get('reasoning', [])
    
    # Handle different types of reasoning (list, ndarray, etc.)
    # Use explicit length check instead of truthiness check for numpy arrays
    if reasoning is None:
        return record, False, "reasoning_is_none"
    
    # Check if reasoning is empty (handles both list and ndarray)
    reasoning_length = 0
    if isinstance(reasoning, list):
        reasoning_length = len(reasoning)
    elif hasattr(reasoning, 'size'):
        reasoning_length = reasoning.size
    else:
        return record, False, "invalid_reasoning_type"
    
    if reasoning_length == 0:
        return record, False, "reasoning_empty"
    
    # Convert to list if it's ndarray-like
    reasoning_list = []
    if isinstance(reasoning, list):
        reasoning_list = reasoning
    elif hasattr(reasoning, '__getitem__') and hasattr(reasoning, 'size'):
        reasoning_list = [reasoning[i] for i in range(reasoning.size)]
    else:
        return record, False, "invalid_reasoning_structure"
    
    # First check: if initial count is less than min_reasoning_length, filter out
    if min_reasoning_length is not None and len(reasoning_list) < min_reasoning_length:
        return record, False, "reason_num_insufficient"
    
    # Second check: filter reasoning items exceeding max_thought_tokens
    if max_thought_tokens is not None and tokenizer is not None:
        filtered_reasoning = []
        token_too_long_items = 0
        
        for reasoning_item in reasoning_list:
            if isinstance(reasoning_item, dict):
                thought = reasoning_item.get('thought', '')
                if thought is not None and thought != '':
                    try:
                        tokens = tokenizer.encode(str(thought), add_special_tokens=False)
                        token_len = len(tokens)
                        # Check if token length is within bounds
                        if (max_thought_tokens is None or token_len <= max_thought_tokens) and \
                           (min_thought_tokens is None or token_len >= min_thought_tokens):
                            # Keep this reasoning item
                            filtered_reasoning.append(reasoning_item)
                        else:
                            token_too_long_items += 1
                    except Exception:
                        # If tokenization fails, keep the item to be safe
                        filtered_reasoning.append(reasoning_item)
                else:
                    # Keep items without thought field or with empty thought
                    filtered_reasoning.append(reasoning_item)
            else:
                # Keep non-dict items as-is
                filtered_reasoning.append(reasoning_item)
        
        # Third check: after filtering, check if we still have enough items
        if min_reasoning_length is not None and len(filtered_reasoning) < min_reasoning_length:
            # Check if it was filtered due to token limit
            if token_too_long_items > 0:
                return record, False, "reason_num_insufficient_after_token_filter"
            else:
                return record, False, "reason_num_insufficient"
        
        # Update the record with filtered reasoning
        record['reasoning'] = filtered_reasoning
    
    return record, True, "passed"


def process_parquet_file_worker(args):
    """
    Process a single parquet file and return math/code problems.
    This function is run in a separate process.
    """
    parquet_file, temp_dir, stop_event, min_reasoning_length, max_thought_tokens, min_thought_tokens, tokenizer_path, problem_type = args
    
    import pyarrow.parquet as pq
    
    # Initialize tokenizer if needed
    tokenizer = None
    if max_thought_tokens is not None or min_thought_tokens is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
    
    math_problems = []
    
    # Statistics for debugging
    total_records = 0
    not_math_count = 0
    token_too_long_count = 0
    reason_num_insufficient_count = 0
    
    try:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        for _, row in tqdm(df.iterrows(), desc=f"Processing {parquet_file.name}", unit="row", total=len(df)):
            # Check if stop event is set
            if stop_event.is_set():
                break
                
            record = row.to_dict()
            total_records += 1
            
            # Check if it's the requested type of problem (without min_reasoning_length check here)
            if is_problem_of_type(record, problem_type=problem_type, min_reasoning_length=None, check_full_response=True):
                # Apply the new filtering logic
                filtered_record, should_keep, filter_reason = filter_reasoning_items_by_token_limit(
                    record, min_reasoning_length, max_thought_tokens, min_thought_tokens, tokenizer
                )
                
                if should_keep:
                    json_item = convert_numpy(filtered_record)
                    math_problems.append(json_item)
                else:
                    # Track filter reasons
                    if filter_reason == "token_too_long" or filter_reason == "reason_num_insufficient_after_token_filter":
                        token_too_long_count += 1
                    elif filter_reason == "reason_num_insufficient":
                        reason_num_insufficient_count += 1
            else:
                not_math_count += 1
        
        # Clean up
        del df
        del table
        
    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")
        return [], {"total_records": total_records, "not_math_count": not_math_count,
                    "token_too_long_count": token_too_long_count,
                    "reason_num_insufficient_count": reason_num_insufficient_count,
                    "error": str(e)}
    
    return math_problems, {"total_records": total_records, "not_math_count": not_math_count,
                           "token_too_long_count": token_too_long_count,
                           "reason_num_insufficient_count": reason_num_insufficient_count}


def collect_problems_parallel(parquet_files, sample_size, temp_dir, seed=42, max_workers=None, min_reasoning_length=None, max_thought_tokens=None, min_thought_tokens=None, tokenizer_path=None, problem_type='math'):
    """
    Process parquet files in parallel using multiple processes.
    Args:
        min_reasoning_length: Minimum length of reasoning list for filtering
        max_thought_tokens: Maximum token length for thought field
        tokenizer_path: Path to tokenizer
        problem_type: Type of problem to collect ('math' or 'code')
    """
    if max_workers is None:
        # Default to number of CPU cores
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
    
    random.seed(seed)
    shuffled_files = parquet_files.copy()
    random.shuffle(shuffled_files)
    
    collected = 0
    processed_files = 0
    temp_file = os.path.join(temp_dir, "math_problems.jsonl")
    
    # Initialize empty file
    open(temp_file, 'w').close()
    
    # Global statistics for debugging
    total_records_global = 0
    not_math_count_global = 0
    token_too_long_count_global = 0
    reason_num_insufficient_count_global = 0
    
    print("\n" + "="*60)
    print("COLLECTING MATH PROBLEMS (MULTI-PROCESS)")
    print("="*60)
    print(f"Target: {sample_size:,} MATH problems" if sample_size is not None else "Target: All available MATH problems (no limit)")
    print(f"Processing files in random order (seed: {seed})")
    print(f"Max workers: {max_workers}")
    if max_thought_tokens is not None:
        print(f"Max thought tokens: {max_thought_tokens}")
        print(f"Tokenizer: {tokenizer_path}")
    print("="*60)
    
    with Manager() as manager:
        stop_event = manager.Event()
        
        # Prepare arguments for worker processes
        worker_args = [(file, temp_dir, stop_event, min_reasoning_length, max_thought_tokens, min_thought_tokens, tokenizer_path, problem_type) for file in shuffled_files]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_parquet_file_worker, args): args[0] 
                      for args in worker_args}
            
            # Collect results as they complete
            for future in as_completed(futures):
                if stop_event.is_set():
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                
                parquet_file = futures[future]
                processed_files += 1
                
                try:
                    math_problems, stats = future.result()
                    
                    # Accumulate statistics
                    total_records_global += stats.get("total_records", 0)
                    not_math_count_global += stats.get("not_math_count", 0)
                    token_too_long_count_global += stats.get("token_too_long_count", 0)
                    reason_num_insufficient_count_global += stats.get("reason_num_insufficient_count", 0)
                    
                    if math_problems:
                        # Append to temp file
                        with open(temp_file, 'a', encoding='utf-8') as f:
                            for problem in math_problems:
                                if sample_size is not None and collected >= sample_size:
                                    stop_event.set()
                                    break
                                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
                                collected += 1
                        
                        problem_type_name = "MATH" if problem_type == 'math' else "CODE"
                        print(f"  [{processed_files}/{len(shuffled_files)}] {parquet_file.name}: "
                              f"+{len(math_problems)} {problem_type_name} problems, total: {collected:,}" + (f"/{sample_size:,}" if sample_size is not None else ""))
                        
                        if sample_size is not None and collected >= sample_size:
                            stop_event.set()
                            print("\n" + "="*60)
                            print("TARGET REACHED!")
                            print("="*60)
                            break
                    else:
                        # Print detailed statistics for each file
                        problem_type_name = "math " if problem_type == 'math' else "code "
                        print(f"  [{processed_files}/{len(shuffled_files)}] {parquet_file.name}: no {problem_type_name}problems found")

                        # Show detailed filtering statistics
                        total = stats.get("total_records", 0)
                        if total > 0:
                            not_target = stats.get("not_math_count", 0)
                            token_too_long = stats.get("token_too_long_count", 0)
                            reason_insufficient = stats.get("reason_num_insufficient_count", 0)
                            passed = total - not_target - token_too_long - reason_insufficient

                            print(f"      └─ Total records: {total:,}")
                            print(f"          ✗ Not {problem_type} problems: {not_target:,} ({not_target/total*100:.2f}%)")
                            print(f"          ✗ Token too long: {token_too_long:,} ({token_too_long/total*100:.2f}%)")
                            print(f"          ✗ Reason count insufficient: {reason_insufficient:,} ({reason_insufficient/total*100:.2f}%)")
                            print(f"          ✓ Passed filters: {passed:,} ({passed/total*100:.2f}%)")
                        else:
                            print(f"      └─ No records processed (possibly an error)")
                        if stats.get("error"):
                            print(f"      └─ Error: {stats['error']}")
                        
                except Exception as e:
                    print(f"  Error processing {parquet_file}: {e}")
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    problem_type_name = "MATH" if problem_type == 'math' else "CODE"
    print("="*60)
    print(f"Files processed: {processed_files}/{len(shuffled_files)}")
    print(f"{problem_type_name} problems collected: {collected:,}" + (f"/{sample_size:,}" if sample_size is not None else ""))

    # Store statistics in a global variable for later use
    collect_problems_parallel.stats = {
        "total_records": total_records_global,
        "not_math_count": not_math_count_global,
        "token_too_long_count": token_too_long_count_global,
        "reason_num_insufficient_count": reason_num_insufficient_count_global
    }
    
    return temp_file, collected


def analyze_data(problems, tokenizer=None):
    """Analyze problems and collect statistics for plotting"""
    print("\nAnalyzing data statistics...")
    
    # Collect lengths of reasoning
    reasoning_lengths = []
    cognitive_difficulty_levels = []
    reasoning_verbosity_levels = []
    thought_token_lengths = []
    full_response_token_lengths = []
    
    # Use tqdm to show progress for tokenization
    if tokenizer is not None:
        print("Tokenizing thoughts and full_responses (this may take a while)...")
        progress_bar = tqdm(total=len(problems), desc="Processing problems", unit="problem")
    else:
        progress_bar = None
    
    for problem in problems:
        reasoning_val = problem.get('reasoning')
        if reasoning_val and isinstance(reasoning_val, list):
            reasoning_lengths.append(len(reasoning_val))
            
            # Collect levels from each reasoning item
            for reasoning_item in reasoning_val:
                if isinstance(reasoning_item, dict):
                    # Get Cognitive Difficulty level
                    cog_diff = reasoning_item.get('Cognitive_Difficulty', {})
                    if isinstance(cog_diff, dict) and 'level' in cog_diff:
                        cognitive_difficulty_levels.append(cog_diff['level'])
                    
                    # Get Reasoning Verbosity level
                    reason_verb = reasoning_item.get('Reasoning_Verbosity', {})
                    if isinstance(reason_verb, dict) and 'level' in reason_verb:
                        reasoning_verbosity_levels.append(reason_verb['level'])
                    
                    # Calculate token lengths if tokenizer is provided
                    if tokenizer is not None:
                        # Calculate thought token length
                        thought = reasoning_item.get('thought', '')
                        if thought:
                            try:
                                tokens = tokenizer.encode(str(thought), add_special_tokens=False)
                                thought_token_lengths.append(len(tokens))
                            except Exception:
                                pass
                        
                        # Calculate full_response token length
                        full_response = reasoning_item.get('full_response', '')
                        if full_response:
                            try:
                                tokens = tokenizer.encode(str(full_response), add_special_tokens=False)
                                full_response_token_lengths.append(len(tokens))
                            except Exception:
                                pass
        
        # Update progress bar
        if progress_bar is not None:
            progress_bar.update(1)
    
    # Close progress bar
    if progress_bar is not None:
        progress_bar.close()
    
    return {
        'reasoning_lengths': reasoning_lengths,
        'cognitive_difficulty_levels': cognitive_difficulty_levels,
        'reasoning_verbosity_levels': reasoning_verbosity_levels,
        'thought_token_lengths': thought_token_lengths,
        'full_response_token_lengths': full_response_token_lengths
    }


def plot_distributions(stats, output_dir):
    """Create and save distribution plots"""
    print("\nCreating distribution plots...")
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Reasoning Length Distribution
    fig, ax = plt.subplots(figsize=(12, 4))
    if stats['reasoning_lengths']:
        lengths = stats['reasoning_lengths']
        ax.hist(lengths, bins=min(50, max(lengths)), alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Length of reasoning (number of items)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Reasoning Lengths')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        ax.text(0.95, 0.95, f'Mean: {mean_len:.2f}\nMedian: {median_len:.2f}\nMax: {max(lengths)}\nMin: {min(lengths)}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No reasoning length data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reasoning_length_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Cognitive Difficulty Level Distribution
    fig, ax = plt.subplots(figsize=(12, 4))
    if stats['cognitive_difficulty_levels']:
        levels = stats['cognitive_difficulty_levels']
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Sort by level
        sorted_levels = sorted(level_counts.items())
        x_labels = [str(item[0]) for item in sorted_levels]
        y_values = [item[1] for item in sorted_levels]
        
        bars = ax.bar(x_labels, y_values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Cognitive Difficulty Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Cognitive Difficulty Levels')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No cognitive difficulty data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cognitive_difficulty_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Reasoning Verbosity Level Distribution
    fig, ax = plt.subplots(figsize=(12, 4))
    if stats['reasoning_verbosity_levels']:
        levels = stats['reasoning_verbosity_levels']
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Sort by level
        sorted_levels = sorted(level_counts.items())
        x_labels = [str(item[0]) for item in sorted_levels]
        y_values = [item[1] for item in sorted_levels]
        
        bars = ax.bar(x_labels, y_values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Reasoning Verbosity Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Reasoning Verbosity Levels')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No reasoning verbosity data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reasoning_verbosity_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Thought Token Length Distribution
    if stats['thought_token_lengths']:
        fig, ax = plt.subplots(figsize=(12, 4))
        lengths = stats['thought_token_lengths']
        ax.hist(lengths, bins=min(50, max(10, max(lengths)//10)), alpha=0.7, color='gold', edgecolor='black')
        ax.set_xlabel('Token Length of Thought')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Thought Token Lengths')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        p95 = np.percentile(lengths, 95)
        p99 = np.percentile(lengths, 99)
        ax.text(0.95, 0.95, f'Mean: {mean_len:.2f}\nMedian: {median_len:.2f}\n95th: {p95:.0f}\n99th: {p99:.0f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'thought_token_length_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Full Response Token Length Distribution
    if stats['full_response_token_lengths']:
        fig, ax = plt.subplots(figsize=(12, 4))
        lengths = stats['full_response_token_lengths']
        ax.hist(lengths, bins=min(50, max(10, max(lengths)//10)), alpha=0.7, color='plum', edgecolor='black')
        ax.set_xlabel('Token Length of Full Response')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Full Response Token Lengths')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        p95 = np.percentile(lengths, 95)
        p99 = np.percentile(lengths, 99)
        ax.text(0.95, 0.95, f'Mean: {mean_len:.2f}\nMedian: {median_len:.2f}\n95th: {p95:.0f}\n99th: {p99:.0f}',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'full_response_token_length_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to: {plots_dir}")
    print("  - reasoning_length_distribution.png")
    print("  - cognitive_difficulty_distribution.png")
    print("  - reasoning_verbosity_distribution.png")
    if stats['thought_token_lengths']:
        print("  - thought_token_length_distribution.png")
    if stats['full_response_token_lengths']:
        print("  - full_response_token_length_distribution.png")


def sample_and_split(temp_file, actual_count, test_ratio, output_dir, max_thought_tokens=None, min_thought_tokens=None, tokenizer_path=None):
    """Load collected problems, split into train/test, and generate plots"""
    print("\nLoading and splitting data...")
    
    # Load all collected problems
    problems = []
    with open(temp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    
    # Verify count
    if len(problems) != actual_count:
        print(f"Warning: Expected {actual_count} problems, got {len(problems)}")
    
    # Initialize tokenizer if needed for token length analysis
    tokenizer = None
    if tokenizer_path:
        try:
            print(f"\nLoading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
    
    # Analyze data (plots disabled for clean output)
    stats = analyze_data(problems, tokenizer)
    # plot_distributions(stats, output_dir)  # Disabled: keep output clean
    
    # Split
    random.seed(42)
    random.shuffle(problems)
    
    split_idx = int(len(problems) * (1 - test_ratio))
    train = problems[:split_idx]
    test = problems[split_idx:]
    
    print(f"Split: {len(train)} train, {len(test)} test")
    return train, test


def save_results(train, test, output_dir, args=None):
    """
    Save train/test splits
    Args:
        args: Command line arguments to determine output format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save train
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Save val (renamed from test)
    val_path = os.path.join(output_dir, "val.jsonl")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved to {output_dir}:")
    print(f"  - train.jsonl ({len(train):,} samples)")
    print(f"  - val.jsonl ({len(test):,} samples)")


def show_sample(train, problem_type='math'):
    """Show a sample problem"""
    if not train:
        return

    print("\n" + "="*60)
    problem_type_name = "MATH" if problem_type == 'math' else "CODE"
    print(f"SAMPLE {problem_type_name} PROBLEM")
    print("="*60)
    
    sample = train[0]
    print(f"\nQuestion (truncated):\n{str(sample.get('question', ''))[:500]}...")
    
    reasoning_val = sample.get('reasoning')
    if reasoning_val and isinstance(reasoning_val, list) and len(reasoning_val) > 0:
        reasoning = reasoning_val[0]
        if isinstance(reasoning, dict) and 'solution' in reasoning:
            solution = reasoning['solution']
            print(f"\nSolution (truncated):\n{str(solution)[:500]}...")
            boxed_str = '\\boxed'
            print(f"\nContains \boxed: {boxed_str in str(solution)}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Filter MATH or CODE data from OmniThought with multi-process parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  - pyarrow: pip install pyarrow
  - transformers: pip install transformers

Example:
  # Filter math problems with sample size limit
  python filter_math_simple_multiprocess.py \
      --input_path /path/to/omnithought/data \
      --output_dir /path/to/output \
      --problem_type math \
      --sample_size 20000 \
      --test_ratio 0.01 \
      --seed 42 \
      --max_workers 8 \
      --max_thought_tokens 1024 \
      --min_thought_tokens 64

  # Filter code problems without sample size limit (process all data)
  python filter_math_simple_multiprocess.py \
      --input_path /path/to/omnithought/data \
      --output_dir /path/to/output \
      --problem_type code \
      --test_ratio 0.01 \
      --seed 42 \
      --max_workers 8
        """
    )
    
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to OmniThought dataset directory containing parquet files")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                       help="Output directory for filtered dataset (will only contain train.jsonl and val.jsonl)")
    parser.add_argument("--problem_type", type=str, default="math", choices=["math", "code"],
                       help="Type of problem to filter (math or code, default: math)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of problems to collect (default: None, no limit)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Ratio of test split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--reason_sample_num", type=int, default=None,
                       help="Minimum number of reasoning samples required (default: None, no filter)")
    parser.add_argument("--max_thought_tokens", type=int, default=None,
                       help="Maximum token length for thought field (default: None, no limit)")
    parser.add_argument("--min_thought_tokens", type=int, default=None,
                       help="Minimum token length for thought field (default: None, no limit)")
    parser.add_argument("--tokenizer_path", type=str, default="/mnt/moonfs/public-models-m2/Qwen/Qwen3-8B",
                       help="Path to tokenizer (default: /mnt/moonfs/public-models-m2/Qwen/Qwen3-8B)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate input directory
    if not os.path.isdir(args.input_path):
        print(f"Error: Input path '{args.input_path}' is not a directory")
        return
    
    # Find all parquet files
    parquet_files = sorted(Path(args.input_path).glob("*.parquet"))
    if not parquet_files:
        print(f"Error: No parquet files found in '{args.input_path}'")
        return

    print("="*60)
    problem_type_name = "MATH" if args.problem_type == 'math' else "CODE"
    print(f"OmniThought {problem_type_name} Dataset Filter (Multi-Process)")
    print("="*60)
    print(f"Input directory: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sample size: {args.sample_size:,}" if args.sample_size is not None else "Target sample size: No limit (process all)")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Max workers: {args.max_workers or 'CPU count'}")
    print(f"Total parquet files available: {len(parquet_files)}")
    if args.max_thought_tokens is not None:
        if args.max_thought_tokens is not None:
            print(f"Max thought tokens: {args.max_thought_tokens}")
        if args.min_thought_tokens is not None:
            print(f"Min thought tokens: {args.min_thought_tokens}")
        print(f"Tokenizer path: {args.tokenizer_path}")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="math_filter_mp_")
    print(f"Temporary directory: {temp_dir}")
    print("="*60)

    try:
        # Collect problems with multi-process processing
        temp_file, actual_count = collect_problems_parallel(
            parquet_files, args.sample_size, temp_dir, args.seed, args.max_workers,
            args.reason_sample_num, args.max_thought_tokens, args.min_thought_tokens, args.tokenizer_path, args.problem_type
        )

        if actual_count == 0:
            problem_type_name = "MATH" if args.problem_type == 'math' else "CODE"
            print(f"\nNo {problem_type_name} problems found!")
            print("\n" + "="*60)
            print("FILTERING STATISTICS (DEBUG INFO)")
            print("="*60)
            stats = getattr(collect_problems_parallel, 'stats', {})
            if stats:
                total = stats.get("total_records", 0)
                not_target = stats.get("not_math_count", 0)
                token_too_long = stats.get("token_too_long_count", 0)
                reason_insufficient = stats.get("reason_num_insufficient_count", 0)

                print(f"Total records processed: {total:,}")
                print(f"  - Not {args.problem_type} problems: {not_target:,} ({not_target/total*100:.2f}%)")
                print(f"  - Token too long: {token_too_long:,} ({token_too_long/total*100:.2f}%)")
                print(f"  - Reason number insufficient: {reason_insufficient:,} ({reason_insufficient/total*100:.2f}%)")
                passed = total - not_target - token_too_long - reason_insufficient
                print(f"  - Passed all filters: {passed:,} ({passed/total*100:.2f}%)")
            else:
                print("No statistics available (this should not happen)")
            print("="*60 + "\n")
            return
        
        # Split into train/test
        train, test = sample_and_split(temp_file, actual_count, args.test_ratio, args.output_dir, args.max_thought_tokens, args.min_thought_tokens, args.tokenizer_path)
        
        # Save results
        save_results(train, test, args.output_dir, args)
        
        # Show statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        total = len(train) + len(test)
        print(f"Train samples: {len(train):,} ({len(train)/total:.1%})")
        print(f"Test samples: {len(test):,} ({len(test)/total:.1%})")
        print(f"Total samples: {total:,}")
        print("="*60)
        
        # Show sample
        show_sample(train, args.problem_type)
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

"""
# Filter math problems with sample size limit
python /home/luoyashuo/research/graph_kd/cdad/data_preprocess/filter_math_simple_multiprocess.py \
    --input_path /mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_math_20k_2 \
    --problem_type math \
    --sample_size 50000 \
    --test_ratio 0.01 \
    --seed 307 \
    --max_workers 32 \
    --reason_sample_num 3 \
    --max_thought_tokens 4096 \
    --min_thought_tokens 1024 \
    --tokenizer_path /mnt/moonfs/public-models-m2/Qwen/Qwen3-8B

# Filter code problems without sample size limit (process all data)
python /home/luoyashuo/research/graph_kd/cdad/data_preprocess/filter_math_simple_multiprocess.py \
    --input_path /mnt/moonfs/luoyashuo-m2/hf_home/dataset/alibaba-pai/OmniThought/data \
    -o /mnt/moonfs/luoyashuo-m2/research/cdad/data/omnithought_code_all \
    --problem_type code \
    --test_ratio 0.01 \
    --seed 307 \
    --max_workers 64 \
    --reason_sample_num 3 \
    --max_thought_tokens 4096 \
    --min_thought_tokens 1024 \
    --tokenizer_path /mnt/moonfs/public-models-m2/Qwen/Qwen3-8B
"""