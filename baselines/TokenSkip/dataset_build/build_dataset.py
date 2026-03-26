#!/usr/bin/env python3
"""Build TokenSkip dataset from deepmath_plain_30k training data.

For each sample, based on `skip_ratio`:
  - (1 - skip_ratio) fraction: keep original CoT
  - skip_ratio fraction: compress thought with LLMLingua-2

Output is ShareGPT-format conversations compatible with LlamaFactory.

Supports data-parallel sharding via --rank / --world-size for multi-GPU speedup.
Each rank saves a partial file; run --merge-only afterwards to combine.
"""

import os
import json
import random
import argparse

import numpy as np
import yaml
from tqdm import tqdm
from llmlingua import PromptCompressor


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def extract_thought_and_solution(content):
    thought = content.split('<think>\n')[-1].split('\n</think>')[0].strip()
    solution = content.split('<think>\n')[-1].split('\n</think>')[1].strip()
    return thought, solution


def part_path(output_path, rank):
    return f"{output_path}.rank{rank}"


def build_dataset(config, rank=0, world_size=1):
    seed_everything(config.get('seed', 42))

    input_file = config['input_file']
    output_dir = config['output_dir']
    output_filename = config['output_filename']
    output_path = os.path.join(output_dir, output_filename)

    llmlingua_model = config['llmlingua_model']
    compression_ratio = config['compression_ratio']
    skip_ratio = config['skip_ratio']

    data = load_jsonl(input_file)
    print(f"[rank {rank}] Loaded {len(data)} samples from {input_file}")

    # Compute global skip_indices deterministically (same seed on every rank).
    n_skip = int(len(data) * skip_ratio)
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    skip_indices = set(all_indices[:n_skip])
    print(f"[rank {rank}] Will compress {n_skip}/{len(data)} samples "
          f"(skip_ratio={skip_ratio}, compression_ratio={compression_ratio})")

    # Each rank handles every world_size-th sample starting at rank.
    local_indices = list(range(rank, len(data), world_size))
    local_skip_count = sum(1 for i in local_indices if i in skip_indices)
    print(f"[rank {rank}] Processing {len(local_indices)} samples "
          f"({local_skip_count} to compress)")

    llm_lingua = None
    if local_skip_count > 0:
        print(f"[rank {rank}] Loading LLMLingua-2 model from {llmlingua_model}")
        llm_lingua = PromptCompressor(
            model_name=llmlingua_model,
            use_llmlingua2=True,
        )

    output_data = []
    n_compressed = 0
    for i in tqdm(local_indices, desc=f"rank {rank}"):
        item = data[i]
        conversations = item['conversations']
        user_content = conversations[0]['content']
        assistant_content = conversations[-1]['content']

        if i in skip_indices and llm_lingua is not None:
            thought, solution = extract_thought_and_solution(assistant_content)
            compressed = llm_lingua.compress_prompt(
                thought,
                rate=compression_ratio,
                force_reserve_digit=True,
                drop_consecutive=True,
            )
            compressed_thought = compressed['compressed_prompt']
            new_assistant_content = f"<think>\n{compressed_thought}\n</think>\n{solution}"
            n_compressed += 1
        else:
            new_assistant_content = assistant_content

        output_data.append({
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": new_assistant_content},
            ]
        })

    if world_size == 1:
        # Single-process path: shuffle and save final output directly.
        random.shuffle(output_data)
        save_jsonl(output_data, output_path)
        print(f"[rank {rank}] Saved {len(output_data)} samples "
              f"({n_compressed} compressed) to {output_path}")
        _maybe_register_llamafactory(config, output_path)
    else:
        # Multi-process path: save partial shard, merge happens separately.
        shard_path = part_path(output_path, rank)
        save_jsonl(output_data, shard_path)
        print(f"[rank {rank}] Saved {len(output_data)} samples "
              f"({n_compressed} compressed) to {shard_path}")


def merge_shards(config, world_size):
    seed = config.get('seed', 42)
    output_path = os.path.join(config['output_dir'], config['output_filename'])

    all_data = []
    for rank in range(world_size):
        shard_path = part_path(output_path, rank)
        shard = load_jsonl(shard_path)
        all_data.extend(shard)
        print(f"Loaded {len(shard)} samples from {shard_path}")

    random.seed(seed)
    random.shuffle(all_data)
    save_jsonl(all_data, output_path)
    print(f"Merged {len(all_data)} samples to {output_path}")

    _maybe_register_llamafactory(config, output_path)

    # Clean up shard files.
    for rank in range(world_size):
        shard_path = part_path(output_path, rank)
        os.remove(shard_path)
        print(f"Removed shard {shard_path}")


def _maybe_register_llamafactory(config, output_path):
    llamafactory_cfg = config.get('llamafactory', {})
    if llamafactory_cfg.get('enabled', False):
        register_to_llamafactory(
            output_path=output_path,
            dataset_info_path=llamafactory_cfg['dataset_info_path'],
            dataset_name=llamafactory_cfg['dataset_name'],
            use_absolute_path=llamafactory_cfg.get('use_absolute_path', True),
        )


def register_to_llamafactory(output_path, dataset_info_path, dataset_name, use_absolute_path=True):
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    file_name = os.path.abspath(output_path) if use_absolute_path else output_path
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"Registered dataset '{dataset_name}' to {dataset_info_path}")


def main():
    parser = argparse.ArgumentParser(description="Build TokenSkip dataset")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of this process (0-indexed)')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total number of parallel processes')
    parser.add_argument('--merge-only', action='store_true',
                        help='Skip processing; only merge existing rank shards')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.merge_only:
        merge_shards(config, args.world_size)
    else:
        build_dataset(config, rank=args.rank, world_size=args.world_size)


if __name__ == '__main__':
    main()
