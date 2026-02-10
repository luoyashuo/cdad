#!/usr/bin/env python3
"""
预处理模型：添加特殊 token 并保存
这样训练时 vLLM 加载的就是正确 vocab_size 的模型

关键：确保 tokenizer.vocab_size == model.config.vocab_size == embedding 层大小
这样 vLLM 加载时才能正确匹配
"""

import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出模型路径")
    parser.add_argument("--special_tokens", nargs="+", default=["<rewrite_reasoning>", "</rewrite_reasoning>"])
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Special tokens to add: {args.special_tokens}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu"  # 在 CPU 上操作，避免显存问题
    )

    original_vocab_size = tokenizer.vocab_size
    original_model_vocab_size = model.config.vocab_size
    print(f"Original tokenizer vocab_size: {original_vocab_size}")
    print(f"Original model vocab_size: {original_model_vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")

    # 关键：使用模型原始的 vocab_size 作为目标大小
    # 这样可以保持 pad_token_id (151643) 在有效范围内
    target_vocab_size = original_model_vocab_size  # 151936

    # 先添加特殊 token 到 tokenizer
    special_tokens_dict = {"additional_special_tokens": args.special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to tokenizer")

    # 添加额外的 token 来扩展 vocab_size 到 target_vocab_size
    num_to_add = target_vocab_size - len(tokenizer)
    print(f"Adding {num_to_add} dummy tokens to reach vocab_size {target_vocab_size}")
    dummy_tokens = [f"<dummy_{i}>" for i in range(num_to_add)]
    tokenizer.add_tokens(dummy_tokens)

    print(f"After adding tokens - len(tokenizer): {len(tokenizer)}, tokenizer.vocab_size: {tokenizer.vocab_size}")

    # Resize 模型的 embedding 层到 target_vocab_size
    model.resize_token_embeddings(target_vocab_size)
    print(f"Resized model embeddings to: {target_vocab_size}")

    # 确保 config 中的 vocab_size 正确
    model.config.vocab_size = target_vocab_size

    # 保存处理后的模型
    print(f"Saving model to: {args.output_path}")
    tokenizer.save_pretrained(args.output_path)
    model.save_pretrained(args.output_path)

    # 关键：直接修改 tokenizer_config.json 中的 vocab_size
    tokenizer_config_path = os.path.join(args.output_path, "tokenizer_config.json")
    with open(tokenizer_config_path, "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["vocab_size"] = target_vocab_size
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Updated vocab_size in tokenizer_config.json to {target_vocab_size}")

    # 验证保存的 tokenizer
    saved_tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
    print(f"After save - tokenizer.vocab_size: {saved_tokenizer.vocab_size}")
    print(f"After save - len(tokenizer): {len(saved_tokenizer)}")

    print("Done!")
    print(f"Final vocab_size: {target_vocab_size}")
    print(f"model.config.vocab_size: {model.config.vocab_size}")
    print(f"pad_token_id: {tokenizer.pad_token_id} (should be < {target_vocab_size})")


if __name__ == "__main__":
    main()
