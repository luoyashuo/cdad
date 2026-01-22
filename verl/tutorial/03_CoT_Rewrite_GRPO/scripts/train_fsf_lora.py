"""
FSF LoRA 训练脚本

训练一个专门用于 FSF (Failed Step Fraction) 分析的 LoRA adapter
基于 Student Model 基座进行微调

训练数据格式 (JSONL):
{
    "reasoning": "推理过程文本",
    "fsf_label": {
        "total_steps": 10,
        "failed_steps": 3,
        "fsf": 0.3
    }
}

使用方法:
python scripts/train_fsf_lora.py \
    --model_path ./models/Qwen2.5-3B-Instruct \
    --train_data ./data/fsf_train.jsonl \
    --output_dir ./outputs/fsf_lora \
    --epochs 3 \
    --batch_size 4 \
    --lora_rank 64
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator


# FSF Prompt 模板
FSF_PROMPT_TEMPLATE = """<|im_start|>system
你是一个推理过程分析专家。请分析推理过程中的"失败分支"占比。
失败分支定义：尝试了某种方法但最终放弃的分支。
请输出 JSON 格式：{{"total_steps": N, "failed_steps": M, "fsf": 0.0到1.0的数值}}<|im_end|>
<|im_start|>user
分析以下推理过程的失败分支占比：

{reasoning}

只输出 JSON，不要其他文字。<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""


class FSFDataset(Dataset):
    """FSF 训练数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)

        print(f"[FSFDataset] 加载 {len(self.data)} 条训练数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        reasoning = item["reasoning"][:2000]  # 截断过长的推理
        fsf_label = item["fsf_label"]

        # 构造响应
        response = json.dumps(fsf_label, ensure_ascii=False)

        # 构造完整文本
        full_text = FSF_PROMPT_TEMPLATE.format(
            reasoning=reasoning,
            response=response
        )

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # 计算 label（只在 response 部分计算 loss）
        # 找到 assistant 开始的位置
        prompt_text = FSF_PROMPT_TEMPLATE.format(reasoning=reasoning, response="")
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        labels = encodings["input_ids"].copy()
        # prompt 部分的 label 设为 -100
        for i in range(len(prompt_ids) - 1):
            labels[i] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }


def create_fsf_training_data(
    reasoning_data_path: str,
    output_path: str,
    fsf_api_url: str = None,
    use_local_judge: bool = False,
):
    """
    使用现有 FSF Judge 生成训练数据

    Args:
        reasoning_data_path: 原始推理数据路径（每行一个推理过程）
        output_path: 输出路径
        fsf_api_url: FSF API 地址（如果使用 API Judge）
        use_local_judge: 是否使用本地 Judge 模型
    """
    import httpx
    import asyncio

    async def analyze_fsf(reasoning: str) -> dict:
        """调用 FSF API 分析"""
        if not fsf_api_url:
            # 返回默认值
            return {"total_steps": 5, "failed_steps": 2, "fsf": 0.4}

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    fsf_api_url,
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"""分析以下推理过程的失败分支占比。
输出 JSON: {{"total_steps": N, "failed_steps": M, "fsf": 0.0-1.0}}

推理过程:
{reasoning[:2000]}"""
                            }
                        ],
                        "temperature": 0.0,
                    }
                )
                result = response.json()
                text = result["choices"][0]["message"]["content"]

                # 解析 JSON
                import re
                match = re.search(r'\{[^{}]*\}', text)
                if match:
                    return json.loads(match.group())
            except Exception as e:
                print(f"API 调用失败: {e}")

        return {"total_steps": 5, "failed_steps": 2, "fsf": 0.4}

    # 读取原始数据
    reasonings = []
    with open(reasoning_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if "reasoning" in item:
                        reasonings.append(item["reasoning"])
                    elif "cot" in item:
                        reasonings.append(item["cot"])
                except:
                    reasonings.append(line.strip())

    print(f"读取 {len(reasonings)} 条推理数据")

    # 批量分析
    async def process_all():
        tasks = [analyze_fsf(r) for r in reasonings]
        return await asyncio.gather(*tasks)

    fsf_labels = asyncio.run(process_all())

    # 保存训练数据
    with open(output_path, "w", encoding="utf-8") as f:
        for reasoning, label in zip(reasonings, fsf_labels):
            item = {
                "reasoning": reasoning,
                "fsf_label": label,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"保存 {len(reasonings)} 条训练数据到 {output_path}")


def train_fsf_lora(args):
    """训练 FSF LoRA"""

    # 初始化 Accelerator
    accelerator = Accelerator()

    print(f"[FSF LoRA 训练] 模型: {args.model_path}")
    print(f"[FSF LoRA 训练] 训练数据: {args.train_data}")
    print(f"[FSF LoRA 训练] 输出目录: {args.output_dir}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据集
    train_dataset = FSFDataset(
        args.train_data,
        tokenizer,
        max_length=args.max_length,
    )

    # 数据 collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("[FSF LoRA 训练] 开始训练...")
    trainer.train()

    # 保存模型
    print(f"[FSF LoRA 训练] 保存 LoRA adapter 到 {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[FSF LoRA 训练] 完成!")


def main():
    parser = argparse.ArgumentParser(description="FSF LoRA 训练")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="Student Model 路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径 (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")

    # LoRA 参数
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # 数据生成参数
    parser.add_argument("--generate_data", action="store_true", help="是否生成训练数据")
    parser.add_argument("--reasoning_data", type=str, help="原始推理数据路径（用于生成训练数据）")
    parser.add_argument("--fsf_api_url", type=str, help="FSF API 地址（用于生成训练数据）")

    args = parser.parse_args()

    # 如果需要生成训练数据
    if args.generate_data:
        if not args.reasoning_data:
            print("错误: 需要指定 --reasoning_data")
            return
        create_fsf_training_data(
            args.reasoning_data,
            args.train_data,
            args.fsf_api_url,
        )

    # 训练 LoRA
    train_fsf_lora(args)


if __name__ == "__main__":
    main()
