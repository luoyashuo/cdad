#!/usr/bin/env python3
"""
使用 Gemini API 处理 CoT 重写任务

功能：
1. 读取输入 JSONL，使用 Gemini 模型基于 system_prompt 和 user_prompt 生成回复
2. 输出 sharegpt 格式数据
3. 支持断点续传和增量更新
4. 支持多线程并发（协程控制）
5. 自动注册到 LlamaFactory

输入格式：
{
    "question": "...",
    "original_thought": "...",
    "solution": "...",
    "system_prompt": "...",
    "user_prompt": "...",
    "fsf": 0.125,
    ...
}

输出格式 (sharegpt):
{
    "conversations": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": model_response}
    ],
    "_fsf": fsf
}
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import aiofiles
from aiofiles import os as aio_os
from loguru import logger
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm

# 配置 OpenAI 客户端用于访问 Gemini API
import openai

client = openai.AsyncOpenAI(
    api_key="sk-rrNoK6KmoZ2YIqsq7b744dB382Ad428cA22f090507DaA45e",
    base_url="https://api.gpt.ge/v1/",
    default_headers={"x-foo": "true"},
)

MODEL = "gemini-3.1-flash-lite-preview"
INPUT_PRICE_PER_TOKEN = 0.00025 / 1000
OUTPUT_PRICE_PER_TOKEN = 0.0015 / 1000

# 配置
CONCURRENT_REQUESTS = 10  # 并发数
BATCH_SIZE = 100  # 每处理多少条保存一次
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒）


@dataclass
class ProcessingState:
    """处理状态"""
    processed_ids: set  # 已处理的样本 ID
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float


async def call_gemini_api(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = MAX_RETRIES
) -> tuple[Optional[str], int, int]:
    """
    调用 Gemini API
    
    Returns:
        (model_response, input_tokens, output_tokens)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            return content, usage.prompt_tokens, usage.completion_tokens
            
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # 指数退避
            else:
                logger.error(f"API 调用最终失败: {e}")
                return None, 0, 0
    
    return None, 0, 0


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """计算 API 调用成本"""
    return (input_tokens * INPUT_PRICE_PER_TOKEN + 
            output_tokens * OUTPUT_PRICE_PER_TOKEN)


def build_sharegpt_item(
    system_prompt: str,
    user_prompt: str,
    model_response: str,
    fsf: float
) -> dict:
    """构建 sharegpt 格式数据项"""
    return {
        "conversations": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": model_response
            }
        ],
        "_fsf": fsf  # 临时保存 FSF 用于后续排序
    }


async def load_state(state_file: Path) -> ProcessingState:
    """加载处理状态"""
    if not state_file.exists():
        return ProcessingState(
            processed_ids=set(),
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0
        )
    
    try:
        async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
            return ProcessingState(
                processed_ids=set(data.get("processed_ids", [])),
                total_input_tokens=data.get("total_input_tokens", 0),
                total_output_tokens=data.get("total_output_tokens", 0),
                total_cost=data.get("total_cost", 0.0)
            )
    except Exception as e:
        logger.warning(f"加载状态文件失败: {e}，重新开始")
        return ProcessingState(
            processed_ids=set(),
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0
        )


async def save_state(state_file: Path, state: ProcessingState):
    """保存处理状态"""
    data = {
        "processed_ids": list(state.processed_ids),
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "total_cost": state.total_cost,
        "timestamp": datetime.now().isoformat()
    }
    
    # 使用临时文件然后重命名，保证原子性
    temp_file = state_file.with_suffix('.tmp')
    async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))
    
    await aio_os.rename(temp_file, state_file)


async def process_single_item(
    item: dict,
    item_id: str,
    state: ProcessingState,
    output_file: Path,
    lock: asyncio.Lock
) -> bool:
    """
    处理单个数据项
    
    Returns:
        是否成功处理
    """
    # 检查是否已处理
    if item_id in state.processed_ids:
        return True
    
    system_prompt = item.get("system_prompt", "")
    user_prompt = item.get("user_prompt", "")
    fsf = item.get("fsf", 0.0)
    
    # 调用 API
    model_response, input_tokens, output_tokens = await call_gemini_api(
        system_prompt, user_prompt
    )
    
    if model_response is None:
        logger.error(f"处理失败: {item_id}")
        return False
    
    # 构建 sharegpt 格式
    sharegpt_item = build_sharegpt_item(
        system_prompt, user_prompt, model_response, fsf
    )
    
    # 写入输出文件（加锁保证并发安全）
    async with lock:
        async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(sharegpt_item, ensure_ascii=False) + '\n')
        
        # 更新状态
        state.processed_ids.add(item_id)
        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens
        state.total_cost += calculate_cost(input_tokens, output_tokens)
    
    return True


async def process_batch(
    batch: list[tuple[dict, str]],
    state: ProcessingState,
    output_file: Path,
    state_file: Path,
    lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    pbar: tqdm_async = None
) -> int:
    """
    处理一个批次的数据
    
    Returns:
        成功处理的样本数
    """
    async def process_with_semaphore(item: dict, item_id: str) -> bool:
        async with semaphore:
            result = await process_single_item(
                item, item_id, state, output_file, lock
            )
            if pbar is not None:
                # 更新进度条描述显示当前花费
                pbar.set_postfix({
                    'cost': f"${state.total_cost:.4f}",
                    'tokens': state.total_input_tokens + state.total_output_tokens
                })
                pbar.update(1)
            return result
    
    # 并发处理批次内所有样本
    tasks = [
        process_with_semaphore(item, item_id)
        for item, item_id in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计成功数
    success_count = sum(1 for r in results if r is True)
    
    # 保存状态
    await save_state(state_file, state)
    
    return success_count


def register_to_dataset_info(
    dataset_info_path: str,
    dataset_name: str,
    file_path: str
) -> bool:
    """
    将数据集注册到 dataset_info.json
    
    Args:
        dataset_info_path: dataset_info.json 的路径
        dataset_name: 数据集名称
        file_path: 数据集文件路径
    
    Returns:
        是否成功注册
    """
    try:
        import json
        from pathlib import Path
        
        # 读取 dataset_info.json
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        # 计算相对路径（相对于 LlamaFactory/data 目录）
        llama_data_dir = Path(dataset_info_path).parent
        try:
            relative_path = str(Path(file_path).relative_to(llama_data_dir))
        except ValueError:
            # 如果文件不在 LlamaFactory/data 目录下，使用绝对路径
            relative_path = str(Path(file_path).resolve())
            logger.warning(f"文件不在 LlamaFactory/data 目录下，使用绝对路径: {relative_path}")
        
        # 添加到 dataset_info
        dataset_info[dataset_name] = {
            "file_name": relative_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        
        # 写回 dataset_info.json
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 数据集 '{dataset_name}' 已注册到 {dataset_info_path}")
        return True
        
    except Exception as e:
        logger.error(f"注册数据集失败: {e}")
        return False


def select_top_k_by_fsf(output_file: Path, k: int) -> list[dict]:
    """按 FSF 排序选择 top k 条数据"""
    items = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError:
                continue
    
    # 按 FSF 倒序排序
    items.sort(key=lambda x: x.get("_fsf", 0.0), reverse=True)
    
    # 选择 top k
    selected = items[:k]
    
    # 移除 _fsf 字段
    for item in selected:
        item.pop("_fsf", None)
    
    return selected





async def main(
    input_file: str = "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_fsf_1to3/train.jsonl",
    output_dir: str = "/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_gemini_cold_start",
    target_count: int = 1000,
    dataset_info_path: str = "../../LlamaFactory/data/dataset_info.json",
    resume: bool = True
):
    """
    主函数
    
    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录
        target_count: 目标处理数量（支持增量更新）
        dataset_info_path: dataset_info.json 路径
        resume: 是否断点续传
    """
    logger.info("=" * 70)
    logger.info("  Gemini CoT 重写数据处理工具")
    logger.info("=" * 70)
    logger.info(f"📁 输入文件: {input_file}")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info(f"🎯 目标数量: {target_count}")
    logger.info(f"🔄 断点续传: {resume}")
    logger.info(f"🔢 并发数: {CONCURRENT_REQUESTS}")
    logger.info(f"🤖 模型: {MODEL}")
    logger.info("=" * 70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义文件路径
    output_file = output_path / "raw_output.jsonl"
    state_file = output_path / "processing_state.json"
    train_file = output_path / "train.jsonl"
    
    # 加载或初始化状态
    if resume:
        state = await load_state(state_file)
        logger.info(f"📝 已处理 {len(state.processed_ids)} 条数据")
        logger.info(f"💰 累计成本: ${state.total_cost:.4f}")
    else:
        state = ProcessingState(
            processed_ids=set(),
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0
        )
        # 清空输出文件
        if output_file.exists():
            output_file.unlink()
    
    # 读取输入文件
    logger.info("📖 读取输入数据...")
    all_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 使用 question 作为唯一标识
                item_id = item.get("question", f"item_{idx}")
                all_items.append((item, item_id))
            except json.JSONDecodeError as e:
                logger.warning(f"第 {idx} 行 JSON 解析失败: {e}")
                continue
    
    total_items = len(all_items)
    logger.info(f"✅ 读取完成，共 {total_items} 条数据")
    
    # 过滤已处理的样本
    remaining_items = [
        (item, item_id) for item, item_id in all_items
        if item_id not in state.processed_ids
    ]
    
    # 计算需要处理的数量
    needed_count = target_count - len(state.processed_ids)
    if needed_count <= 0:
        logger.info(f"✅ 已处理 {len(state.processed_ids)} 条，达到目标数量 {target_count}")
    else:
        items_to_process = remaining_items[:needed_count]
        logger.info(f"🎯 还需处理 {needed_count} 条数据")
        
        # 创建锁和信号量
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        # 创建总进度条
        pbar = tqdm_async(
            total=len(items_to_process),
            desc="处理进度",
            unit="条",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        # 分批处理
        batches = [
            items_to_process[i:i + BATCH_SIZE]
            for i in range(0, len(items_to_process), BATCH_SIZE)
        ]
        
        logger.info(f"📦 分成 {len(batches)} 个批次处理")
        
        # 处理每个批次
        for batch_idx, batch in enumerate(batches, 1):
            success_count = await process_batch(
                batch, state, output_file, state_file, lock, semaphore, pbar
            )
            logger.debug(f"批次 {batch_idx}/{len(batches)} 完成，成功 {success_count}/{len(batch)} 条")
        
        pbar.close()
    
    logger.info("=" * 70)
    logger.info("🎉 数据处理完成!")
    logger.info(f"📊 总计处理: {len(state.processed_ids)} 条")
    logger.info(f"📊 输入 Token: {state.total_input_tokens}")
    logger.info(f"📊 输出 Token: {state.total_output_tokens}")
    logger.info(f"💰 总成本: ${state.total_cost:.4f}")
    logger.info("=" * 70)
    
    # 选择 top k
    logger.info(f"📊 按 FSF 排序选择 top-{target_count} 条数据...")
    selected_data = select_top_k_by_fsf(output_file, target_count)
    logger.info(f"✅ 选择完成，共 {len(selected_data)} 条")
    
    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(selected_data)
    
    # 保存训练集
    logger.info(f"💾 保存训练集: {train_file} ({len(selected_data)} 条)")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in selected_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info("✅ 数据保存完成")
    
    # 注册到 dataset_info.json
    logger.info("📝 注册数据集到 LlamaFactory...")
    dataset_base_name = output_path.name
    
    train_registered = register_to_dataset_info(
        dataset_info_path,
        f"{dataset_base_name}_train",
        str(train_file)
    )
    
    if train_registered:
        logger.info("🎉 数据集注册成功!")
        logger.info(f"   训练集名称: {dataset_base_name}_train")
    else:
        logger.error("数据集注册失败!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="使用 Gemini API 处理 CoT 重写任务"
    )
    parser.add_argument(
        "--input", "-i",
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_fsf_1to3/train.jsonl",
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/cot_rewrite_deepmath_gemini",
        help="输出目录"
    )
    parser.add_argument(
        "--target-count", "-n",
        type=int,
        default=1000,
        help="目标处理数量（支持增量更新）"
    )
    parser.add_argument(
        "--dataset-info", "-r",
        default="../LlamaFactory/data/dataset_info.json",
        help="dataset_info.json 路径"
    )
    parser.add_argument(
        "--resume", "-c",
        action="store_true",
        default=True,
        help="启用断点续传"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传，从头开始"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="并发请求数（默认: 10）"
    )
    
    args = parser.parse_args()
    
    # 更新并发数
    CONCURRENT_REQUESTS = args.concurrent
    
    # 运行主函数
    asyncio.run(main(
        input_file=args.input,
        output_dir=args.output,
        target_count=args.target_count,
        dataset_info_path=args.dataset_info,
        resume=not args.no_resume
    ))
