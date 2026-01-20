#!/usr/bin/env python3
"""
数据筛选脚本 - 支持双等级过滤

过滤等级说明：
  等级1（宽松）: FSF最低集合不能同时与RV最高和CG最高集合都有交集
               （即：最多只能与其中一个有交集）
  等级2（严格）: FSF最低集合与RV最高、CG最高集合都没有交集

输出格式：
  保留完整的原始数据格式，并添加：
  - selected_by_cognitive_difficulty: 按CG最高筛选的thought和full_response
  - selected_by_reasoning_verbosity: 按RV最高筛选的thought和full_response  
  - selected_by_lowest_fsf: 按FSF最低筛选的thought和full_response
  - selection_metadata: 选择过程的元数据

输入：待处理文件（默认固定路径）
输出：用户指定的输出路径
"""

import json
import argparse
import os
import collections
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import time
import signal
from loguru import logger


DEFAULT_INPUT_FILE = "/mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_part.jsonl"

# 全局变量用于传递过滤等级
FILTER_LEVEL = 2

def init_worker():
    """工作进程初始化"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="数据筛选脚本 - 支持双等级过滤逻辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 等级1过滤（宽松）
  python select_diff_data.py -o /path/to/output.jsonl --level 1

  # 等级2过滤（严格）
  python select_diff_data.py -o /path/to/output.jsonl --level 2

  # 查看FSF分布分析
  python select_diff_data.py -o /tmp/output.jsonl --analyze-only

过滤等级说明:
  等级1（宽松）: FSF最低集合不能同时与RV最高和CG最高集合都有交集
  等级2（严格）: FSF最低集合与RV最高、CG最高集合都没有交集
        """
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出JSONL文件路径（必需）"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT_FILE,
        help=f"输入JSONL文件路径（默认: {DEFAULT_INPUT_FILE}）"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        choices=[1, 2],
        default=2,
        help="过滤等级：1=宽松, 2=严格（默认: 2）"
    )
    parser.add_argument(
        "--processes", "-p",
        type=int,
        default=None,
        help="进程数（默认使用所有可用CPU核心）"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100,
        help="每个进程处理的chunk大小（默认: 100）"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="仅分析FSF分布，不进行筛选"
    )
    return parser.parse_args()

def read_jsonl_file(file_path: str, show_progress: bool = False, desc: str = "读取进度") -> List[str]:
    """读取JSONL文件，返回内容列表
    
    Args:
        file_path: JSONL文件路径
        show_progress: 是否显示进度条
        desc: 进度条描述
    """
    if show_progress:
        # 先计算总行数
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # 再次打开文件读取，显示进度
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc=desc, unit="行"):
                lines.append(line.strip())
    else:
        # 不显示进度，直接读取
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line.strip())
    
    return lines

def analyze_item(item_str: str) -> Tuple[bool, List[float]]:
    """
    分析单个item，返回(FSF是否有变化, FSF值列表)
    FSF有变化 = 不全为0
    """
    try:
        item = json.loads(item_str)
        if 'reasoning' not in item or not item['reasoning']:
            return False, []

        # 提取所有FSF值
        fsf_values = []
        for thought in item['reasoning']:
            reasoning_graph = thought.get('reasoning_graph', [])
            fsf = reasoning_graph[0].get('fsf', 0.0) if reasoning_graph else 0.0
            fsf_values.append(float(fsf))

        # 判断FSF是否有变化（不全为0）
        has_variance = any(fsf != 0.0 for fsf in fsf_values)

        return has_variance, fsf_values
    except Exception as e:
        logger.warning(f"分析item时出错: {e}")
        return False, []

def analyze_fsf_distribution(input_path: str, processes: int = None) -> Tuple[List[int], Dict]:
    """
    并行分析FSF分布
    返回(需要处理的行索引列表, 统计信息)
    """
    logger.info("📊 分析FSF分布中...")

    # 读取文件
    logger.info("读取输入文件...")
    lines = read_jsonl_file(input_path, show_progress=True, desc="读取进度")
    total_lines = len(lines)

    # 设置进程数
    if processes is None:
        processes = min(cpu_count(), 8)

    logger.info(f"使用 {processes} 个进程并行处理 {total_lines} 行数据...")

    start_time = time.time()

    with Pool(processes=processes, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(analyze_item, lines, chunksize=100),
            total=total_lines,
            desc="分析进度",
            unit="行"
        ))

    # 处理结果
    valid_lines = []
    stats = {
        'total_items': 0,
        'items_with_variance': 0,
        'all_zero_items': 0,
        'thought_length_dist': collections.Counter(),
    }

    for line_idx, (has_variance, fsf_values) in enumerate(results):
        if not fsf_values:
            continue

        stats['total_items'] += 1
        num_thoughts = len(fsf_values)
        stats['thought_length_dist'][num_thoughts] += 1

        if has_variance:
            valid_lines.append(line_idx)
            stats['items_with_variance'] += 1
        else:
            stats['all_zero_items'] += 1

    elapsed_time = time.time() - start_time

    logger.info(f"\n📈 FSF分布分析结果:")
    logger.info(f"  分析用时: {elapsed_time:.2f} 秒")
    logger.info(f"  总行数: {stats['total_items']}")
    logger.info(f"  FSF全是0的items: {stats['all_zero_items']}")
    logger.info(f"  FSF不全为0的items: {stats['items_with_variance']}")
    logger.info(f"  筛选比例: {stats['items_with_variance']/stats['total_items']*100:.1f}%")

    logger.info(f"\nThoughts数量分布:")
    for length, count in sorted(stats['thought_length_dist'].items()):
        logger.info(f"    {length} thoughts: {count} items")

    return valid_lines, stats

def get_thought_metrics(thought: Dict[str, Any]) -> Tuple[int, int, float]:
    """提取thought的三个指标"""
    cg = thought.get("Cognitive_Difficulty", {}).get("level", 0)
    rv = thought.get("Reasoning_Verbosity", {}).get("level", 0)

    reasoning_graph = thought.get("reasoning_graph", [])
    fsf = reasoning_graph[0].get("fsf", 0.0) if reasoning_graph else 0.0

    return cg, rv, fsf

def process_item(item_str: str) -> Optional[Dict[str, Any]]:
    """
    处理单个item，根据全局变量FILTER_LEVEL进行过滤
    
    FILTER_LEVEL=1: 宽松等级 - FSF最低集合不能同时与RV最高和CG最高集合都有交集
    FILTER_LEVEL=2: 严格等级 - FSF最低集合与RV最高、CG最高集合都没有交集
    """
    global FILTER_LEVEL
    
    try:
        item = json.loads(item_str)
        if "reasoning" not in item or not item['reasoning']:
            return None

        reasoning = item["reasoning"]
        num_thoughts = len(reasoning)

        if num_thoughts < 2:
            return None

        # 提取所有thought的指标
        thought_metrics = []
        for idx, thought in enumerate(reasoning):
            cg, rv, fsf = get_thought_metrics(thought)
            thought_metrics.append({
                'index': idx,
                'cg': cg,
                'rv': rv,
                'fsf': fsf,
                'thought': thought
            })

        # Step 1: 找出FSF最低的thought(s)的索引集合
        min_fsf = min(tm['fsf'] for tm in thought_metrics)
        fsf_min_indices = {tm['index'] for tm in thought_metrics if tm['fsf'] == min_fsf}

        # Step 2: 找出CG最高的thought(s)的索引集合
        max_cg = max(tm['cg'] for tm in thought_metrics)
        cg_max_indices = {tm['index'] for tm in thought_metrics if tm['cg'] == max_cg}

        # Step 3: 找出RV最高的thought(s)的索引集合
        max_rv = max(tm['rv'] for tm in thought_metrics)
        rv_max_indices = {tm['index'] for tm in thought_metrics if tm['rv'] == max_rv}

        # Step 4: 根据等级检查冲突
        cg_has_conflict = len(cg_max_indices & fsf_min_indices) > 0
        rv_has_conflict = len(rv_max_indices & fsf_min_indices) > 0

        if FILTER_LEVEL == 2:
            # 等级2：严格 - FSF最低不能与CG最高或RV最高有交集
            if cg_has_conflict or rv_has_conflict:
                return None
        elif FILTER_LEVEL == 1:
            # 等级1：宽松 - FSF最低不能同时与CG最高和RV最高都有交集
            if cg_has_conflict and rv_has_conflict:
                return None
        else:
            raise ValueError(f"不支持的过滤等级: {FILTER_LEVEL}")

        # Step 5: 无冲突，保留该item
        # 从对应集合中选择一个thought（优先选择索引最小的）
        cg_selected_idx = min(cg_max_indices)
        rv_selected_idx = min(rv_max_indices)
        fsf_selected_idx = min(fsf_min_indices)

        # 获取对应的thoughts
        cg_thought = thought_metrics[cg_selected_idx]['thought']
        rv_thought = thought_metrics[rv_selected_idx]['thought']
        fsf_thought = thought_metrics[fsf_selected_idx]['thought']

        # 构建输出，保留所有原始字段
        result = item.copy()
        
        # 保留筛选得到的thoughts和full_response
        result['selected_by_cognitive_difficulty'] = {
            'thought': cg_thought.get('thought', ''),
            'full_response': cg_thought.get('full_response', ''),
            'Cognitive_Difficulty': cg_thought.get('Cognitive_Difficulty', {}),
            'Reasoning_Verbosity': cg_thought.get('Reasoning_Verbosity', {}),
            'reasoning_graph': cg_thought.get('reasoning_graph', []),
            'index': cg_selected_idx
        }
        
        result['selected_by_reasoning_verbosity'] = {
            'thought': rv_thought.get('thought', ''),
            'full_response': rv_thought.get('full_response', ''),
            'Cognitive_Difficulty': rv_thought.get('Cognitive_Difficulty', {}),
            'Reasoning_Verbosity': rv_thought.get('Reasoning_Verbosity', {}),
            'reasoning_graph': rv_thought.get('reasoning_graph', []),
            'index': rv_selected_idx
        }
        
        result['selected_by_lowest_fsf'] = {
            'thought': fsf_thought.get('thought', ''),
            'full_response': fsf_thought.get('full_response', ''),
            'Cognitive_Difficulty': fsf_thought.get('Cognitive_Difficulty', {}),
            'Reasoning_Verbosity': fsf_thought.get('Reasoning_Verbosity', {}),
            'reasoning_graph': fsf_thought.get('reasoning_graph', []),
            'index': fsf_selected_idx
        }

        # metadata记录选择信息
        result['selection_metadata'] = {
            "filter_level": FILTER_LEVEL,
            "total_thoughts": num_thoughts,
            "fsf_min_value": min_fsf,
            "fsf_min_indices": sorted(fsf_min_indices),
            "cg_max_value": max_cg,
            "cg_max_indices": sorted(cg_max_indices),
            "rv_max_value": max_rv,
            "rv_max_indices": sorted(rv_max_indices),
            "cg_selected_index": cg_selected_idx,
            "rv_selected_index": rv_selected_idx,
            "fsf_selected_index": fsf_selected_idx,
            "cg_has_conflict_with_fsf": cg_has_conflict,
            "rv_has_conflict_with_fsf": rv_has_conflict,
            "filtered": False
        }

        return result

    except Exception as e:
        logger.warning(f"处理item时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_data(input_path: str, output_path: str, valid_line_indices: List[int],
                level: int, processes: int = None, chunk_size: int = 100) -> Tuple[int, float]:
    """
    多进程处理数据
    
    返回: (成功处理的items数, 用时)
    """
    global FILTER_LEVEL
    FILTER_LEVEL = level
    
    if processes is None:
        processes = min(cpu_count(), 8)

    logger.info(f"🚀 开始多进程处理数据（过滤等级: {level}）...")
    logger.info(f"  使用进程数: {processes}")
    logger.info(f"  Chunk大小: {chunk_size}")

    # 读取文件
    lines = read_jsonl_file(input_path)

    # 只保留FSF不全为0的行
    processing_data = []
    for idx, line in enumerate(lines):
        if idx in valid_line_indices:
            processing_data.append(line)

    total_items = len(processing_data)
    logger.info(f"  需要处理的items: {total_items}")

    start_time = time.time()
    processed_count = 0

    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        with Pool(processes=processes, initializer=init_worker, maxtasksperchild=1000) as pool:
            # 使用tqdm显示进度
            results = pool.imap(process_item, processing_data, chunksize=chunk_size)

            with tqdm(total=total_items, desc="处理进度", unit="items") as pbar:
                for result in results:
                    if result:
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        processed_count += 1

                        if processed_count % 100 == 0:
                            outfile.flush()
                    
                    # 无论是否筛选成功，都要更新进度条
                    pbar.update(1)

    elapsed_time = time.time() - start_time

    logger.info(f"✅ 处理完成！")
    logger.info(f"  处理用时: {elapsed_time:.2f} 秒")
    logger.info(f"  处理速度: {processed_count/elapsed_time:.1f} items/秒")
    logger.info(f"  成功筛选: {processed_count} items")
    logger.info(f"  输出文件: {output_path}")
    
    return processed_count, elapsed_time

def main():
    args = parse_arguments()

    logger.info("\n" + "="*70)
    logger.info("  数据筛选脚本 - 支持双等级过滤")
    logger.info("="*70)
    logger.info(f"📁 输入文件: {args.input}")
    logger.info(f"📁 输出文件: {args.output}")
    logger.info(f"🔢 过滤等级: {args.level} ({'宽松' if args.level == 1 else '严格'})")
    logger.info(f"🔢 进程数: {args.processes or min(cpu_count(), 8)}")
    logger.info(f"📦 Chunk大小: {args.chunk_size}")
    logger.info("="*70)

    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件 '{args.input}' 不存在！")
        return 1

    try:
        total_start = time.time()

        # 第一步：分析FSF分布，筛选FSF不全为0的items
        valid_line_indices, stats = analyze_fsf_distribution(args.input, args.processes)

        if args.analyze_only:
            logger.info("\n✅ 仅分析完成，未进行处理")
            return 0

        if not valid_line_indices:
            logger.warning("没有找到FSF不全为0的items！")
            return 0

        # 第二步：处理数据，选择thoughts
        filtered_count, process_time = process_data(
            args.input, args.output, valid_line_indices,
            args.level, args.processes, args.chunk_size
        )

        logger.info(f"\n📊 最终筛选统计:")
        logger.info(f"  输入总items: {stats['total_items']}")
        logger.info(f"  FSF不全为0: {stats['items_with_variance']}")
        logger.info(f"  过滤后保留: {filtered_count}")
        logger.info(f"  最终保留率: {filtered_count/stats['total_items']*100:.1f}%")
        logger.info(f"  处理用时: {process_time:.2f} 秒")

        total_time = time.time() - total_start
        logger.info(f"\n🎉 全部完成！总用时: {total_time:.2f} 秒")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\n用户中断处理")
        return 130
    except Exception as e:
        logger.error(f"\n处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
"""
python reasoning_graph/select_diff_data.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge.jsonl \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level2.jsonl \
    --level 2 \
    -p 32
python reasoning_graph/select_diff_data.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge.jsonl \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_math_all/train_merge_level1.jsonl \
    --level 1 \
    -p 32

python reasoning_graph/select_diff_data.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge.jsonl \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level2.jsonl \
    --level 2 \
    -p 32
python reasoning_graph/select_diff_data.py \
    -i /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge.jsonl \
    -o /mnt/moonfs/luoyashuo-ksyun/research/cdad/data/omnithought_code_all/train_merge_level1.jsonl \
    --level 1 \
    -p 32
"""