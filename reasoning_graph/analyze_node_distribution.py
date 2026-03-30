#!/usr/bin/env python3
"""
统计 FSF 输出文件中 node 类型的分布

Usage:
    python analyze_node_distribution.py \
        --input_file /data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl \
        --output_file node_distribution.json
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np


def analyze_node_distribution(input_file: str, output_file: str = None, plot: bool = True):
    """
    分析 node 类型的分布
    """
    print(f"Reading input file: {input_file}")
    
    # 统计数据
    stats = {
        'total_questions': 0,
        'total_thoughts': 0,
        'successful_parses': 0,
        'failed_parses': 0,
        'total_nodes': 0,
        'failed_nodes': 0,
        'successful_nodes': 0,
        'other_nodes': 0,
        'node_count_distribution': defaultdict(int),  # 每个 thought 的 node 数量分布
        'fsf_distribution': [],  # FSF 值列表
        'node_type_by_thought': {  # 每个 thought 的 node 类型统计
            'failed_only': 0,      # 只有 failed nodes
            'successful_only': 0,  # 只有 successful nodes
            'mixed': 0,            # 混合类型
            'other_present': 0,    # 包含 other 类型
        }
    }
    
    # 读取文件
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    stats['total_questions'] = len(data)
    
    # 分析每个问题
    for question_idx, item in enumerate(data):
        question = item.get('question', '')
        reasoning = item.get('reasoning', [])
        
        for thought_idx, r in enumerate(reasoning):
            stats['total_thoughts'] += 1
            
            # 检查是否解析成功
            if r.get('parse_failed') or r.get('fsf') is None:
                stats['failed_parses'] += 1
                continue
            
            stats['successful_parses'] += 1
            
            node_count_dict = r.get('node_count_dict', {})
            if not node_count_dict:
                continue
            
            total = node_count_dict.get('total', 0)
            failed = node_count_dict.get('failed', 0)
            successful = node_count_dict.get('successful', 0)
            other = node_count_dict.get('other', 0)
            
            # 累加统计
            stats['total_nodes'] += total
            stats['failed_nodes'] += failed
            stats['successful_nodes'] += successful
            stats['other_nodes'] += other
            
            # node 数量分布
            stats['node_count_distribution'][total] += 1
            
            # FSF 分布
            fsf = r.get('fsf')
            if fsf is not None:
                stats['fsf_distribution'].append(fsf)
            
            # node 类型分类
            if other > 0:
                stats['node_type_by_thought']['other_present'] += 1
            elif failed > 0 and successful > 0:
                stats['node_type_by_thought']['mixed'] += 1
            elif failed > 0:
                stats['node_type_by_thought']['failed_only'] += 1
            elif successful > 0:
                stats['node_type_by_thought']['successful_only'] += 1
    
    # 计算百分比
    if stats['total_nodes'] > 0:
        stats['failed_pct'] = 100 * stats['failed_nodes'] / stats['total_nodes']
        stats['successful_pct'] = 100 * stats['successful_nodes'] / stats['total_nodes']
        stats['other_pct'] = 100 * stats['other_nodes'] / stats['total_nodes']
    else:
        stats['failed_pct'] = 0
        stats['successful_pct'] = 0
        stats['other_pct'] = 0
    
    # FSF 统计
    if stats['fsf_distribution']:
        stats['fsf_mean'] = np.mean(stats['fsf_distribution'])
        stats['fsf_median'] = np.median(stats['fsf_distribution'])
        stats['fsf_min'] = np.min(stats['fsf_distribution'])
        stats['fsf_max'] = np.max(stats['fsf_distribution'])
        stats['fsf_std'] = np.std(stats['fsf_distribution'])
    
    # 打印报告
    print_report(stats)
    
    # 保存详细统计
    if output_file:
        save_stats(stats, output_file)
    
    # 绘制图表
    if plot:
        plot_distributions(stats, input_file)
    
    return stats


def print_report(stats: Dict[str, Any]):
    """打印统计报告"""
    print("\n" + "="*70)
    print("NODE DISTRIBUTION ANALYSIS REPORT")
    print("="*70)
    
    print(f"\n📊 基本统计:")
    print(f"  总问题数: {stats['total_questions']}")
    print(f"  总 thought 数: {stats['total_thoughts']}")
    print(f"  解析成功: {stats['successful_parses']} ({100*stats['successful_parses']/max(1,stats['total_thoughts']):.2f}%)")
    print(f"  解析失败: {stats['failed_parses']} ({100*stats['failed_parses']/max(1,stats['total_thoughts']):.2f}%)")
    
    print(f"\n📈 Node 类型分布 (总计: {stats['total_nodes']} nodes):")
    print(f"  ├── Failed (lightpink): {stats['failed_nodes']} ({stats['failed_pct']:.2f}%)")
    print(f"  ├── Successful (lightblue): {stats['successful_nodes']} ({stats['successful_pct']:.2f}%)")
    print(f"  └── Other: {stats['other_nodes']} ({stats['other_pct']:.2f}%)")
    
    print(f"\n📉 Thought 分类统计:")
    print(f"  ├── 纯 Failed nodes: {stats['node_type_by_thought']['failed_only']}")
    print(f"  ├── 纯 Successful nodes: {stats['node_type_by_thought']['successful_only']}")
    print(f"  ├── 混合类型: {stats['node_type_by_thought']['mixed']}")
    print(f"  └── 包含 Other 类型: {stats['node_type_by_thought']['other_present']}")
    
    print(f"\n📊 FSF 统计:")
    if 'fsf_mean' in stats:
        print(f"  ├── 平均值: {stats['fsf_mean']:.4f}")
        print(f"  ├── 中位数: {stats['fsf_median']:.4f}")
        print(f"  ├── 最小值: {stats['fsf_min']:.4f}")
        print(f"  ├── 最大值: {stats['fsf_max']:.4f}")
        print(f"  └── 标准差: {stats['fsf_std']:.4f}")
    
    print(f"\n📊 Node 数量分布 (每个 thought):")
    sorted_dist = sorted(stats['node_count_distribution'].items())
    for node_count, frequency in sorted_dist[:20]:  # 只显示前 20 个
        bar = "█" * int(frequency / max(1, stats['successful_parses']) * 50)
        print(f"  {node_count:3d} nodes: {frequency:5d} {bar}")
    
    if len(sorted_dist) > 20:
        print(f"  ... 还有 {len(sorted_dist) - 20} 种其他数量")
    
    print("\n" + "="*70)


def save_stats(stats: Dict[str, Any], output_file: str):
    """保存统计结果到 JSON"""
    # 移除不能序列化的数据
    save_stats = {k: v for k, v in stats.items() if k != 'fsf_distribution'}
    save_stats['fsf_distribution'] = {
        'count': len(stats.get('fsf_distribution', [])),
        'mean': stats.get('fsf_mean'),
        'median': stats.get('fsf_median'),
        'min': stats.get('fsf_min'),
        'max': stats.get('fsf_max'),
        'std': stats.get('fsf_std'),
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 统计结果已保存到: {output_file}")


def plot_distributions(stats: Dict[str, Any], input_file: str):
    """绘制分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Node Distribution Analysis', fontsize=16)
        
        # 1. Node 类型饼图
        ax1 = axes[0, 0]
        labels = ['Failed\n(lightpink)', 'Successful\n(lightblue)', 'Other']
        sizes = [stats['failed_nodes'], stats['successful_nodes'], stats['other_nodes']]
        colors = ['#ff9999', '#66b3ff', '#c2c2f0']
        explode = (0.05, 0.05, 0.1) if stats['other_nodes'] > 0 else (0.05, 0.05, 0)
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title('Node Type Distribution')
        
        # 2. Node 数量分布直方图
        ax2 = axes[0, 1]
        node_counts = []
        for count, freq in stats['node_count_distribution'].items():
            node_counts.extend([count] * freq)
        
        if node_counts:
            ax2.hist(node_counts, bins=range(0, max(node_counts)+2), alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Number of Nodes per Thought')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Node Count Distribution per Thought')
            ax2.axvline(np.mean(node_counts), color='r', linestyle='--', label=f'Mean: {np.mean(node_counts):.1f}')
            ax2.legend()
        
        # 3. FSF 分布直方图
        ax3 = axes[1, 0]
        if stats.get('fsf_distribution'):
            ax3.hist(stats['fsf_distribution'], bins=50, alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('FSF (Failure Step Fraction)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('FSF Distribution')
            ax3.axvline(stats.get('fsf_mean', 0), color='r', linestyle='--', label=f"Mean: {stats.get('fsf_mean', 0):.3f}")
            ax3.legend()
        
        # 4. Thought 类型柱状图
        ax4 = axes[1, 1]
        thought_types = ['Failed Only', 'Successful Only', 'Mixed', 'Has Other']
        thought_counts = [
            stats['node_type_by_thought']['failed_only'],
            stats['node_type_by_thought']['successful_only'],
            stats['node_type_by_thought']['mixed'],
            stats['node_type_by_thought']['other_present']
        ]
        colors_bar = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']
        bars = ax4.bar(thought_types, thought_counts, color=colors_bar, edgecolor='black')
        ax4.set_ylabel('Count')
        ax4.set_title('Thought Classification by Node Types')
        ax4.tick_params(axis='x', rotation=15)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = input_file.replace('.jsonl', '_node_distribution.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 分布图已保存到: {plot_file}")
        
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze node distribution in FSF output file")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/hf_home/datasets/Yaseal/CDAD_EXP/deepmath_rl_5k/train_fsf.jsonl",
        help="Input FSF JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for statistics (optional)"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plotting"
    )
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = args.input_file.replace('.jsonl', '_stats.json')
    
    analyze_node_distribution(
        input_file=args.input_file,
        output_file=args.output_file,
        plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
