#!/usr/bin/env python3
"""
collect_results.py - 汇总多个模型的 evalscope 评测结果到一个 Excel 文件。

用法:
  python collect_results.py model_path1 model_path2 ... -o output.xlsx
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def find_latest_version_dir(model_path: str) -> Path:
    """找到 model_path/evalscope/ 下最新的 version_* 目录。"""
    evalscope_dir = Path(model_path) / 'evalscope'
    if not evalscope_dir.exists():
        return None
    version_dirs = sorted(evalscope_dir.glob('version_*'), reverse=True)
    return version_dirs[0] if version_dirs else None


def load_model_results(version_dir: Path) -> pd.DataFrame:
    """
    从版本目录加载结果，返回 DataFrame，列为 benchmark, dataset, mean, std。
    优先读 multi_run_summary.json（多次运行），其次读 summary.xlsx（单次运行）。
    """
    multi_json = version_dir / 'multi_run_summary.json'
    if multi_json.exists():
        with open(multi_json, encoding='utf-8') as f:
            data = json.load(f)
        rows = data.get('results', [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        run_cols = sorted([c for c in df.columns if c.startswith('run_')])
        keep_cols = ['benchmark', 'dataset', 'mean', 'std'] + run_cols
        df = df[[c for c in keep_cols if c in df.columns]]
        df['key'] = df['benchmark'] + ' / ' + df['dataset']
        return df.set_index('key').drop(columns=['benchmark', 'dataset'])

    summary_xlsx = version_dir / 'summary.xlsx'
    if summary_xlsx.exists():
        df = pd.read_excel(summary_xlsx)
        df['key'] = df['benchmark'] + ' / ' + df['dataset']
        df = df.set_index('key')[['score']].rename(columns={'score': 'mean'})
        df['std'] = 0.0
        return df

    return None


def collect(model_paths: list, output: str, num_runs: int):
    model_results = {}
    for mp in model_paths:
        name = Path(mp).name
        vdir = find_latest_version_dir(mp)
        if vdir is None:
            print(f"[跳过] 未找到 evalscope 结果目录: {mp}")
            continue
        df = load_model_results(vdir)
        if df is None:
            print(f"[跳过] 无法读取结果: {vdir}")
            continue
        model_results[name] = df
        print(f"[读取] {name} <- {vdir.name}")

    if not model_results:
        print("没有找到任何可汇总的结果，退出。")
        sys.exit(1)

    # ---- 构造对比表 ----
    # 所有模型的 benchmark/dataset 行取并集
    all_keys = []
    seen = set()
    for df in model_results.values():
        for k in df.index:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # 拆分 key 回 benchmark / dataset
    split = pd.Series(all_keys, name='key').str.split(' / ', n=1, expand=True)
    summary_df = pd.DataFrame({
        'benchmark': split[0].values,
        'dataset':   split[1].values,
    })

    for name, df in model_results.items():
        summary_df[f'{name}_mean'] = df['mean'].reindex(all_keys).values
        if num_runs > 1:
            summary_df[f'{name}_std%'] = (
                df['std'].reindex(all_keys) /
                df['mean'].reindex(all_keys).replace(0, float('nan')) * 100
            ).round(2).values
            run_cols = sorted([c for c in df.columns if c.startswith('run_')])
            for rc in run_cols:
                summary_df[f'{name}_{rc}'] = df[rc].reindex(all_keys).values

    # ---- 写入 Excel ----
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='汇总', index=False)

        # 每个模型单独一个 sheet（含完整字段）
        for name, df in model_results.items():
            df_full = df.reset_index().rename(columns={'key': 'benchmark/dataset'})
            safe_name = name[:31]  # Excel sheet 名最长 31 字符
            df_full.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"\n[完成] 汇总结果已保存至: {output_path}")
    print(f"       包含 {len(model_results)} 个模型 x {len(all_keys)} 条评测指标")


def main():
    parser = argparse.ArgumentParser(description='汇总多模型 evalscope 评测结果')
    parser.add_argument('model_paths', nargs='+', help='模型目录路径列表')
    parser.add_argument('-o', '--output', default='vllm_logs/batch_summary.xlsx',
                        help='输出 Excel 路径（默认: vllm_logs/batch_summary.xlsx）')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='每个模型的运行次数，>1 时附加 std 列（默认: 1）')
    args = parser.parse_args()
    collect(args.model_paths, args.output, args.num_runs)


if __name__ == '__main__':
    main()
