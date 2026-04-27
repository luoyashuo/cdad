#!/usr/bin/env python3
"""
Step 2: plot original vs rewrite PPL distributions from saved JSONL results.

By default, for each sample we keep:
- 1 original CoT
- 1 rewrite CoT (rewrite_idx=0)

This yields a balanced comparison such as 2000 original vs 2000 rewrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = (
    "/home/luoys/cdad/analysis/results/"
    "cot_ppl_qwen2_5_7b_instruct_rollout_output_2k.jsonl"
)
DEFAULT_OUTPUT = (
    "/home/luoys/cdad/analysis/figures/"
    "cot_ppl_qwen2_5_7b_instruct_rollout_output_2k_one_rewrite.png"
)

# NeurIPS-friendly palette
PALETTE = [
    "#2878b5",  # deep blue
    "#9ac9db",  # light blue
    "#f8ac8c",  # orange
    "#c82423",  # red
    "#ff8884",  # pink
]
ORIGINAL_COLOR = "#c82423"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot balanced original/rewrite PPL distributions.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Saved PPL JSONL from step 1.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Figure path.")
    parser.add_argument(
        "--rewrite-idx",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Use exactly one rewrite per sample, selected by rewrite_idx. "
             "Pass multiple values to plot 1xN subplots (default: 0 1 2).",
    )
    parser.add_argument(
        "--max-ppl",
        type=float,
        default=None,
        help="Optional PPL clip for plotting only.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_balanced_df(df: pd.DataFrame, rewrite_idx: int) -> pd.DataFrame:
    original_df = df[df["variant_type"] == "original"].copy()
    rewrite_df = df[
        (df["variant_type"] == "rewrite") & (df["rewrite_idx"] == rewrite_idx)
    ].copy()

    original_ids = set(original_df["sample_idx"].tolist())
    rewrite_ids = set(rewrite_df["sample_idx"].tolist())
    common_ids = sorted(original_ids & rewrite_ids)

    original_df = original_df[original_df["sample_idx"].isin(common_ids)].copy()
    rewrite_df = rewrite_df[rewrite_df["sample_idx"].isin(common_ids)].copy()
    return pd.concat([original_df, rewrite_df], ignore_index=True)


def plot_single(ax, balanced_df: pd.DataFrame, rewrite_idx: int, color: str) -> None:
    """Plot one original-vs-rewrite histogram on a given Axes."""
    # Temporarily override the variant_type so histplot picks up the hue correctly
    plot_df = balanced_df.copy()
    # Ensure consistent hue order and colors
    palette = {"original": ORIGINAL_COLOR, "rewrite": color}

    sns.histplot(
        data=plot_df,
        x="ppl_plot",
        hue="variant_type",
        hue_order=["original", "rewrite"],
        palette=palette,
        stat="percent",
        common_norm=False,
        bins=60,
        kde=True,
        alpha=0.35,
        ax=ax,
        line_kws={"linewidth": 1.2},
        legend=True,
    )

    ax.set_xlabel("PPL", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10)
    sns.despine(ax=ax, top=True, right=True)

    # Clean legend: no title, minimal frame
    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.set_title(None)
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor("gray")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input}")

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["ppl"])].copy()

    rewrite_indices = args.rewrite_idx
    if isinstance(rewrite_indices, int):
        rewrite_indices = [rewrite_indices]

    # Build balanced DataFrames for each requested rewrite_idx
    balanced_dfs = {}
    for ridx in rewrite_indices:
        bdf = build_balanced_df(df, ridx)
        if bdf.empty:
            raise RuntimeError(f"No rows remain after selecting rewrite_idx={ridx}")
        if args.max_ppl is not None:
            bdf["ppl_plot"] = bdf["ppl"].clip(upper=args.max_ppl)
        else:
            bdf["ppl_plot"] = bdf["ppl"]
        balanced_dfs[ridx] = bdf

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # NeurIPS / clean paper style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
    sns.set_theme(style="white", context="paper", rc={"axes.edgecolor": "black", "axes.linewidth": 0.8})

    n = len(rewrite_indices)
    if n == 1:
        # Single plot, suitable for single-column width
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        axes = [ax]
    else:
        # 1-row subplots; width scales with number of panels
        # Typical double-column width ~7.0in; each panel ~2.2-2.4in
        fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 0.8, 2.2), sharey=True)
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    for idx, ridx in enumerate(rewrite_indices):
        ax = axes[idx]
        color = PALETTE[idx % len(PALETTE)]
        plot_single(ax, balanced_dfs[ridx], ridx, color)

    # Shared y-label on the leftmost axis
    axes[0].set_ylabel("Percentage", fontsize=11)

    # Remove redundant legends from all but the last panel to save space
    if n > 1:
        for ax in axes[:-1]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        # Adjust last legend position to upper-right
        if axes[-1].get_legend() is not None:
            axes[-1].get_legend().set_bbox_to_anchor((1.02, 1.0))
            axes[-1].get_legend().set_loc("upper right")

    plt.tight_layout(w_pad=0.8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save filtered data and summary for the first rewrite_idx only
    # (keeps backward-compatible file naming)
    first_ridx = rewrite_indices[0]
    balanced_df = balanced_dfs[first_ridx]
    original_df = balanced_df[balanced_df["variant_type"] == "original"].copy()
    rewrite_df = balanced_df[balanced_df["variant_type"] == "rewrite"].copy()

    filtered_jsonl = output_path.with_suffix(".selected.jsonl")
    filtered_rows = balanced_df.sort_values(["sample_idx", "variant_type", "rewrite_idx"]).to_dict("records")
    with open(filtered_jsonl, "w", encoding="utf-8") as f:
        for row in filtered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = output_path.with_suffix(".summary.csv")
    pd.DataFrame(
        [
            {
                "variant_type": "original",
                "count": len(original_df),
                "mean_ppl": float(original_df["ppl"].mean()),
                "median_ppl": float(original_df["ppl"].median()),
                "std_ppl": float(original_df["ppl"].std()),
            },
            {
                "variant_type": "rewrite",
                "count": len(rewrite_df),
                "mean_ppl": float(rewrite_df["ppl"].mean()),
                "median_ppl": float(rewrite_df["ppl"].median()),
                "std_ppl": float(rewrite_df["ppl"].std()),
            },
        ]
    ).to_csv(summary_csv, index=False)

    print(f"figure saved to: {output_path}")
    print(f"selected rows saved to: {filtered_jsonl}")
    print(f"summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
