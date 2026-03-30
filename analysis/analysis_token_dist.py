"""
Analyze token length distribution of model responses by bench (difficulty level) x model.
Data source: evalscope prediction jsonl files, field messages[0]['content'].
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TOKENIZER_PATH = "/data/modelscope_cache/models/LLM-Research/Llama-3.2-3B-Instruct"

# Each entry: (label, prediction_dir)
MODEL_GROUPS = {
    "3B": [
        ("3B-Instruct\n(baseline)",
         "/data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
         "/evalscope/version_20260324_213517/MATH500/20260324_213517"
         "/predictions/Llama-3.2-3B-Instruct"),
        ("3B-SFT\nmix-r100",
         "/data/cdad/checkpoints/rewrite_cot/llama3_3b_instruct_full_sft_mix_30k_r100"
         "/evalscope/version_20260329_111401/MATH500/20260329_111401"
         "/predictions/llama3_3b_instruct_full_sft_mix_30k_r100"),
        ("3B-SFT\nrds-r50",
         "/data/cdad/checkpoints/rewrite_cot/llama3_3b_instruct_full_sft_rds_30k_r50"
         "/evalscope/version_20260329_183955/MATH500/20260329_194732"
         "/predictions/llama3_3b_instruct_full_sft_rds_30k_r50"),
        ("3B-vanilla\nSFT-30k",
         "/data/cdad/checkpoints/vallina_sft/llama3_3b_instruct_vallina_full_sft_30k"
         "/evalscope/version_20260324_143124/MATH500/20260324_143124"
         "/predictions/llama3_3b_instruct_vallina_full_sft_30k"),
    ],
    "1B": [
        ("1B-Instruct\n(baseline)",
         "/data/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
         "/evalscope/version_20260324_213517/MATH500/20260324_213517"
         "/predictions/Llama-3.2-1B-Instruct"),
        ("1B-SFT\nmix-r100",
         "/data/cdad/checkpoints/rewrite_cot/llama3_1b_instruct_full_sft_mix_30k_r100"
         "/evalscope/version_20260322_113012/MATH500/20260323_004709"
         "/predictions/llama3_1b_instruct_full_sft_mix_30k_r100"),
        ("1B-vanilla\nSFT-30k",
         "/data/cdad/checkpoints/vallina_sft/llama3_1b_instruct_vallina_full_sft_30k"
         "/evalscope/version_20260324_005510/MATH500/20260324_005510"
         "/predictions/llama3_1b_instruct_vallina_full_sft_30k"),
    ],
}

OUTPUT_DIR = Path(__file__).parent / "figures"

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

def load_predictions(pred_dir: str) -> list[dict]:
    """Load all jsonl files under pred_dir, return list of records."""
    records = []
    for fpath in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        level = Path(fpath).stem.replace("math_500_", "")  # e.g. "Level 1"
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                content = d["messages"][0]["content"]
                records.append({"level": level, "content": content})
    return records


def build_dataframe(tokenizer, groups: dict) -> pd.DataFrame:
    rows = []
    for group_name, models in groups.items():
        for label, pred_dir in models:
            if not os.path.isdir(pred_dir):
                print(f"[WARN] directory not found: {pred_dir}")
                continue
            records = load_predictions(pred_dir)
            print(f"  {label.replace(chr(10), ' ')}: {len(records)} samples")
            # Batch tokenize
            contents = [r["content"] for r in records]
            encoded = tokenizer(contents, add_special_tokens=False)
            for r, ids in zip(records, encoded["input_ids"]):
                rows.append({
                    "group": group_name,
                    "model": label,
                    "level": r["level"],
                    "token_len": len(ids),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

LEVEL_ORDER = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
LEVEL_PALETTE = sns.color_palette("husl", 5)


def plot_violins_by_model(df: pd.DataFrame, group: str, ax: plt.Axes):
    """Violin plot: x=model, hue=level, for a single model group."""
    gdf = df[df["group"] == group].copy()
    models_order = [label for label, _ in MODEL_GROUPS[group] if label in gdf["model"].unique()]
    levels = [l for l in LEVEL_ORDER if l in gdf["level"].unique()]

    sns.violinplot(
        data=gdf,
        x="model", y="token_len",
        hue="level", hue_order=levels,
        order=models_order,
        palette=LEVEL_PALETTE[: len(levels)],
        inner="quartile",
        density_norm="width",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title(f"Token Length Distribution — {group} Models (MATH500)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Token Count")
    ax.legend(title="Difficulty", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def plot_boxplots_all_levels(df: pd.DataFrame, group: str, ax: plt.Axes):
    """Box plot pooling all levels: x=model, shows overall distribution."""
    gdf = df[df["group"] == group].copy()
    models_order = [label for label, _ in MODEL_GROUPS[group] if label in gdf["model"].unique()]

    sns.boxplot(
        data=gdf,
        x="model", y="token_len",
        hue="model",
        order=models_order,
        hue_order=models_order,
        palette="Set2",
        width=0.5,
        legend=False,
        flierprops={"marker": ".", "alpha": 0.3, "markersize": 3},
        ax=ax,
    )
    # Overlay mean
    means = gdf.groupby("model")["token_len"].mean()
    for i, m in enumerate(models_order):
        if m in means:
            ax.scatter(i, means[m], color="red", zorder=5, s=40, marker="D")

    ax.set_title(f"Token Length (all levels) — {group} Models", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Token Count")
    ax.grid(axis="y", alpha=0.3)


def plot_cdf(df: pd.DataFrame, group: str, ax: plt.Axes):
    """CDF plot per model, pooling all levels."""
    gdf = df[df["group"] == group].copy()
    models_order = [label for label, _ in MODEL_GROUPS[group] if label in gdf["model"].unique()]
    palette = sns.color_palette("tab10", len(models_order))

    for color, model in zip(palette, models_order):
        vals = np.sort(gdf[gdf["model"] == model]["token_len"].values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=model.replace("\n", " "), color=color, linewidth=1.5)

    ax.set_xlabel("Token Count")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF of Token Length — {group} Models", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_mean_by_level(df: pd.DataFrame, group: str, ax: plt.Axes):
    """Line plot: mean token length per level per model."""
    gdf = df[df["group"] == group].copy()
    models_order = [label for label, _ in MODEL_GROUPS[group] if label in gdf["model"].unique()]
    levels = [l for l in LEVEL_ORDER if l in gdf["level"].unique()]
    palette = sns.color_palette("tab10", len(models_order))

    for color, model in zip(palette, models_order):
        mdf = gdf[gdf["model"] == model]
        means = [mdf[mdf["level"] == lv]["token_len"].mean() for lv in levels]
        ax.plot(levels, means, marker="o", label=model.replace("\n", " "),
                color=color, linewidth=1.5)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Mean Token Count")
    ax.set_title(f"Mean Token Length by Level — {group} Models", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n=== Summary Statistics (token_len) ===")
    summary = (
        df.groupby(["group", "model", "level"])["token_len"]
        .agg(["count", "mean", "median", "std",
              lambda x: np.percentile(x, 25),
              lambda x: np.percentile(x, 75),
              "max"])
        .rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"})
        .round(1)
    )
    print(summary.to_string())

    print("\n=== Overall per Model ===")
    overall = (
        df.groupby(["group", "model"])["token_len"]
        .agg(["count", "mean", "median", "std",
              lambda x: np.percentile(x, 25),
              lambda x: np.percentile(x, 75),
              "max"])
        .rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"})
        .round(1)
    )
    print(overall.to_string())


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("\nLoading predictions ...")
    df = build_dataframe(tokenizer, MODEL_GROUPS)

    if df.empty:
        print("No data loaded. Check paths above.")
        return

    print_summary(df)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    for group in ["3B", "1B"]:
        gdf = df[df["group"] == group]
        if gdf.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Token Length Analysis — {group} Student Models on MATH500", fontsize=15, y=1.01)

        plot_violins_by_model(df, group, axes[0, 0])
        plot_boxplots_all_levels(df, group, axes[0, 1])
        plot_cdf(df, group, axes[1, 0])
        plot_mean_by_level(df, group, axes[1, 1])

        fig.tight_layout()
        out_path = OUTPUT_DIR / f"token_dist_{group}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out_path}")
        plt.close(fig)

    # Combined: all models, box plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Token Length Distribution — All Models (MATH500)", fontsize=14)
    for ax, group in zip(axes, ["3B", "1B"]):
        plot_boxplots_all_levels(df, group, ax)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "token_dist_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
