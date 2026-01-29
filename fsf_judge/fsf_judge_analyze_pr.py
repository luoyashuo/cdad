#!/usr/bin/env python3
"""
FSF Judge PR Curve Analysis Script

This script analyzes the inference results from FSF Judge and computes
Precision-Recall curves at different thresholds.

Usage:
    python scripts/fsf_judge_analyze_pr.py \
        --input_path /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val/inference_results.jsonl \
        --output_dir /data-share/yeesuanAI08/lihaitao/luoys/checkpoints/cdad/fsf_judge/qwen3_8b_fsf_judge/checkpoint-12000/fsf_judge_val

Output:
    - pr_curve_data.csv: CSV file with precision, recall, and threshold values
    - pr_curve.png: PR curve plot
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="FSF Judge PR Curve Analysis")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the inference results jsonl file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for PR curve data and plots. If None, uses the directory of input_path",
    )
    parser.add_argument(
        "--num_thresholds",
        type=int,
        default=100,
        help="Number of threshold points to compute",
    )
    parser.add_argument(
        "--positive_label",
        type=str,
        default="Y",
        help="Label considered as positive class (default: Y for contains failed branches)",
    )
    return parser.parse_args()


def load_results(input_path: str) -> List[Dict[str, Any]]:
    """Load inference results from jsonl file."""
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_pr_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    positive_label: str = "Y",
) -> Tuple[float, float, float, float]:
    """
    Compute precision, recall, TP, FP at a given threshold.
    
    Args:
        y_true: Ground truth labels (1 for positive, 0 for negative)
        y_scores: Predicted probabilities for positive class
        threshold: Decision threshold
        positive_label: The label considered as positive
        
    Returns:
        (precision, recall, f1, accuracy)
    """
    # Predict positive if score >= threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Compute confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return precision, recall, f1, accuracy, tp, fp, fn, tn


def compute_pr_curve(
    results: List[Dict[str, Any]],
    num_thresholds: int = 100,
    positive_label: str = "Y",
) -> List[Dict[str, Any]]:
    """
    Compute PR curve data at different thresholds.
    
    Returns:
        List of dictionaries containing threshold, precision, recall, etc.
    """
    # Extract ground truth labels and predicted probabilities
    y_true_list = []
    y_scores_list = []
    skipped_anomaly = 0
    
    for r in results:
        target_label = r.get("target_label")
        failed_branch_prob = r.get("failed_branch_prob")
        is_anomaly = r.get("is_anomaly", False)
        
        # Skip anomaly samples (where predicted token is not Y or N)
        if is_anomaly:
            skipped_anomaly += 1
            continue
        
        if target_label is None or failed_branch_prob is None:
            continue
        
        # Ground truth: 1 if positive_label, 0 otherwise
        y_true_list.append(1 if target_label == positive_label else 0)
        # Score is the probability of positive class (failed_branch_prob = P(Y))
        y_scores_list.append(failed_branch_prob)
    
    y_true = np.array(y_true_list)
    y_scores = np.array(y_scores_list)
    
    if len(y_true) == 0:
        print("Warning: No valid samples found for PR curve computation")
        return []
    
    print(f"Computing PR curve for {len(y_true)} valid samples")
    if skipped_anomaly > 0:
        print(f"  - Skipped {skipped_anomaly} anomaly samples (predicted token is not Y or N)")
    print(f"  - Positive class ('{positive_label}'): {np.sum(y_true)} samples")
    print(f"  - Negative class: {len(y_true) - np.sum(y_true)} samples")
    
    # Generate thresholds
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    
    pr_data = []
    for threshold in thresholds:
        precision, recall, f1, accuracy, tp, fp, fn, tn = compute_pr_at_threshold(
            y_true, y_scores, threshold, positive_label
        )
        
        pr_data.append({
            "threshold": round(float(threshold), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
            "accuracy": round(float(accuracy), 6),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })
    
    return pr_data


def find_precision_recall_thresholds(
    pr_data: List[Dict[str, Any]],
    target_precisions: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    Find thresholds that achieve specific precision levels.
    
    Returns:
        List of dictionaries with precision level, recall, and threshold.
    """
    if target_precisions is None:
        target_precisions = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    
    precision_recall_thresholds = []
    
    for target_prec in target_precisions:
        # Find all points where precision >= target
        candidates = [p for p in pr_data if p["precision"] >= target_prec]
        
        if candidates:
            # Among points with precision >= target, find the one with highest recall
            best = max(candidates, key=lambda x: x["recall"])
            precision_recall_thresholds.append({
                "target_precision": target_prec,
                "achieved_precision": best["precision"],
                "recall": best["recall"],
                "threshold": best["threshold"],
                "f1_score": best["f1_score"],
            })
        else:
            # No point achieves this precision
            precision_recall_thresholds.append({
                "target_precision": target_prec,
                "achieved_precision": None,
                "recall": None,
                "threshold": None,
                "f1_score": None,
            })
    
    return precision_recall_thresholds


def save_pr_data(pr_data: List[Dict[str, Any]], output_path: str):
    """Save PR curve data to CSV file."""
    import csv
    
    if not pr_data:
        print("Warning: No PR data to save")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pr_data[0].keys())
        writer.writeheader()
        writer.writerows(pr_data)
    
    print(f"PR curve data saved to: {output_path}")


def save_precision_recall_thresholds(prt_data: List[Dict[str, Any]], output_path: str):
    """Save precision-recall-threshold data to CSV file."""
    import csv
    
    if not prt_data:
        print("Warning: No precision-recall-threshold data to save")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prt_data[0].keys())
        writer.writeheader()
        writer.writerows(prt_data)
    
    print(f"Precision-Recall-Threshold data saved to: {output_path}")


def plot_pr_curve(pr_data: List[Dict[str, Any]], output_path: str, positive_label: str = "Y"):
    """Plot PR curve and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping plot generation.")
        print("  To install: pip install matplotlib")
        return
    
    if not pr_data:
        print("Warning: No PR data to plot")
        return
    
    # Extract data
    thresholds = [p["threshold"] for p in pr_data]
    precisions = [p["precision"] for p in pr_data]
    recalls = [p["recall"] for p in pr_data]
    f1_scores = [p["f1_score"] for p in pr_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. PR Curve
    ax1 = axes[0, 0]
    ax1.plot(recalls, precisions, "b-", linewidth=2)
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title(f"Precision-Recall Curve (Positive Class: '{positive_label}')", fontsize=14)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True, alpha=0.3)
    
    # Add baseline (random classifier)
    # Baseline precision = proportion of positive class
    baseline_precision = sum(precisions) / len(precisions) if precisions else 0.5
    ax1.axhline(y=baseline_precision, color="r", linestyle="--", alpha=0.5, label=f"Baseline (~{baseline_precision:.3f})")
    ax1.legend()
    
    # 2. Precision vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(thresholds, precisions, "g-", linewidth=2, label="Precision")
    ax2.plot(thresholds, recalls, "m-", linewidth=2, label="Recall")
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Precision and Recall vs Threshold", fontsize=14)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 Score vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(thresholds, f1_scores, "r-", linewidth=2)
    ax3.set_xlabel("Threshold", fontsize=12)
    ax3.set_ylabel("F1 Score", fontsize=12)
    ax3.set_title("F1 Score vs Threshold", fontsize=14)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.0])
    ax3.grid(True, alpha=0.3)
    
    # Mark maximum F1 point
    max_f1_idx = np.argmax(f1_scores)
    max_f1_threshold = thresholds[max_f1_idx]
    max_f1_value = f1_scores[max_f1_idx]
    ax3.axvline(x=max_f1_threshold, color="k", linestyle="--", alpha=0.5)
    ax3.scatter([max_f1_threshold], [max_f1_value], color="red", s=100, zorder=5)
    ax3.annotate(
        f"Max F1: {max_f1_value:.3f}\n@ threshold={max_f1_threshold:.3f}",
        xy=(max_f1_threshold, max_f1_value),
        xytext=(max_f1_threshold + 0.1, max_f1_value - 0.1),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    
    # 4. Distribution of predicted probabilities
    ax4 = axes[1, 1]
    
    # Extract scores from pr_data if available, otherwise skip
    # We'll need to load the original results for this
    ax4.text(0.5, 0.5, "Distribution plot requires\noriginal data file", 
             ha="center", va="center", transform=ax4.transAxes, fontsize=12)
    ax4.set_title("Score Distribution (Placeholder)", fontsize=14)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"PR curve plot saved to: {output_path}")
    plt.close()


def plot_score_distribution(results: List[Dict[str, Any]], output_path: str, positive_label: str = "Y"):
    """Plot distribution of predicted probabilities for each class."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping distribution plot.")
        return
    
    # Extract scores by class
    positive_scores = []
    negative_scores = []
    skipped_anomaly = 0
    
    for r in results:
        target_label = r.get("target_label")
        failed_branch_prob = r.get("failed_branch_prob")
        is_anomaly = r.get("is_anomaly", False)
        
        # Skip anomaly samples
        if is_anomaly:
            skipped_anomaly += 1
            continue
        
        if target_label is None or failed_branch_prob is None:
            continue
        
        if target_label == positive_label:
            positive_scores.append(failed_branch_prob)
        else:
            negative_scores.append(failed_branch_prob)
    
    if not positive_scores and not negative_scores:
        print("Warning: No valid scores for distribution plot")
        return
    
    if skipped_anomaly > 0:
        print(f"  - Skipped {skipped_anomaly} anomaly samples in distribution plot")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 51)
    
    if negative_scores:
        ax.hist(negative_scores, bins=bins, alpha=0.6, label=f"Negative (n={len(negative_scores)})", color="blue")
    if positive_scores:
        ax.hist(positive_scores, bins=bins, alpha=0.6, label=f"Positive '{positive_label}' (n={len(positive_scores)})", color="red")
    
    ax.set_xlabel("Predicted Probability (Failed Branch)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Predicted Probabilities by Class", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Score distribution plot saved to: {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_path)
    
    print(f"Input file: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"\nLoading inference results...")
    results = load_results(args.input_path)
    print(f"Loaded {len(results)} samples")
    
    # Compute PR curve
    print(f"\nComputing PR curve...")
    pr_data = compute_pr_curve(results, args.num_thresholds, args.positive_label)
    
    if not pr_data:
        print("Error: Could not compute PR curve")
        sys.exit(1)
    
    # Save full PR curve data
    pr_curve_path = os.path.join(args.output_dir, "pr_curve_data.csv")
    save_pr_data(pr_data, pr_curve_path)
    
    # Find precision-recall-threshold data
    print(f"\nComputing precision-recall-threshold data...")
    prt_data = find_precision_recall_thresholds(pr_data)
    
    # Save precision-recall-threshold data
    prt_path = os.path.join(args.output_dir, "precision_recall_threshold.csv")
    save_precision_recall_thresholds(prt_data, prt_path)
    
    # Print summary table
    print("\n" + "="*70)
    print("Precision-Recall-Threshold Summary")
    print("="*70)
    print(f"{'Target Prec':<15} {'Achieved Prec':<15} {'Recall':<12} {'Threshold':<12}")
    print("-"*70)
    for p in prt_data:
        if p["achieved_precision"] is not None:
            print(f"{p['target_precision']:<15.2f} {p['achieved_precision']:<15.4f} {p['recall']:<12.4f} {p['threshold']:<12.4f}")
        else:
            print(f"{p['target_precision']:<15.2f} {'N/A':<15} {'N/A':<12} {'N/A':<12}")
    print("="*70)
    
    # Plot PR curve
    print(f"\nGenerating plots...")
    plot_path = os.path.join(args.output_dir, "pr_curve.png")
    plot_pr_curve(pr_data, plot_path, args.positive_label)
    
    # Plot score distribution
    dist_plot_path = os.path.join(args.output_dir, "score_distribution.png")
    plot_score_distribution(results, dist_plot_path, args.positive_label)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
