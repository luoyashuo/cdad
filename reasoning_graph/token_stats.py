import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool
import os

DATASET_PATH = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rv_optimal_60k/train.jsonl"
TOKENIZER_PATH = "/fs/fast/u20240075/luoyashuo/modelscope_cache/models/LLM-Research/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "/fs/fast/u20240075/hf_home/datasets/Yaseal/CDAD_EXP/omnithought_rv_optimal_60k"
NUM_WORKERS = 32
CHUNK_SIZE = 500  # lines per worker chunk

# ── Worker ────────────────────────────────────────────────────────────────────

_tokenizer = None

def init_worker(tokenizer_path):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def process_chunk(lines):
    """Return (total_lengths, loss_lengths) for a list of raw JSON lines."""
    full_texts = []
    prefix_texts = []   # flattened list of prefix texts across all samples
    full_up_texts = []  # flattened list of full-up-to texts across all samples
    n_pairs_per_sample = []

    for line in lines:
        conversations = json.loads(line)["conversations"]

        full_text = _tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        full_texts.append(full_text)

        n_pairs = 0
        for i, msg in enumerate(conversations):
            if msg["role"] == "assistant":
                prefix_texts.append(_tokenizer.apply_chat_template(
                    conversations[:i], tokenize=False, add_generation_prompt=True
                ))
                # For last assistant turn in single-turn chats this equals full_text,
                # but we keep it general for multi-turn.
                full_up_texts.append(_tokenizer.apply_chat_template(
                    conversations[:i+1], tokenize=False, add_generation_prompt=False
                ))
                n_pairs += 1
        n_pairs_per_sample.append(n_pairs)

    # Batch encode all texts in one call — much faster than individual encode()
    all_texts = full_texts + prefix_texts + full_up_texts
    encoded = _tokenizer(all_texts, add_special_tokens=False, return_attention_mask=False)
    all_lengths = [len(ids) for ids in encoded["input_ids"]]

    n = len(full_texts)
    np_ = len(prefix_texts)
    total_lens = all_lengths[:n]
    prefix_lens = all_lengths[n : n + np_]
    full_up_lens = all_lengths[n + np_ :]

    loss_lens = []
    pair_idx = 0
    for k in n_pairs_per_sample:
        loss = sum(
            full_up_lens[pair_idx + j] - prefix_lens[pair_idx + j]
            for j in range(k)
        )
        loss_lens.append(loss)
        pair_idx += k

    return total_lens, loss_lens


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading dataset...")
    with open(DATASET_PATH) as f:
        lines = f.readlines()

    # Split into chunks
    chunks = [lines[i : i + CHUNK_SIZE] for i in range(0, len(lines), CHUNK_SIZE)]
    print(f"Processing {len(lines):,} samples with {NUM_WORKERS} workers ({len(chunks)} chunks)...")

    total_lengths = []
    loss_lengths = []

    with Pool(
        processes=NUM_WORKERS,
        initializer=init_worker,
        initargs=(TOKENIZER_PATH,),
    ) as pool:
        for t_lens, l_lens in tqdm(
            pool.imap(process_chunk, chunks, chunksize=1),
            total=len(chunks),
            desc="chunks",
        ):
            total_lengths.extend(t_lens)
            loss_lengths.extend(l_lens)

    total_lengths = np.array(total_lengths)
    loss_lengths = np.array(loss_lengths)

    # ── Stats ─────────────────────────────────────────────────────────────────
    def print_and_write_stats(name, arr, fh=None):
        lines_out = [
            f"=== {name} ===",
            f"  Count  : {len(arr)}",
            f"  Min    : {arr.min()}",
            f"  Max    : {arr.max()}",
            f"  Mean   : {arr.mean():.1f}",
            f"  Median : {np.median(arr):.1f}",
            f"  Std    : {arr.std():.1f}",
        ]
        for p in [50, 75, 90, 95, 99]:
            lines_out.append(f"  p{p:<3}   : {np.percentile(arr, p):.0f}")
        text = "\n".join(lines_out) + "\n"
        print(text)
        if fh:
            fh.write(text + "\n")

    stats_path = f"{OUTPUT_DIR}/token_stats.txt"
    with open(stats_path, "w") as fh:
        print_and_write_stats("Total Token Lengths", total_lengths, fh)
        print_and_write_stats("Loss Token Lengths", loss_lengths, fh)
    print(f"Stats saved to {stats_path}")

    # ── Histogram plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Token Length Distribution — omnithought_rv_optimal_60k", fontsize=14)

    def plot_hist(ax, arr, title, color, log_y=False):
        ax.hist(arr, bins=100, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(arr.mean(), color="red", ls="--", lw=1.2, label=f"mean={arr.mean():.0f}")
        ax.axvline(np.median(arr), color="orange", ls="--", lw=1.2, label=f"median={np.median(arr):.0f}")
        ax.axvline(np.percentile(arr, 99), color="purple", ls=":", lw=1.2, label=f"p99={np.percentile(arr,99):.0f}")
        ax.set_title(title)
        ax.set_xlabel("Token count")
        ax.set_ylabel("Frequency (log)" if log_y else "Frequency")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plot_hist(axes[0, 0], total_lengths, "Total Tokens — Linear Y", "#4C72B0")
    plot_hist(axes[0, 1], total_lengths, "Total Tokens — Log Y",    "#4C72B0", log_y=True)
    plot_hist(axes[1, 0], loss_lengths,  "Loss Tokens — Linear Y",  "#DD8452")
    plot_hist(axes[1, 1], loss_lengths,  "Loss Tokens — Log Y",     "#DD8452", log_y=True)

    plt.tight_layout()
    plot_path = f"{OUTPUT_DIR}/token_length_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Histogram saved to {plot_path}")

    # ── CDF plots ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("CDF of Token Lengths — omnithought_rv_optimal_60k", fontsize=14)

    def plot_cdf(ax, arr, title, color):
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, color=color, lw=1.5)
        for p in [50, 75, 90, 95, 99]:
            v = np.percentile(arr, p)
            ax.axvline(v, ls=":", lw=0.8, alpha=0.6)
            ax.text(v, p / 100, f"p{p}={v:.0f}", fontsize=7, rotation=90, va="bottom", ha="right")
        ax.set_xlabel("Token count")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plot_cdf(axes2[0], total_lengths, "Total Tokens CDF", "#4C72B0")
    plot_cdf(axes2[1], loss_lengths,  "Loss Tokens CDF",  "#DD8452")

    plt.tight_layout()
    cdf_path = f"{OUTPUT_DIR}/token_length_cdf.png"
    plt.savefig(cdf_path, dpi=150, bbox_inches="tight")
    print(f"CDF plot saved to {cdf_path}")
