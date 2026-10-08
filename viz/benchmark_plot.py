"""
Stage 10: Plot the KV-cache benchmark.

Reads results/benchmarks.json (written by src/kv_cache.py) and saves a bar chart
to results/kv_cache_benchmark.png. Error bars are the standard deviation over runs.

Run from the project root, after python src/kv_cache.py:
    python viz/benchmark_plot.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="results/benchmarks.json")
    parser.add_argument("--out", type=str, default="results/kv_cache_benchmark.png")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        r = json.load(f)

    labels = ["Naive\n(recompute everything)", "KV cache"]
    means = [r["naive_tokens_per_sec"]["mean"], r["cached_tokens_per_sec"]["mean"]]
    stds = [r["naive_tokens_per_sec"]["std"], r["cached_tokens_per_sec"]["std"]]

    fig, ax = plt.subplots(figsize=(6, 4.8))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, width=0.55,
                  color=["#9aa0a6", "#1a73e8"])

    top = max(m + s for m, s in zip(means, stds))
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + top * 0.02,
                f"{m:.0f} tok/s", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("tokens / second (higher is better)")
    ax.set_title(f"Generation speed: KV cache is {r['speedup']:.2f}x faster", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.012,
        f"{r['device'].upper()} | {r['n_layers']} layers, d_model={r['d_model']} | "
        f"{r['new_tokens']} new tokens | mean +/- std of {r['runs']} runs",
        ha="center", fontsize=8, color="gray",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()