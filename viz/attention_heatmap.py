"""
Stage 8: Attention heatmaps.

For a chunk of text, plot what every head in every layer attends to.
Each panel is a T x T grid:
    row    = the query position (the character that is "asking")
    column = the key position   (the character being looked at)
    bright = high attention weight
Because of the causal mask, everything above the diagonal is exactly zero.

This needs MultiHeadAttention to keep its weights: set store_attn = True, run
one forward pass, then read last_attn, shaped (B, n_heads, T, T).

Run from the project root:
    python viz/attention_heatmap.py
    python viz/attention_heatmap.py --prompt "First Citizen:\nBefore we" 

Note down in docs/notes/stage8_attention_maps.md:
  1) Why is everything above the diagonal exactly zero?
  2) Why do we call model.eval() before plotting? (Think back to the row sums
     of 1.1111 you saw in stage 2.)
  3) Which heads look specialized (previous character, same character seen
     earlier, newline or colon)? Does layer 0 look different from layer 3?
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save to files, no window needed
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tokenizer import CharTokenizer
from model import GPT


def pretty(ch):
    """Make whitespace visible on the plot axes."""
    return {"\n": "\\n", " ": "·"}.get(ch, ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--length", type=int, default=32,
                        help="characters taken from the start of the corpus if no --prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--out_dir", type=str, default="results/attention_maps")
    args = parser.parse_args()

    device = "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    with open(cfg["data_path"], "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    assert tok.vocab_size == ckpt["vocab_size"], "tokenizer vocab does not match the checkpoint"

    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=cfg["block_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # dropout off, so every attention row sums to exactly 1

    # Default: a snippet of the training text (guaranteed to be in the vocabulary)
    prompt = args.prompt.replace("\\n", "\n") if args.prompt else text[: args.length]
    prompt = prompt[: cfg["block_size"]]
    try:
        ids = tok.encode(prompt)
    except KeyError as e:
        raise SystemExit(f"Prompt contains a character not in the vocabulary: {e}")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    T = idx.shape[1]

    # Ask every attention module to keep its weights during the forward pass
    for block in model.blocks:
        block.attn.store_attn = True

    with torch.no_grad():
        model(idx)

    chars = [pretty(c) for c in prompt]
    os.makedirs(args.out_dir, exist_ok=True)
    n_heads = cfg["n_heads"]

    for layer, block in enumerate(model.blocks):
        attn = block.attn.last_attn[0]  # (n_heads, T, T) for the single sequence
        fig, axes = plt.subplots(1, n_heads, figsize=(4.2 * n_heads, 4.6))
        if n_heads == 1:
            axes = [axes]

        for h, ax in enumerate(axes):
            ax.imshow(attn[h].numpy(), cmap="magma", vmin=0.0)
            ax.set_xticks(range(T))
            ax.set_xticklabels(chars, fontsize=6)
            ax.set_yticks(range(T))
            ax.set_yticklabels(chars, fontsize=6)
            ax.set_title(f"Layer {layer} - Head {h}", fontsize=10)
            ax.set_xlabel("attended to (key)", fontsize=8)
            if h == 0:
                ax.set_ylabel("query (the character asking)", fontsize=8)

        fig.tight_layout()
        path = os.path.join(args.out_dir, f"layer{layer}_heads.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()