"""
Stage 7: Text generation (autoregressive sampling).

The loop, repeated max_new_tokens times:
    1. take the text so far (as token ids), cropped to the last block_size ids
    2. run the model, keep only the logits at the LAST position
       (those are the scores for "what comes next")
    3. divide by temperature, optionally keep only the top_k scores
    4. softmax -> probabilities, then SAMPLE one id from them
    5. append that id and repeat

This is the NAIVE version: every step re-runs the whole model on the whole
context, recomputing keys and values for tokens it has already seen. That
waste is exactly what the KV cache removes in kv_cache.py. The tokens/sec
number printed at the end is your baseline for that benchmark.

Run from the project root:
    python src/generate.py
    python src/generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 40

Note down in docs/notes/stage7_generation.md:
  1) Why do we crop the context to the last block_size tokens?
  2) What does temperature do to the probabilities? What do you see at
     0.3 vs 1.0 vs 1.5?
  3) Why sample from the distribution (multinomial) instead of always
     taking the most likely character (argmax)?
"""

import argparse
import time

import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from model import GPT


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """idx: (B, T) tensor of token ids. Returns (B, T + max_new_tokens)."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]            # crop to the context window
        logits, _ = model(idx_cond)                      # (B, T, vocab_size)
        logits = logits[:, -1, :] / temperature          # last position only: (B, vocab_size)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")  # drop everything below the k-th best

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise SystemExit("--temperature must be greater than 0")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    # Rebuild the tokenizer from the same text; the vocab is deterministic (sorted chars)
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

    try:
        prompt_ids = tok.encode(args.prompt)
    except KeyError as e:
        raise SystemExit(f"Prompt contains a character not in the vocabulary: {e}")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start = time.time()
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k)
    elapsed = time.time() - start

    print(tok.decode(out[0].tolist()))
    print(f"\n[{args.max_new_tokens} tokens in {elapsed:.2f}s = "
          f"{args.max_new_tokens / elapsed:.1f} tokens/sec on {device}]")


if __name__ == "__main__":
    main()