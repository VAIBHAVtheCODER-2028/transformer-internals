"""
Stage 9: KV cache.

THE PROBLEM
In generate.py, every new character re-runs the whole model on the whole context
(up to block_size characters). But for characters we have already processed, the
keys and values at every layer are identical to last time: they depend only on
those characters and the ones before them (the causal mask guarantees this).
We recompute the same numbers over and over.

THE FIX
Keep each layer's keys and values, for every head, in a cache. For a new character,
compute only ITS query, key and value, append the key and value to the cache, and
attend over the cached keys. One position of work per step instead of T.

What we cache is K and V, not Q. A query is used once, by the position that made
it. Keys and values are read again at every later step.

No causal mask is needed in forward_step: the cache holds only the past plus the
current position, so there is no future to hide.

LIMIT
Our positional embeddings are absolute and the table has block_size rows, so cached
generation works for prompt_len + new_tokens <= block_size. Sliding past that would
shift every position, which makes the cached keys and values stale. (This is part of
why many modern models use relative or rotary position schemes.)

Run from the project root:
    python src/kv_cache.py

Note down in docs/notes/stage9_kv_cache.md:
  1) Why is no causal mask needed inside forward_step?
  2) Why do we cache K and V but not Q?
  3) How much memory does the cache use for this model at full length?
     (layers x 2 x block_size x d_model x 4 bytes, per sequence.) Why does that
     number matter for serving many users at once?
  4) Why does the benchmark need a warm-up run and several runs averaged?
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from model import GPT
from generate import generate as generate_naive


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
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
    model.eval()
    return model, tok, cfg


def new_cache(model):
    """One K and one V slot per layer per head. They start empty (None)."""
    n_layers = len(model.blocks)
    n_heads = len(model.blocks[0].attn.heads)
    k_cache = [[None] * n_heads for _ in range(n_layers)]
    v_cache = [[None] * n_heads for _ in range(n_layers)]
    return k_cache, v_cache


@torch.no_grad()
def forward_step(model, idx_new, pos, k_cache, v_cache):
    """
    Process ONE new token per sequence, reusing and extending the cache.

    idx_new: (B, 1) token ids      pos: int, the position index of this token
    Returns logits for the next token: (B, vocab_size)
    (Dropout is skipped because the model is in eval mode, where it is a no-op.)
    """
    pos_ids = torch.tensor([pos], device=idx_new.device)
    x = model.tok_emb(idx_new) + model.pos_emb.pos_embedding(pos_ids)  # (B, 1, d_model)

    for l, block in enumerate(model.blocks):
        h = block.ln1(x)
        head_outs = []
        for i, head in enumerate(block.attn.heads):
            q = head.query(h)   # (B, 1, d_k)
            k = head.key(h)     # (B, 1, d_k)
            v = head.value(h)   # (B, 1, d_k)

            # Append this position's key and value to the cache
            k_cache[l][i] = k if k_cache[l][i] is None else torch.cat([k_cache[l][i], k], dim=1)
            v_cache[l][i] = v if v_cache[l][i] is None else torch.cat([v_cache[l][i], v], dim=1)

            # The new query attends over every cached key: (B,1,d_k) @ (B,d_k,t) -> (B,1,t)
            scores = q @ k_cache[l][i].transpose(-2, -1) / (head.d_k ** 0.5)
            w = F.softmax(scores, dim=-1)
            head_outs.append(w @ v_cache[l][i])  # (B, 1, d_k)

        x = x + block.attn.proj(torch.cat(head_outs, dim=-1))
        x = x + block.ffn(block.ln2(x))

    x = model.ln_f(x)
    return model.lm_head(x)[:, -1, :]  # (B, vocab_size)


def sample_next(logits, temperature, top_k):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """Same job as generate.generate, but one position of work per new character."""
    model.eval()
    B, P = idx.shape
    if P + max_new_tokens > model.block_size:
        raise ValueError(
            f"cached generation needs prompt_len + new_tokens <= block_size "
            f"({P} + {max_new_tokens} > {model.block_size})"
        )

    k_cache, v_cache = new_cache(model)

    # Prefill: feed the prompt one token at a time to fill the cache.
    # (Real systems do this as one batched pass; one at a time is simpler.)
    logits = None
    for p in range(P):
        logits = forward_step(model, idx[:, p : p + 1], p, k_cache, v_cache)

    for i in range(max_new_tokens):
        next_id = sample_next(logits, temperature, top_k)
        idx = torch.cat([idx, next_id], dim=1)
        if i < max_new_tokens - 1:  # the last token needs no further forward pass
            logits = forward_step(model, next_id, idx.shape[1] - 1, k_cache, v_cache)
    return idx


@torch.no_grad()
def check_equivalence(model, length=24):
    """
    Teacher-forced check: feed a random sequence through the full model once, then
    through the cached step-by-step path, and compare the logits at every position.
    """
    model.eval()
    device = next(model.parameters()).device
    length = min(length, model.block_size)
    vocab = model.lm_head.out_features
    idx = torch.randint(0, vocab, (1, length), device=device)

    full_logits, _ = model(idx)  # (1, L, vocab)
    k_cache, v_cache = new_cache(model)

    max_diff = 0.0
    for p in range(length):
        step_logits = forward_step(model, idx[:, p : p + 1], p, k_cache, v_cache)
        diff = (step_logits - full_logits[:, p, :]).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/benchmarks.json")
    args = parser.parse_args()

    device = "cpu"  # benchmark on CPU so timings are comparable from run to run
    model, tok, cfg = load_model(args.checkpoint, device)

    # 1) Correctness first: a fast cache that changes the output is useless
    diff = check_equivalence(model)
    print(f"Max logit difference, cached vs full forward: {diff:.2e}")
    assert diff < 1e-4, "the cache is NOT equivalent to the full forward pass"
    print("Equivalence check passed\n")

    # 2) Benchmark
    prompt = args.prompt.replace("\\n", "\n")
    try:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    except KeyError as e:
        raise SystemExit(f"Prompt contains a character not in the vocabulary: {e}")
    new_tokens = cfg["block_size"] - ids.shape[1]

    def run_naive():
        generate_naive(model, ids, new_tokens, temperature=1.0)

    def run_cached():
        generate_cached(model, ids, new_tokens, temperature=1.0)

    run_naive()   # warm-up: first calls are slower (allocations, lazy init)
    run_cached()

    naive_tps, cached_tps = [], []
    for r in range(args.runs):  # alternate, so both see the same background load
        naive_tps.append(new_tokens / timed(run_naive))
        cached_tps.append(new_tokens / timed(run_cached))
        print(f"run {r + 1}: naive {naive_tps[-1]:7.1f} tok/s | cached {cached_tps[-1]:7.1f} tok/s")

    def stats(xs):
        return {
            "mean": statistics.mean(xs),
            "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        }

    results = {
        "device": device,
        "block_size": cfg["block_size"],
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "prompt_len": ids.shape[1],
        "new_tokens": new_tokens,
        "runs": args.runs,
        "naive_tokens_per_sec": stats(naive_tps),
        "cached_tokens_per_sec": stats(cached_tps),
    }
    results["speedup"] = (
        results["cached_tokens_per_sec"]["mean"] / results["naive_tokens_per_sec"]["mean"]
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nnaive : {results['naive_tokens_per_sec']['mean']:.1f} +/- {results['naive_tokens_per_sec']['std']:.1f} tok/s")
    print(f"cached: {results['cached_tokens_per_sec']['mean']:.1f} +/- {results['cached_tokens_per_sec']['std']:.1f} tok/s")
    print(f"speedup: {results['speedup']:.2f}x   (saved to {args.out})")


if __name__ == "__main__":
    main()