"""
Stage 6b: Training loop.
 
Each step:
  1. get a random batch (x, y) from dataset.py
  2. forward pass -> logits and cross-entropy loss
  3. zero old gradients, backward pass, clip gradients, optimizer step
 
Every eval_interval steps we measure average train and val loss over
eval_iters batches (with dropout off) and save a checkpoint whenever the
validation loss improves.
 
Run from the project root:  python src/train.py
"""

import os
import yaml
import torch 

from tokenizer import CharTokenizer
from dataset import  CharDataset
from model import GPT 


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) 


@torch.no_grad() 
def estimate_loss(model, ds, cfg, device):
    model.eval()
    out={}
    for split in ("train","val"):
        losses=torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y=ds.get_batch(split, cfg["batch_size"], device)
            _, loss=model(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean().item()
    model.train()       #back to training mode
    return out


def main():
    cfg=load_config()
    torch.manual_seed(cfg["seed"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device) 

    with open(cfg["data_path"], "r", encoding="utf-8") as f:
        text=f.read()

    tok=CharTokenizer(text) 
    ds=CharDataset(text, tok, cfg["block_size"])

    model=GPT(
        vocab_size=tok.vocab_size,
        block_size=cfg["block_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    print("Parameters:",sum(p.numel() for p in model.parameters()))

    optimizer=torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    os.makedirs("checkpoints", exist_ok=True)
    best_val=float("inf") 

    for it in range(cfg["max_iters"]+1):
        if it % cfg["eval_interval"]==0:
            losses=estimate_loss(model, ds, cfg, device)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"]<best_val:
                best_val=losses["val"]
                torch.save(
                    {"model_state": model.state_dict(), "config":cfg,
                    "vocab_size":tok.vocab_size},
                    "checkpoints/best.pt"
                )

        if it==cfg["max_iters"]:
            break

        x, y=ds.get_batch("train", cfg["batch_size"], device)
        _, loss=model(x,y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("Done. Best val loss:", round(best_val, 4))


if __name__=="__main__":
    main()