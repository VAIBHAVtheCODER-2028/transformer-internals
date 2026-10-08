"""
Stage 6a: The full GPT model.
 
Pipeline for a batch of token ids idx with shape (B, T):
 
    token embedding      (B, T)    -> (B, T, d_model)   "what is this token?"
  + positional embedding (T,)      -> (T, d_model)      "where is it?"
    N x TransformerBlock (B, T, d_model) -> (B, T, d_model)
    final LayerNorm
    lm_head (Linear)     (B, T, d_model) -> (B, T, vocab_size)   logits
 
The logits at position t are the model's scores for "which character comes
next after tokens 0..t". Cross-entropy against the targets from dataset.py
turns those scores into a single loss number.
 
Note down in docs/notes/stage6_training.md:
  1) Why is the expected loss at initialization about ln(vocab_size)?
     (For 65 characters that is about 4.17.)
  2) Why do we need a final LayerNorm before lm_head in a pre-LN model?
  3) Why call model.eval() when measuring validation loss, and
     model.train() afterwards?
  4) Why is loss.backward() preceded by optimizer.zero_grad()?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

from positional_encoding import LearnedPositionalEmbedding
from transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float=0.1):
        super().__init__()
        self.block_size=block_size

        self.tok_emb=nn.Embedding(vocab_size, d_model)
        self.pos_emb=LearnedPositionalEmbedding(block_size, d_model)
        self.drop=nn.Dropout(dropout) 

        self.blocks=nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f=nn.LayerNorm(d_model)  # final norm (needed in pre-LN)
        self.lm_head=nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-style init: small normal weights keep the initial logits close to uniform , so the starting loss in close to ln(vocab_size).
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  #we can also writr it like nn.init.zeros=module.bias
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T=idx.shape
        assert T <= self.block_size, "sequence longer than block_size"

        x=self.tok_emb(idx) + self.pos_emb(T, idx.device) # (B, T, d_model)
        x=self.drop(x)
        x=self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss=None
        if targets is not None:
            # cross_entropy wants (N, C) logits and (N,) targets, so flatten B and T
            loss=F.cross_entropy(logits.view(B*T, -1),targets.view(B*T))
        return logits, loss



if __name__=="__main__":
    import math 
    torch.manual_seed(0) 

    vocab_size, block_size=65 , 16
    model=GPT(vocab_size, block_size, d_model=32, n_heads=4, n_layers=2)

    idx=torch.randint(0, vocab_size, (4, block_size))
    targets=torch.randint(0, vocab_size, (4, block_size))
    logits, loss=model(idx, targets) 

    print("Logits shape:", logits.shape) # (4, 16, 65)
    print("Initial loss:",round(loss.item(),4))
    print("In(vocab_size):",round(math.log(vocab_size), 4))
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
