"""
Stage 4: Positional encoding.
 
Attention has no built-in sense of order — Q·K scores only depend on
CONTENT (the values in each token's embedding), not position. If you
shuffled the tokens in a sentence, self-attention would compute the exact
same set of pairwise scores, just permuted. So we must explicitly inject
position information before attention ever sees the embeddings.
 
Why not just add the raw position number (0, 1, 2, ...) to the embedding?
Embeddings are trained to live in a small, bounded range (~ -1 to 1).
Adding a raw large number like 500 would completely dominate the content
signal and would never generalize to sequence lengths not seen in training.
 
This file implements TWO approaches:
  1) LearnedPositionalEmbedding — a simple nn.Embedding lookup table,
     one learned vector per position. This is what we'll actually use in
     model.py — simple, effective, and how GPT-2 style models do it.
  2) sinusoidal_positional_encoding — the fixed, non-learned sin/cos scheme
     from the original "Attention is All You Need" paper. Included as a
     reference implementation so you understand the classic approach too,
     even though we won't wire it into the main model.
 
Note down in docs/notes/stage4_positional_encoding.md:
  1) What's the practical tradeoff between LEARNED positional embeddings
     and FIXED sinusoidal ones? (Hint: think about what happens if you feed
     the model a sequence LONGER than anything it saw during training.)
  2) Positional encoding is added to the token embedding, not concatenated.
     Why does addition work here instead of needing concatenation?
"""

import math
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        # One learned d_model-dimensional vector per position, 0..block_size-1
        self.pos_embedding=nn.Embedding(block_size, d_model)

    def forward(self, T: int, device):
        # Returns shape (T, d_model) — one positional vector per position in the current sequence length T(T<=block_size) 
        positions = torch.arange(T, device=device) # [0, 1, 2, ..., T-1]
        return self.pos_embedding(positions) 

def sinusoidal_positional_encoding(block_size: int, d_model: int):
    """
    Reference implementation of the fixed sin/cos scheme.
    Returns a (block_size, d_model) tensor, precomputed  once (no learning).
    """
    pe=torch.zeros(block_size, d_model)
    position=torch.arange(0, block_size, dtype=torch.float32).unsqueeze(1) #(block_size, 1) 

    # The frequency term: 10000^(2i/d_model), computed in log-space for numerical stability
    div_term=torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)*(-math.log(10000.0)/d_model)
    )

    pe[:, 0::2] = torch.sin(position*div_term) #even dimensions
    pe[: ,1::2] = torch.cos(position*div_term) #odd dimesntions 
    return pe 


if __name__=="__main__":
    block_size, d_model = 8, 16 

    # ---Learned version ---
    learned = LearnedPositionalEmbedding(block_size ,d_model)
    pos_vecs = learned(T=5, device="cpu") 
    print("Learned positional embedding shape:", pos_vecs.shape) #(5,16) 

    # ---Sinusoidal version ---
    sin_pe=sinusoidal_positional_encoding(block_size, d_model)
    print("Sinusoidal positional encoding shape:", sin_pe.shape) #(8,16) 
    print("\sinusoidal PE for position 0 (first 8 val uses):")
    print(sin_pe[0, :8])
    print("\nSinusoidal PE for position 4 (first 8 val uses):")
    print(sin_pe[4, :8])
    print("\nAll values should be between -1 and 1 --check min/max:" )
    print("min:", sin_pe.min().item(), "max:", sin_pe.max().item())