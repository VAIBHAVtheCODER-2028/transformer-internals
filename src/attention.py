"""
Stage 2: Self-attention, from scratch (single head first).
 
The core question attention answers, for every token: "given everything
I've seen so far, which other tokens should I pay attention to, and how
much, in order to build a better representation of myself?"
 
Three learned projections per token:
  - Query (Q): "what am I looking for?"
  - Key   (K): "what do I contain, for others to match against?"
  - Value (V): "what do I actually offer, if someone attends to me?"
 
Note down in docs/notes/stage2_attention.md:
  1) Why do we scale the dot product by 1/sqrt(d_k)? What breaks without it?
  2) Why is the causal mask applied BEFORE softmax, not after?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, block_size: int,dropout: float=0.1):
        super().__init__() 

        # Linear layers that project the input embedding into Q, K, V.
        # bias=False is standard here — these are pure projections
        self.query=nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)

        # The causal mask: a lower-triangular matrix of 1s.
        # register_buffer means it's part of the model's state but NOT a learnable parameter — it never gets updated by gradients.

        self.register_buffer(
            "tril",torch.tril(torch.ones(block_size,block_size))
        )

        self.dropout=nn.Dropout(dropout) 
        self.d_k=d_k 

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        B,T,_=x.shape

        q = self.query(x)  # (B, T, d_k)
        k = self.key(x)    # (B, T, d_k)
        v = self.value(x)  # (B, T, d_k)

        # Attention scores: how much does each token's query match every other token's key? (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
        scores=q @ k.transpose(-2,-1)

        # Scale by sqrt(d_k). Without this, dot products grow large as d_k grows, pushing softmax into regions with tiny gradients.
        scores = scores / (self.d_k ** 0.5)


        # Causal mask: position i can only attend to positions <= i
        # We set future positions to -inf BEFORE softmax, so softmax turns them into exactly 0 probability(not just "small")
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Turn scores into a probability distribution over positions.
        attn_weights=F.softmax(scores,dim=-1) # (B, T, T)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values, weighted by attention probabilities.
        # (B, T, T) @ (B, T, d_k) -> (B, T, d_k)
        out=attn_weights @ v 

        return out, attn_weights # returning weights too, for visualization later

    
if __name__ == "__main__":
    # Sanity check with tiny fake data before trusting this in the full model
    torch.manual_seed(0) 

    B, T, d_model, d_k=2,5,16,8  
    x=torch.randn(B, T, d_model)

    head=SelfAttentionHead(d_model=d_model, d_k=d_k, block_size=T)
    out,attn_weight=head(x)

    print("Output shape:", out.shape) # (2, 5, 8)
    print("Attention weights shape:", attn_weight.shape)  # (2, 5, 5)

    print("\nAttention weights for batch 0 (rows=query pos, cols=key pos):")
    print(attn_weight[0])

    # Check: row 0 should have weight only in column 0 (can only see itself)
    # Row 4 should have weights spread across columns 0-4 (can see everyone up to and including itself), and each row should sum to 1.0
    print("\nRow sums (should all be 1.0):", attn_weight[0].sum(dim=-1))

