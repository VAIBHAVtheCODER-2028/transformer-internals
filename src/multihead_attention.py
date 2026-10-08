"""
Stage 3: Multi-head attention.
 
Instead of one attention head computing over the full d_model dimensions,
we run n_heads independent heads in parallel, each operating in a smaller
d_k = d_model // n_heads subspace, each with its OWN learned Q/K/V weights.
 
Each head can specialize in a different kind of relationship (e.g. one
might learn to attend to the previous token, another to a distant
syntactically related word). Their outputs are concatenated back together
and passed through one final linear projection to mix information across
heads.
 
Note down in docs/notes/stage3_multihead.md:
  1) Why must d_model be divisible by n_heads?
  2) What is the purpose of the final output projection (self.proj) after
     concatenating all heads — why not just return the concatenation directly?
"""

import torch
import torch.nn as nn
from attention import SelfAttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size:int, dropout: float=0.1):
        super().__init__()
        assert d_model % n_heads == 0 ,"d_model must be divisible by n_heads"

        d_k=d_model//n_heads 

        # Create n_heads independent SelfAttentionHead instances.
        # nn.ModuleList (not a plain Python list!) so PyTorch tracks their parameters properly for training
        self.heads=nn.ModuleList([
            SelfAttentionHead(d_model=d_model,d_k=d_k, block_size=block_size, dropout=dropout) 
            for _ in range(n_heads)
        ])

        # After concatenating all heads' outputs back to d_model dimensions, this linear layer lets the model mix/combine information across. heads — without it, the heads' outputs would just sit side by side with no interaction.
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # Run every head independently, each returns (B, T, d_k)
        head_outputs=[h(x)[0] for h in self.heads]   # [0] because head returns (out, attn_weights)

        # Concatenate along the last dimension: n_heads * d_k = d_model again
        out = torch.cat(head_outputs, dim=-1)  # (B, T, d_model)

        out=self.proj(out)
        out = self.dropout(out)
        return out 


if __name__=="__main__":
    torch.manual_seed(0)

    B, T, d_model, n_heads=2, 5, 16, 4
    x=torch.randn(B ,T, d_model)

    mha=MultiHeadAttention(d_model=d_model, n_heads=n_heads, block_size=T)
    out=mha(x)

    print("Input Space: ", x.shape) #(2, 5, 16)
    print("Output shape:", out.shape) #(2, 5, 16) — same as input, ready to feed into next block
    print("Number of heads:", len(mha.heads))
    print("d_k per head:", mha.heads[0].d_k)

