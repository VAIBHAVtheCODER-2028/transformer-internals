"""
Stage 5: The Transformer block.
 
One block = two sub-layers, each wrapped in a residual connection:
 
    x = x + MultiHeadAttention(LayerNorm(x))   # tokens communicate
    x = x + FeedForward(LayerNorm(x))          # each token "thinks" on its own
 
This is the PRE-LN layout (LayerNorm applied BEFORE each sub-layer), as in
GPT-2. The original paper used POST-LN (LayerNorm after the residual add).
 
Input and output shapes are identical: (B, T, d_model). That is what lets us
stack N of these blocks in stage 6.
 
Note down in docs/notes/stage5_transformer_block.md:
  1) Why does the residual connection use ADDITION (x + sublayer(x))? What
     would break if the block simply returned sublayer(x) and dropped x?
  2) Pre-LN vs post-LN: what is the practical difference, and why do most
     modern models prefer pre-LN?
  3) Why does the feedforward network expand to 4 * d_model in the middle
     and then project back down? Why is it applied per-token with no
     interaction between positions?
"""

import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model, 4*d_model), #expand
            nn.GELU(),                     #non-linearity
            nn.Linear(4*d_model, d_model), #project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, d_model) -> (B, T, d_model). nn.Linear acts on the last
        # dimension only, so every position is processed independently
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float=0.1):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.ffn=FeedForward(d_model, dropout)

    def forward(self, x):
        x=x+self.attn(self.ln1(x)) #residual around attention
        x=x+self.ffn(self.ln2(x))  #residual around feedforward
        return x


if __name__=="__main__":
    torch.manual_seed(0)

    B, T, d_model, n_heads=2, 5, 16, 4
    x=torch.randn(B, T, d_model)

    block=TransformerBlock(d_model=d_model, n_heads=n_heads, block_size=T)
    out=block(x)

    print("Input shape: ", x.shape)     #(2, 5, 16)
    print("Output shape: ", out.shape)  #(2, 5, 16), must match input
    assert out.shape == x.shape

    n_params=sum(p.numel() for p in block.parameters())
    print("Parameters in one block:", n_params)


    #gradient check: confirm gradients flow back to the input through the residual paths
    x.requires_grad = True
    block(x).sum().backward()
    print("Input gradient exists:", x.grad is not None)
