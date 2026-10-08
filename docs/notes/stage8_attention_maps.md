# Stage 8: Attention Maps

Setup: 4 layers x 4 heads, one 32-character prompt (the opening line of the corpus),
model in eval() mode. Each panel is a grid: row = query (the character asking),
column = key (the character being looked at), bright = high attention weight.

## 1) Why is everything above the diagonal exactly zero?
The causal mask sets future scores to -inf before softmax, and exp(-inf) = 0, so those
weights are exactly 0, not just small. Position i can never attend to a position after i.

## 2) Why call model.eval() before plotting?
Dropout is applied to the attention weights. In training mode it randomly zeroes some
weights and scales the rest by 1/(1-p), which is why my stage 2 rows summed to 1.1111.
In eval mode dropout is off, every row sums to exactly 1, and the map shows only what
the model learned.

## 3) What did I see?
- Layer 0, Head 1: an almost perfect diagonal, so each character attends to itself (a pass-through head).
- Layer 0, Heads 2 and 3: diffuse, spreading weight over the earlier context like a rough average.
- Layers 1 and 2: several heads have a ridge just below the diagonal, attending to the last few characters.
- Layer 1, Head 0: a bright vertical stripe, so many later characters attend to one earlier position. 
- Layer 2, Head 3: vertical stripes that may sit on word starts. 
- Layer 3: sparse and spotty, with a few strong cells instead of clean geometry.
- Different heads in the same layer look different, which is the "multiple perspectives" idea from stage 3 made visible.

## Caveats
- The top-left cell is bright by construction: row 0 can only see itself, so its weight is 1.0.
- Early rows look brighter partly because fewer positions share the weight.
- Models often park attention on the first token as a default, so first-column stripes need care.
- Attention weights show where a head reads from, not proof that this information drove the output.
- One prompt, 4 layers and 3000 training steps is a small sample, so these are observations, not general claims.