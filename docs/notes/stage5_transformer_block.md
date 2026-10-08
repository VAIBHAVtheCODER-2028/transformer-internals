## Stage 5: Transformer block

**Residual addition (x + sublayer(x))**
It acts as a gradient highway: the derivative is 1 + f'(x), so the gradient always has a direct path back to earlier layers, which prevents vanishing gradients. It also lets each layer learn an incremental modification (a residual) instead of rebuilding the full representation from scratch.

**Pre-LN vs Post-LN**
Post-LN applies LayerNorm after the residual add. Pre-LN applies it before each sublayer, inside the residual branch, so the identity path stays free of LayerNorm. Post-LN has very large gradients near the output layers at initialization, which makes it unstable unless you use a learning-rate warmup. Pre-LN has well-behaved gradients, so it is much less sensitive to warmup and trains deep models more stably. The tradeoff is that the residual stream's magnitude grows with depth, which is why a final LayerNorm (ln_f) is needed before the output head.

**Feedforward network (4 * d_model expansion, per-token)**
Attention is the only place tokens exchange information, and on its own it is mostly weighted averaging of linear projections. The FFN adds the non-linearity and extra per-token capacity: it expands to a wider space, applies GELU, and projects back down, independently at every position. The 4x width is a convention from the original paper, not something derived. One interpretation from interpretability research is that FFN layers act like a key-value memory, but that is a hypothesis, not settled fact.

output update:
The train/val gap is widening slightly. At step 250 they're nearly identical (2.4503 vs 2.4532). By step 3000, train is 1.5445 but val is 1.7224 — a gap of about 0.18. That's the model starting to memorize training-specific patterns rather than only learning generalizable ones. Not alarming at this scale, but it's the visible signature of overfitting, and it's worth being able to point to this exact curve if an interviewer asks "how do you know if a model is overfitting?" — you have real evidence now, not just the definition.

The loss curve shape itself is healthy — steep drop early (4.18 → 2.45 in the first 250 steps), then a long, steady, decelerating decline. That's exactly what you want to see; a curve that plateaus immediately or oscillates wildly would signal a learning-rate or architecture problem.