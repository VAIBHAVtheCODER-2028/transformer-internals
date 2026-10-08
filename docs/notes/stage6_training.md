## Stage 6: Training

**1. Why is the initial loss about ln(vocab_size)?**
An untrained model outputs near-zero logits (the init uses std 0.02), so softmax gives almost equal probability 1/V to every token. Cross-entropy is -ln(1/V) = ln(V). For 65 characters that is about 4.17, and my run started at 4.19. If the initial loss is far above this, something is wrong with the init or the code before training even starts.

**2. Why a final LayerNorm before lm_head?**
In a pre-LN model, each block adds its output to the residual stream without normalizing it, so the stream's magnitude grows with depth. The final LayerNorm rescales it (with a learnable scale and shift) before the linear head, so the logits come from stable activations. It is standard practice rather than strictly required.

**3. Why model.eval() for validation and model.train() afterwards?**
They switch layers that behave differently in training and evaluation. In this model that is only Dropout, which randomly zeroes activations while training and must be off when measuring loss. (BatchNorm would also change behavior, but this model uses LayerNorm, which is identical in both modes.) Forgetting eval() makes the validation loss noisier and higher. torch.no_grad() is a separate switch: it stops gradient tracking but does not turn dropout off.

**4. Why optimizer.zero_grad() before loss.backward()?**
PyTorch accumulates gradients across backward() calls instead of overwriting them. Without clearing, the gradients from earlier batches would add into the current one and produce a wrong update. Accumulation is useful on purpose (gradient accumulation simulates a bigger batch). My loop uses set_to_none=True, which frees the gradient tensors instead of filling them with zeros, so it is slightly faster and uses less memory.


output Update:
Baseline: 109.9 tokens/sec on CPU, which is the number the KV cache has to beat. One thing to keep in mind for later: our positional embeddings are absolute and the context is cropped to 128 tokens, so the KV-cache benchmark should compare generation up to 128 total tokens, where the cache applies cleanly. Past that, the cache would need a sliding-window design. We'll handle that when we get there.