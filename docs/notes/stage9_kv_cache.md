# Stage 9: KV Cache

## 1) Why is no causal mask needed inside forward_step?
Each step processes one new position. Its query attends over every cached key: the entire
past plus itself. All of those are legal to see, and future positions do not exist yet, so
there is nothing to hide. In the full forward pass all T positions are processed at once,
so the mask is what stops position i from reading position j > i. In the cached path the
cache itself enforces that.

## 2) Why cache K and V but not Q?
A query is used once, by the position that produced it, to look back over the context.
Keys and values describe the earlier positions, and every future token will need them
again, so they are stored. The new token's own K and V are appended to the cache so later
tokens can attend to it. Without the cache, every step recomputes K and V for the whole
context.

## 3) How much memory does the cache use?
layers x 2 (K and V) x block_size x d_model x 4 bytes (fp32), per sequence.
For this model: 4 x 2 x 128 x 128 x 4 = 524,288 bytes = 0.5 MiB per sequence.
That is tiny, but the cache grows linearly with sequence length and with the number of
sequences served at once. Back-of-envelope for a 7B-class model (32 layers, d_model 4096,
fp16, 4096-token context): about 2 GiB per sequence. Multiply by many concurrent users
and the cache becomes one of the main limits on how many requests a GPU can serve.
PagedAttention targets this by storing the cache in fixed-size blocks, so memory is not
wasted on contiguous buffers sized for the maximum length.

## 4) Why a warm-up run and several runs averaged?
The first call is slower than later ones (memory allocation, library and thread-pool
initialization, cold CPU caches), so it is run