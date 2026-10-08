# Stage 7: Generation

## 1) Why crop the context to the last block_size tokens?
The model can only handle block_size positions: the positional embedding table has exactly
block_size rows, and the causal mask is block_size x block_size. model.py refuses longer
input with an assert. The model has also never trained on longer contexts. Cropping turns
generation into a sliding window, so the model cannot see anything older than block_size
characters. Attention cost also grows quadratically, O(T^2), with sequence length, but the
hard architectural limit is the main reason.

## 2) What does temperature do?
Logits are divided by T before softmax:
    P_i = exp(z_i / T) / sum_j exp(z_j / T)
- T < 1 sharpens the distribution (T near 0 approaches argmax). Low-probability characters
  are crushed, so output is safe, repetitive and predictable.
- T = 1 samples from exactly the distribution the model learned.
- T > 1 flattens it. Unlikely characters get more chance, so output gets more varied and
  eventually chaotic. (Fully uniform is only the limit as T goes to infinity.)
- T must be above 0, because dividing by 0 is undefined (generate.py checks this).

What I saw in my own runs:
- T = 0.3: ( every word is a real English word, but the text loops on a few phrases ("the comes of the world", "of the world of the the"). It is safe and repetitive, and it has no meaning.)

- T = 1.5: (it invents words ("Prestioum", "cohsomes", "meqrad"), capitalizes randomly inside words ("lUCqORD", "PrOZELIZABETHUM") and produces stray symbols ("V&ANK"). The play format still holds, with speaker names, colons and line breaks.)

## 3) Why sample (multinomial) instead of argmax?
Argmax picks the single most likely character every time. The same prompt always gives the
identical output, and text tends to fall into repetitive loops. Multinomial sampling treats
the probabilities as real chances, so output varies between runs and follows the natural
variety of the training text. top_k is the middle ground: it still samples, but only from
the k most likely characters, which cuts off the unlikely tail that produces gibberish.