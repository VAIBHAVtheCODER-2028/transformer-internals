"""
Stage 1a:Character-level tokenizer.

Why charcater-level and not word/subword (BPE) or wordpiece?
-Simplicity: vocab is just very unique character in text (~65-100 tokens)
-No tokenizer training needed - you can focus purely on the Transformer itself
-Tradeoff: sequences get longer (1 char = 1 token), and the model has to learn spelling from scratch, but that's fine for a from-scratch learning project.

Note down in docs/notes/stage1_tokenizer.md: why would BPE(BYTE PAIR ENCODING) be necessary for a real production model, and what problem does it solve the char-level doesn't?
"""

class CharTokenizer:
    def __init__(self,text: str):
        # Build vocabulary: sorted unique characters in the corpus
        chars=sorted(list(set(text)))
        self.vocab_size=len(chars)

        # Two lookup tables: char -> integer id, and id -> char
        self.stoi={ch: i for i, ch in enumerate(chars)}
        self.itos={i: ch for i, ch in enumerate(chars)}

    def encode(self, s:str) -> list[int]:
        """String -> list of integer token ids."""
        return [self.stoi[c] for c in s] 

    def decode(self,ids: list[int]) -> str:
        """List of integer token ids -> string."""
        return ''.join(self.itos[i] for i in ids)


if __name__=="__main__":
    # Quick sanity check — run this file directly to verify encode/decode
    # round-trips correctly before moving on to dataset.py
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text=f.read()

    tok=CharTokenizer(text)
    print(f"Vocab size: {tok.vocab_size}")

    sample = text[:50]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)

    print(f"original: {sample!r}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded!r}")
    assert decoded == sample, "Round-trip failed!"
    print("Round-trip OK")