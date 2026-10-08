"""
Stage 1b: Data pipeline — turning raw token ids into (input, target) batches.
 
The core idea of language modeling: given a chunk of tokens, predict the
NEXT token at every position. So if block_size=8 and we have tokens:
    [24, 43, 58, 5, 57, 1, 46, 43, 39]
Then:
    input  = tokens[0:8]  = [24, 43, 58, 5, 57, 1, 46, 43]
    target = tokens[1:9]  = [43, 58, 5, 57, 1, 46, 43, 39]
 
i.e. target is just input shifted right by one. Every position in the
sequence is simultaneously a training example: "given everything up to
here, predict the next token."
 
Note down in docs/notes/stage1_dataset.md: why does training on EVERY
position in the block (not just predicting one token per sequence) make
training so much more sample-efficient?
"""

import torch

class CharDataset:
    def __init__(self, text: str, tokenizer, block_size: int, train_split: float=0.9):
        data = torch .tensor(tokenizer.encode(text), dtype=torch.long)  # converts that list into a PyTorch tensor of 64-bit integers, which is the standard format required for training neural networks in PyTorch.

        # Simple train/val split — no shuffling since it's one continuous text
        n=int(train_split * len(data))
        self.train_data=data[:n]
        self.val_data=data[n:]
        self.block_size=block_size 

    def get_batch(self, split: str, batch_size: int, device: str="cpu"):
        """
        Sample a random batch of (input, target) sequences.

        Returns:
            x: (batch_size, block_size) — input token ids
            y: (batch_size, block_size) — target token ids (x shifted by 1)
        """
        data=self.train_data if split=="train" else self.val_data 

        # Pick batch_size random starting positions
        ix=torch.randint(len(data) - self.block_size, (batch_size,))
        
        x=torch.stack([data[i:i+self.block_size] for i in ix])
        y=torch.stack([data[i+1:i+self.block_size + 1]for i in ix]) 

        return x.to(device), y.to(device) #Just moves the tensors to CPU or GPU depending on what we are training on


if __name__=="__main__":
    from tokenizer import CharTokenizer

    with open("data/input.txt","r",encoding="utf-8") as f:
        text=f.read() 

    tok=CharTokenizer(text)
    ds=CharDataset(text,tok,block_size=8)

    x,y=ds.get_batch("train",batch_size=4)
    print("Input shape:", x.shape) #(4,8)
    print("Target shape:", y.shape) #(4,8)

    # Look at one example unrolled — this makes the "predict next token at every position" idea concrete 
    print("\nOne training example undrolled:")
    for t in range(x.shape[1]):
        context=x[0, :t + 1].tolist() 
        target=y[0, t].item()
        print(f" context={context} -> target={target}")