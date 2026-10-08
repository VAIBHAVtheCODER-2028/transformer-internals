Data_pipeline - Tokenizer + data pipeline

Tokenizer:
A Model can not understand human language so when we give a model raw text , the Tokenizer performs the job of converting that raw text into numbers that model can understand and reason over.
why would BPE(BYTE PAIR ENCODING) be necessary for a real production model, and what problem does it solve the char-level doesn't? : 
a) BPE(Byte pair encoding) helps increase the efficiency and reduce the token usage of model as it learns to merge frequent occuring character pairs into larger chunks.So for attention computation shorter token sequences for the same amount of text=dramatically cheaper training and inference.
b) Learning efficiency / generalization: for char level the model has to learn everything from scratch but BPE gives the model pre-chunked , common units/words, so the model does not have to make much effort for it rather it can spend its time learning to estimate better.



Use of tokenizer.py file:
Context 1: During the time of the training
When we are training the model , the tokenizer converts the entire training dataset into list of integers, then the dataset.py chops it into input and target that the model gets trained on.

Context 2: During Inference/Usage
During inference, the user's prompt is encoded into integers the same way, but after the model predicts new integers, decode() converts them back into readable text — this reverse direction (decode) never happens during training.


Dataset:
dataset.py's job is narrower and more mechanical: it takes the full integer sequence and produces random (input, target) pairs — chunks of block_size tokens paired with the same chunk shifted by one position — ready to feed into a model later. It prepares the data; it doesn't predict.

why does training on EVERY position in the block (not just predicting one token per sequence) make training so much more sample-efficient?
Predicting the next output depends on the previous output and prediction as well so thus using every position in the block for training helps to increase the efficiency of prediction.