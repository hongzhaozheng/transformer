"""
Lesson 7: Multi-Head Attention + Decoder Block

This lesson builds on Lesson 6.

Goal:
- rebuild the character next-token dataset
- upgrade one attention head into multi-head attention
- add a feed-forward network
- add residual connections
- add layer normalization
- wrap everything into one decoder block
- use the block inside a tiny decoder-only language model
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

text = "transformers are pattern learners"
block_size = 8
batch_size = 8
d_model = 16
num_heads = 4
head_size = d_model // num_heads
hidden_size = 4 * d_model
epochs = 200
generate_steps = 20
learning_rate = 1e-2


# Section A: Dataset Refresh
# Rebuild the character dataset from Lesson 6.
#
# Build:
# - imports
# - random seed
# - text
# - block_size
# - batch_size
# - d_model
# - num_heads
# - head_size
# - hidden_size
# - epochs
# - generate_steps
# - learning_rate
#
# Build:
# - chars
# - vocab_size
# - stoi
# - itos
#
# Encode the text into integer ids.
#
# Build fixed-window next-token examples:
# - inputs should have shape [num_examples, block_size]
# - targets should have shape [num_examples, block_size]
#
# Convert X and Y to torch.long tensors.
#
# Create:
# - TensorDataset
# - DataLoader
#
# Print:
# - vocab_size
# - number of examples
# - one sample input/target pair

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encoded = [stoi[ch] for ch in text]

X = []
Y = []
for i in range(len(encoded)-block_size):
    x = encoded[i : i+block_size]
    y = encoded[i+1 : i+1+block_size]
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X,Y)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

print(f"vocab_size: {vocab_size}")
print(f"number of examples: {X.shape[0]}")
print(f"Sample input: {X[0]}, sample output: {Y[0]}")


# Section B: Single Attention Head Review
# Implement one causal self-attention head.
#
# This should be very similar to Lesson 6.
#
# Input shape:
# [batch, seq_len, d_model]
#
# Output shape:
# [batch, seq_len, head_size]
#
# Implement:
# - class SingleHeadSelfAttention(nn.Module)
# - __init__(self, d_model, head_size, block_size)
# - forward(self, x)
#
# In __init__:
# - create query projection
# - create key projection
# - create value projection
# - each projection maps from d_model to head_size
# - register a reusable causal mask with shape [block_size, block_size]
#
# In forward:
# - project x into q, k, v
# - compute scaled attention scores with shape [batch, seq_len, seq_len]
# - apply the causal mask using only the current seq_len x seq_len slice
# - apply softmax over the last dimension
# - mix the values with the attention weights
# - return shape [batch, seq_len, head_size]

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.q = nn.Linear(d_model, head_size)
        self.k = nn.Linear(d_model, head_size)
        self.v = nn.Linear(d_model, head_size)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        _, T, head_size = q.shape
        scores = q @ k.transpose(-2,-1) / math.sqrt(head_size)
        masked_scores = torch.masked_fill(scores, self.mask[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(masked_scores, dim = -1)
        return weights @ v


# Section C: Multi-Head Self-Attention
# Build multiple SingleHeadSelfAttention modules in parallel.
#
# Hints:
# - use nn.ModuleList
# - each head returns [batch, seq_len, head_size]
# - concatenate head outputs along the last dimension
# - final shape after concat should be [batch, seq_len, d_model]
# - add a final projection layer from d_model to d_model
#
# Implement:
# - class MultiHeadSelfAttention(nn.Module)
# - __init__(self, d_model, num_heads, block_size)
# - forward(self, x)
#
# In __init__:
# - check that d_model can be evenly split across num_heads
# - compute head_size = d_model // num_heads
# - create num_heads SingleHeadSelfAttention modules
# - create a final projection layer from d_model to d_model
#
# In forward:
# - run each head on x
# - concatenate the head outputs along the last dimension
# - apply the final projection
# - return shape [batch, seq_len, d_model]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        assert d_model % num_heads == 0
        head_size = d_model // num_heads
        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(d_model, head_size, block_size) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        out = torch.cat(heads, dim=-1) # heads is a sequence (Python List) of tensors
        return self.linear(out)

# Section D: Feed-Forward Network
# Build the position-wise MLP used inside a Transformer block.
#
# Input shape:
# [batch, seq_len, d_model]
#
# Output shape:
# [batch, seq_len, d_model]
#
# Hint:
# nn.Linear applies to the last dimension.
#
# Implement:
# - class FeedForward(nn.Module)
# - __init__(self, d_model, hidden_size)
# - forward(self, x)
#
# In __init__:
# - create Linear d_model -> hidden_size
# - apply ReLU
# - create Linear hidden_size -> d_model
#
# In forward:
# - apply the network position-by-position
# - return shape [batch, seq_len, d_model]

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

# Section E: Decoder Block
# Combine:
# - multi-head causal self-attention
# - feed-forward network
# - residual connections
# - layer normalization
#
# Common pre-norm structure:
#
# x = x + self_attention(layer_norm_1(x))
# x = x + feed_forward(layer_norm_2(x))
#
# Return shape:
# [batch, seq_len, d_model]
#
# Implement:
# - class DecoderBlock(nn.Module)
# - __init__(self, d_model, num_heads, block_size, hidden_size)
# - forward(self, x)
#
# In __init__:
# - create first LayerNorm over d_model
# - create MultiHeadSelfAttention
# - create second LayerNorm over d_model
# - create FeedForward
#
# In forward:
# - apply attention with a residual connection
# - apply feed-forward with a residual connection
# - return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, hidden_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mhsa = MultiHeadSelfAttention(
            d_model = d_model, 
            num_heads = num_heads, 
            block_size = block_size
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(
            d_model = d_model, 
            hidden_size = hidden_size
        )
        

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# Section F: Tiny Decoder-Only Language Model
# Create a tiny language model using:
# - token embeddings
# - positional embeddings
# - one DecoderBlock
# - final layer norm
# - final linear layer to vocab_size
#
# Input:
# idx shape [batch, seq_len]
#
# Output:
# logits shape [batch, seq_len, vocab_size]
#
# Implement:
# - class TinyDecoderLM(nn.Module)
# - __init__(self, vocab_size, block_size, d_model, num_heads, hidden_size)
# - forward(self, idx)
#
# In __init__:
# - create token embedding table
# - create positional embedding table
# - create one DecoderBlock
# - create final LayerNorm over d_model
# - create lm head from d_model to vocab_size
#
# In forward:
# - build token embeddings
# - build position ids for current seq_len
# - add token and position embeddings
# - run through decoder block
# - apply final layer norm
# - project to vocab logits
# - return logits

class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, hidden_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.decoder = DecoderBlock(
            d_model = d_model, 
            num_heads = num_heads, 
            block_size = block_size, 
            hidden_size = hidden_size
        )
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        token_embedding = self.token_embedding(idx)
        T = idx.shape[-1]
        pos_id = torch.arange(T, device = idx.device)
        pos_embedding = self.pos_embedding(pos_id)
        embedding = token_embedding + pos_embedding
        decoded = self.decoder(embedding)
        normalized_decoded = self.norm(decoded)
        logits = self.linear(normalized_decoded)
        return logits

# Section G: Smoke Test
# After implementing the classes above:
#
# 1. Instantiate TinyDecoderLM.
# 2. Pull one batch from the DataLoader.
# 3. Run the model on that batch.
# 4. Verify:
#    - input batch shape is [batch, block_size]
#    - output logits shape is [batch, block_size, vocab_size]

model = TinyDecoderLM(
    vocab_size = vocab_size, 
    block_size = block_size, 
    d_model = d_model, 
    num_heads = num_heads, 
    hidden_size = hidden_size
)
X, Y = next(iter(loader))
logits = model(X)
print(f"input batch shape: {X.shape}")
print(f"output logits shape: {logits.shape}")

# Section H: Training Step
# Train the tiny decoder model for next-token prediction.
#
# Hints:
# - logits shape: [B, T, vocab_size]
# - targets shape: [B, T]
# - reshape logits to [B*T, vocab_size]
# - reshape targets to [B*T]
#
# Track:
# - average loss per epoch
#
# Print progress every so often.

opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    for step, (X,Y) in enumerate(loader):
        logits = model(X)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            Y.reshape(-1)
        )

        total_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, average loss: {avg_loss}")


# Section I: Greedy Generation
# Write a helper that:
# - starts from a seed string
# - keeps the most recent block_size token ids
# - runs the model
# - takes the last position's logits
# - chooses the next token id
# - appends it and repeats
#
# Return the decoded generated string.

def TextGenerator(seed_text, model, stoi, itos, block_size, steps):
    generated = seed_text
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            context = generated[-block_size:]
            context_id = torch.tensor([[stoi[ch] for ch in context]], dtype = torch.long)
            logits = model(context_id)
            pred = torch.argmax(logits[:,-1,:], dim = -1)
            next_char = itos[pred.item()]
            generated += next_char

    return generated

seed_text = text[:block_size]
print(TextGenerator(seed_text, model, stoi, itos, block_size, generate_steps))

# Reflection
# Why do multi-head attention outputs get concatenated on the last dimension?
# Why do residual connections require matching shapes?
# Why does LayerNorm use d_model as its normalized shape?
# Why does the feed-forward network not mix information across token positions?


# Next Lesson Preview
# Stack multiple decoder blocks and add sampling with temperature.
