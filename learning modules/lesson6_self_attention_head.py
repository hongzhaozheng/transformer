"""
Lesson 6: Token + Positional Embeddings + One Self-Attention Head

This lesson continues from lesson 5 and starts the
"Next Build After This Notebook" project.

Goal:
- rebuild a tiny character dataset
- add token embeddings
- add positional embeddings
- wrap attention math into one causal self-attention head
- use that head inside a tiny decoder-style language model
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(0)

text = "transformers are pattern learners"
block_size = 8
batch_size = 8
d_model = 16
head_size = 16
epochs = 200
generate_steps = 20
learning_rate = 1e-2


# Section A: Character Dataset Refresh
# Reuse the workflow from lessons 2-4.
#
# Build:
# - chars
# - vocab_size
# - stoi
# - itos

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# Encode the text into integer ids.
encoded = [stoi[ch] for ch in text]

# Build fixed-window next-token examples:
# - inputs should have shape [num_examples, block_size]
# - targets should have shape [num_examples, block_size]
#
# Convert them to torch.long tensors.
#
# Create:
# - dataset
# - loader
#
# Print:
# - vocab_size
# - number of examples
# - one sample input/target pair

X = []
Y = []

for i in range(len(encoded) - block_size):
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


# Section B: Token + Positional Embeddings
# Create a small language-model class that starts with:
# - a token embedding table
# - a positional embedding table
#
# Hint:
# token ids have shape [batch, seq_len]
# token embeddings should become [batch, seq_len, d_model]
# position embeddings should become [seq_len, d_model]
# after broadcasting and addition, x should be [batch, seq_len, d_model]


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()

        # Create linear projections for:
        # - queries
        # - keys
        # - values
        #
        # Each projection should map from d_model to head_size.
        #
        # Also create or register a reusable causal mask tied to block_size.
        self.q = nn.Linear(d_model, head_size)
        self.k = nn.Linear(d_model, head_size)
        self.v = nn.Linear(d_model, head_size)
        self.register_buffer("mask", torch.tril(torch.ones(block_size,block_size)))

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        #
        # Project x into q, k, v.
        #
        # Compute attention scores so each position compares against every position.
        # The score tensor should end up with shape [batch, seq_len, seq_len].
        #
        # Scale the scores before softmax.
        #
        # Apply the causal mask so future positions cannot be attended to.
        # Be careful to use only the seq_len x seq_len slice needed for the current batch.
        #
        # Apply softmax over the dimension that represents "which tokens can I attend to?"
        #
        # Use the resulting attention weights to mix the value vectors.
        #
        # Return shape:
        # [batch, seq_len, head_size]
        B, T, d_model = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        head_size = q.shape[-1]
        scores = q @ k.transpose(-2,-1) / math.sqrt(head_size)
        scores = scores.masked_fill(self.mask[:T,:T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim = -1)
        result = weights @ v
        return result



class TinySingleHeadLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, head_size):
        super().__init__()

        # Create:
        # - token embedding table
        # - positional embedding table
        # - one SingleHeadSelfAttention module
        # - a final linear layer that maps from head_size to vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.shsa = SingleHeadSelfAttention(
            d_model = d_model, 
            head_size = head_size, 
            block_size = block_size
        )
        self.linear = nn.Linear(head_size, vocab_size)

    def forward(self, idx):
        # idx shape: [batch, seq_len]
        #
        # Build token embeddings from idx.
        #
        # Build position ids for the current sequence length.
        # Then look up positional embeddings.
        #
        # Add token embeddings and positional embeddings.
        #
        # Run the result through the self-attention head.
        #
        # Project to logits over the vocabulary.
        #
        # Return logits shape:
        # [batch, seq_len, vocab_size]

        B, T = idx.shape
        token_embedding = self.token_embedding(idx)
        pos_id = torch.arange(T, device = idx.device)
        pos_embedding = self.pos_embedding(pos_id)
        embedding = token_embedding + pos_embedding
        logits = self.shsa(embedding)
        result = self.linear(logits)
        return result


# Section C: Smoke Test
# After implementing the classes above:
#
# 1. Instantiate TinySingleHeadLM.
# 2. Pull one mini-batch from the DataLoader.
# 3. Run the model on that batch.
# 4. Verify:
#    - input batch shape is [batch, block_size]
#    - output logits shape is [batch, block_size, vocab_size]

model = TinySingleHeadLM(
    vocab_size = vocab_size, 
    block_size = block_size, 
    d_model = d_model, 
    head_size = head_size
)

X, Y = next(iter(loader))
logits = model(X)
print(f"input batch shape: {X.shape}")
print(f"output logits shape: {logits.shape}")

# Section D: Training Step
# Add a simple training loop for next-token prediction.
#
# Hints:
# - the model returns one logit vector per position
# - the targets should line up position-by-position
# - you may need to reshape logits and targets before calling cross-entropy
#
# Track:
# - average loss
#
# Print progress every so often.

model = TinySingleHeadLM(
    vocab_size = vocab_size, 
    block_size = block_size, 
    d_model = d_model, 
    head_size = head_size
)

opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
# loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0

    for step, (X,Y) in enumerate(loader):
        logits = model(X)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            Y.reshape(-1)
        )
        total_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, step {step}, Loss: {loss.item():.4f}")
    
    average_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")

# Section E: Greedy Generation
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
            context_id = [stoi[ch] for ch in context]
            X = torch.tensor([context_id], dtype = torch.long)
            logits = model(X)
            pred = logits[:,-1,:].argmax(dim = -1)
            next_char = itos[pred.item()]
            generated += next_char
    return generated

seed_text = text[:block_size]
print(TextGenerator(seed_text, model, stoi, itos, block_size, generate_steps))

# Reflection
# Why do we need positional embeddings if attention can compare tokens?
# Why does a single self-attention head still preserve one output vector per position?
# Why do logits end with vocab_size as the last dimension?


# Next Lesson Preview
# Add a feed-forward network, residual connections,
# and layer normalization to form a full decoder block.
