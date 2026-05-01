"""
Lesson 9: Dropout + Train/Eval Mode + Cleaner Model Class

This lesson builds on Lesson 8.

Goal:
- reuse the tiny stacked decoder-only language model
- add dropout in the Transformer-style places
- understand why model.train() and model.eval() matter
- verify that dropout behaves differently during training and evaluation
- clean up the model API so generation functions do not depend on global variables

Big idea:
Dropout randomly zeroes some activations during training. During evaluation,
dropout is disabled, so predictions become deterministic for the same input.
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
num_heads = 4
hidden_size = 4 * d_model
num_layers = 2
dropout = 0.2
epochs = 200
generate_steps = 40
learning_rate = 1e-2


# Section A: Dataset Refresh
# Rebuild the same character next-token dataset from Lesson 8.
#
# Build:
# - chars
# - vocab_size
# - stoi
# - itos
# - encoded
# - X
# - Y
# - TensorDataset
# - DataLoader
#
# Print:
# - vocab_size
# - number of examples
# - input/target tensor shapes

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encoded = [stoi[ch] for ch in text]
X = []
Y = []
for i in range(len(text) - block_size):
    x = encoded[i : i+block_size]
    y = encoded[i+1 : i+1+block_size]
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype = torch.long)
Y = torch.tensor(Y, dtype = torch.long)

dataset = TensorDataset(X ,Y)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

print(f"vocab_size: {vocab_size}")
print(f"number of examples: {X.shape[0]}")
print(f"Sample input: {X[0]}, sample output: {Y[0]}")

# Section B: Attention With Dropout
# Upgrade SingleHeadSelfAttention so it includes dropout on attention weights.
#
# Implement:
# - class SingleHeadSelfAttention(nn.Module)
# - __init__(self, d_model, head_size, block_size, dropout)
# - forward(self, x)
#
# In __init__:
# - create q, k, v projections
# - register the causal mask
# - create self.dropout = nn.Dropout(dropout)
#
# In forward:
# - compute q, k, v
# - compute scaled attention scores
# - apply the causal mask
# - apply softmax
# - apply dropout to the attention weights
# - return weights @ v
#
# Shape target:
# input:  [B, T, d_model]
# output: [B, T, head_size]



# Section C: Multi-Head Attention With Output Dropout
# Upgrade MultiHeadSelfAttention.
#
# Implement:
# - class MultiHeadSelfAttention(nn.Module)
# - __init__(self, d_model, num_heads, block_size, dropout)
# - forward(self, x)
#
# In __init__:
# - assert d_model can split evenly across num_heads
# - create num_heads attention heads
# - create final projection layer d_model -> d_model
# - create output dropout
#
# In forward:
# - run each head
# - concatenate along the last dimension
# - apply final projection
# - apply output dropout
# - return [B, T, d_model]


# Section D: Feed-Forward With Dropout
# Upgrade FeedForward.
#
# Implement:
# - class FeedForward(nn.Module)
# - __init__(self, d_model, hidden_size, dropout)
# - forward(self, x)
#
# Common pattern:
# d_model -> hidden_size -> ReLU -> d_model -> Dropout
#
# Shape target:
# input:  [B, T, d_model]
# output: [B, T, d_model]


# Section E: Decoder Block With Dropout
# Upgrade DecoderBlock so it passes dropout into attention and feed-forward.
#
# Keep the same pre-norm residual structure:
# x = x + attention(norm1(x))
# x = x + feed_forward(norm2(x))


# Section F: Reusable Stacked Decoder LM
# Create a cleaner model class that stores its own block_size.
#
# Implement:
# - class TinyTransformerLM(nn.Module)
# - __init__(
#       self,
#       vocab_size,
#       block_size,
#       d_model,
#       num_heads,
#       hidden_size,
#       num_layers,
#       dropout,
#   )
# - forward(self, idx)
#
# In __init__:
# - store self.block_size = block_size
# - create token embedding
# - create positional embedding
# - create a dropout after token + position embeddings
# - create num_layers DecoderBlock modules
# - create final LayerNorm
# - create lm head
#
# In forward:
# - optionally check that idx length does not exceed block_size
# - create token embeddings
# - create position ids on the same device as idx
# - add token and position embeddings
# - apply embedding dropout
# - pass through decoder blocks
# - apply final norm
# - apply lm head
# - return logits [B, T, vocab_size]


# Section G: Smoke Test
# Instantiate TinyTransformerLM.
# Pull one batch from the loader.
# Run the model and print:
# - input batch shape
# - logits shape
#
# Expected:
# input batch shape: [batch_size, block_size]
# logits shape:      [batch_size, block_size, vocab_size]


# Section H: Train/Eval Dropout Check
# Verify dropout behavior before training.
#
# Use the same input batch twice in train mode:
# model.train()
# logits1 = model(X)
# logits2 = model(X)
#
# Then use the same input batch twice in eval mode:
# model.eval()
# logits3 = model(X)
# logits4 = model(X)
#
# Print:
# - torch.allclose(logits1, logits2)
# - torch.allclose(logits3, logits4)
#
# Expected idea:
# - train mode should usually be False because dropout is active
# - eval mode should be True because dropout is disabled


# Section I: Training Step
# Train the model for next-token prediction.
#
# Important:
# - call model.train() before the training loop
# - logits shape is [B, T, vocab_size]
# - targets shape is [B, T]
# - reshape logits to [B*T, vocab_size]
# - reshape targets to [B*T]
#
# Print average loss every 20 epochs.


# Section J: Cleaner Greedy Generation
# Write a generation helper that does not depend on a global model.
#
# Implement:
# - def greedy_generate(seed_text, model, stoi, itos, steps):
#
# Use:
# - model.block_size instead of passing block_size separately
# - model.eval()
# - torch.no_grad()
# - last-position logits
# - argmax


# Section K: Cleaner Temperature Sampling
# Write:
# - def sample_text(seed_text, model, stoi, itos, steps, temperature):
#
# Use:
# - model.block_size
# - model.eval()
# - torch.no_grad()
# - last-position logits
# - logits / temperature
# - softmax
# - torch.multinomial
#
# Try:
# - greedy generation
# - temperature = 0.5
# - temperature = 1.0
# - temperature = 1.5


# Reflection
# What changes when model.train() is active?
# What changes when model.eval() is active?
# Why should generation use eval mode?
# Why is embedding dropout useful?
# Why do we store block_size inside the model?
# Why is it cleaner to pass model into generation functions?


# Next Lesson Preview
# Save and load model weights with state_dict, then generate from a reloaded model.
