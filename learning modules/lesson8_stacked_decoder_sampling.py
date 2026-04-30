"""
Lesson 8: Stacked Decoder Blocks + Temperature Sampling

This lesson builds on Lesson 7.

Goal:
- rebuild the character next-token dataset
- reuse the decoder-block components from Lesson 7
- stack multiple decoder blocks
- train a deeper tiny decoder-only language model
- compare greedy generation with temperature sampling
- understand how temperature changes output randomness
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
head_size = d_model // num_heads
hidden_size = 4 * d_model
num_layers = 2
epochs = 200
generate_steps = 40
learning_rate = 1e-2


# Section A: Dataset Refresh
# Rebuild the same character next-token dataset from Lesson 7.
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
# Shapes:
# - X should be [num_examples, block_size]
# - Y should be [num_examples, block_size]
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
for i in range(len(encoded) - block_size):
    x = encoded[i : i+block_size]
    y = encoded[i+1 : i+1+block_size]
    X.append(x)
    Y.append(y)
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X,Y)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"vocab_size: {vocab_size}")
print(f"number of examples: {X.shape[0]}")
print(f"Sample input: {X[0]}, sample target: {Y[0]}")


# Section B: Rebuild Lesson 7 Components
# Recreate the classes from Lesson 7:
#
# - SingleHeadSelfAttention
# - MultiHeadSelfAttention
# - FeedForward
# - DecoderBlock
#
# Important reminders:
# - SingleHeadSelfAttention input: [batch, seq_len, d_model]
# - SingleHeadSelfAttention output: [batch, seq_len, head_size]
# - use scaled dot-product attention:
#   scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
# - slice the causal mask to the current sequence length:
#   mask[:T, :T]
# - MultiHeadSelfAttention output should be [batch, seq_len, d_model]
# - FeedForward output should be [batch, seq_len, d_model]
# - DecoderBlock should use two separate LayerNorm modules
# - DecoderBlock should use pre-norm residual structure:
#   x = x + attention(layer_norm_1(x))
#   x = x + feed_forward(layer_norm_2(x))


# Section C: Stacked Decoder-Only Language Model
# Upgrade TinyDecoderLM so it uses multiple DecoderBlock modules.
#
# Implement:
# - class StackedDecoderLM(nn.Module)
# - __init__(
#       self,
#       vocab_size,
#       block_size,
#       d_model,
#       num_heads,
#       hidden_size,
#       num_layers,
#   )
# - forward(self, idx)
#
# In __init__:
# - create token embedding table
# - create positional embedding table
# - create num_layers DecoderBlock modules using nn.ModuleList
# - create final LayerNorm over d_model
# - create lm head from d_model to vocab_size
#
# In forward:
# - idx shape is [batch, seq_len]
# - build token embeddings
# - build position ids on the same device as idx
# - add token and position embeddings
# - pass x through each decoder block in order
# - apply final layer norm
# - apply lm head
# - return logits with shape [batch, seq_len, vocab_size]
#
# Shape target:
# idx:    [B, T]
# logits: [B, T, vocab_size]


# Section D: Smoke Test
# After implementing StackedDecoderLM:
#
# 1. Instantiate the model.
# 2. Pull one batch from the DataLoader.
# 3. Run the model on the batch.
# 4. Print:
#    - input batch shape
#    - output logits shape
#
# Expected:
# - input batch shape: [batch_size, block_size]
# - output logits shape: [batch_size, block_size, vocab_size]


# Section E: Training Step
# Train the stacked decoder model for next-token prediction.
#
# Hints:
# - logits shape: [B, T, vocab_size]
# - targets shape: [B, T]
# - reshape logits to [B*T, vocab_size]
# - reshape targets to [B*T]
# - use F.cross_entropy or nn.CrossEntropyLoss
#
# Track:
# - total loss per epoch
# - average loss per epoch
#
# Print progress every so often.
#
# Optional:
# - compare the loss curve with the one-block model from Lesson 7
# - notice whether the deeper model overfits this tiny dataset faster


# Section F: Greedy Generation Review
# Recreate greedy generation from Lesson 7.
#
# Greedy generation:
# - start with seed_text
# - keep only the latest block_size characters
# - encode context to shape [1, T]
# - run the model
# - take logits at the last position
# - choose argmax token id
# - append the decoded character
#
# Return the generated string.


# Section G: Temperature Sampling
# Write a sampling generator that uses temperature.
#
# Temperature idea:
# - lower temperature makes predictions sharper and more conservative
# - higher temperature makes predictions flatter and more random
#
# Implement:
# - def sample_text(seed_text, model, stoi, itos, block_size, steps, temperature):
#
# Inside the loop:
# - get last-position logits:
#   logits = logits[:, -1, :]
# - divide logits by temperature:
#   logits = logits / temperature
# - convert to probabilities:
#   probs = F.softmax(logits, dim=-1)
# - sample one token id:
#   next_id = torch.multinomial(probs, num_samples=1)
# - append the decoded character
#
# Try temperatures:
# - 0.5
# - 1.0
# - 1.5
#
# Watch how the output changes.


# Section H: Compare Generation Methods
# Print generated text from:
# - greedy generation
# - temperature = 0.5
# - temperature = 1.0
# - temperature = 1.5
#
# Use the same seed text for all methods.
#
# Questions:
# - Which output is most repetitive?
# - Which output is most random?
# - Which one looks most like the training text?


# Reflection
# Why does stacking decoder blocks keep the shape [batch, seq_len, d_model]?
# Why do we need nn.ModuleList for multiple decoder blocks?
# What happens when temperature is less than 1?
# What happens when temperature is greater than 1?
# Why does torch.multinomial use probabilities instead of logits?


# Next Lesson Preview
# Add dropout, train/eval behavior checks, and a cleaner reusable model class.
