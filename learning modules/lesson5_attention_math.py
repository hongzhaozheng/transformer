"""
Lesson 5: Tensor Shapes + Attention Math + Causal Masking

This lesson continues from lesson 4 and follows Sections 8-10
of the Transformer prep notebook.
"""

import math

import torch
import torch.nn.functional as F


torch.manual_seed(0)

batch_size = 2
seq_len = 4
d_model = 6
d_k = 3


# Section A: Tensor Shape Drills
# Print and verify:
# - token_ids shape -> [batch_size, seq_len]
# - embeddings shape -> [batch_size, seq_len, d_model]
#
# Shape meanings:
# [batch_size, seq_len] means one token id per position in each sequence.
# [batch_size, seq_len, d_model] means one feature vector per token position.
# [batch_size, seq_len, seq_len] means each token attends to every token position.
# [batch_size, seq_len, d_model] output means each position gets a new mixed feature vector.
token_ids = torch.tensor([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
], dtype=torch.long)

embeddings = torch.randn(batch_size, seq_len, d_model)


# Section B: Attention Math Warm-Up
# Create:
# - Q with shape [batch_size, seq_len, d_k]
# - K with shape [batch_size, seq_len, d_k]
# - V with shape [batch_size, seq_len, d_model]
#
# Print and verify:
# - Q shape -> [batch_size, seq_len, d_k]
# - K shape -> [batch_size, seq_len, d_k]
# - V shape -> [batch_size, seq_len, d_model]
#
# Compute attention scores by comparing each query with every key.
# Think about which dimensions need to line up for batched matrix multiplication.
#
# Verify scores shape:
# [batch_size, seq_len, seq_len]
#
# Scale the scores using sqrt(d_k).
#
# Apply softmax over the last dimension to get attention weights.
# Verify weights shape:
# [batch_size, seq_len, seq_len]
#
# Check that each row sums to 1.
#
# Use the attention weights to combine information from V.
# The final output should keep one feature vector per token position.
#
# Verify output shape:
# [batch_size, seq_len, d_model]
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_model)
print(f"Q shape -> ", tuple(Q.shape))
print(f"K shape -> ", tuple(K.shape))
print(f"V shape -> ", tuple(V.shape))

scores = Q @ K.transpose(-2,-1)
print(f"scores shape -> ", tuple(scores.shape))

scaled_scores = scores / math.sqrt(d_k)

weights = F.softmax(scaled_scores, dim=-1)
print(f"weights shape -> ", tuple(weights.shape))
print(weights.sum(-1)[0])

output = weights @ V
print(f"output shape -> ", tuple(output.shape))

# Section C: Causal Masking
# Create a lower-triangular mask with shape [seq_len, seq_len].
# This allows each token to attend only to itself and earlier positions.
#
# Convert it into a boolean future mask where future positions are True.
#
# Apply the mask to scaled_scores using masked_fill.
# Fill masked positions with negative infinity so softmax gives them zero weight.
#
# Hint:
# masked_scores should still have shape [batch_size, seq_len, seq_len]
#
# Apply softmax again to get masked_weights.
# Verify that each row still sums to 1.
#
# Use the masked attention weights to combine V again.
#
# Verify masked_output shape:
# [batch_size, seq_len, d_model]
mask = torch.tril(torch.ones(seq_len,seq_len))

future_mask = mask == 0
masked_scores = scaled_scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))
masked_weights = F.softmax(masked_scores, dim=-1)
print(masked_weights.sum(-1)[0])

masked_output = masked_weights @ V
print("masked_output shape -> ", tuple(masked_output.shape))


# Section D: Reflection
# Why are attention scores [batch, seq_len, seq_len]?
# Why do we divide by sqrt(d_k)?
# Why does a causal mask prevent looking into the future?


# Next Lesson Preview
# Wrap this math into one self-attention head.
# Then build toward a minimal decoder-only Transformer.
