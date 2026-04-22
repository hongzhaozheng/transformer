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


print("\n=== Lesson 5: Attention Prep ===")


print("\n=== Section A: Tensor Shape Drills ===")

token_ids = torch.tensor([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
], dtype=torch.long)

embeddings = torch.randn(batch_size, seq_len, d_model)

print("token_ids shape:", tuple(token_ids.shape), "-> [batch_size, seq_len]")
print("embeddings shape:", tuple(embeddings.shape), "-> [batch_size, seq_len, d_model]")

print("\nShape meanings:")
print("[batch_size, seq_len] means one token id per position in each sequence.")
print("[batch_size, seq_len, d_model] means one feature vector per token position.")
print("[batch_size, seq_len, seq_len] means each token attends to every token position.")
print("[batch_size, seq_len, d_model] output means each position gets a new mixed feature vector.")


print("\n=== Section B: Attention Math Warm-Up ===")

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_model)

print("Q shape:", tuple(Q.shape), "-> [batch_size, seq_len, d_k]")
print("K shape:", tuple(K.shape), "-> [batch_size, seq_len, d_k]")
print("V shape:", tuple(V.shape), "-> [batch_size, seq_len, d_model]")

scores = Q @ K.transpose(-2, -1)

print("scores shape:", tuple(scores.shape), "-> [batch_size, seq_len, seq_len]")
print("Example raw score matrix:")
print(scores[0])

scaled_scores = scores / math.sqrt(d_k)

print("scaled_scores shape:", tuple(scaled_scores.shape))

weights = F.softmax(scaled_scores, dim=-1)

print("weights shape:", tuple(weights.shape), "-> [batch_size, seq_len, seq_len]")
print("Example attention weight matrix:")
print(weights[0])
print("Row sums for first example:")
print(weights[0].sum(dim=-1))

output = weights @ V

print("output shape:", tuple(output.shape), "-> [batch_size, seq_len, d_model]")
print("Example output row:")
print(output[0, 0])


print("\n=== Section C: Causal Masking ===")

mask = torch.tril(torch.ones(seq_len, seq_len))
print("Lower-triangular mask:")
print(mask)

future_mask = mask == 0
print("Boolean future mask:")
print(future_mask)

masked_scores = scaled_scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))

print("masked_scores shape:", tuple(masked_scores.shape))
print("Example masked score matrix:")
print(masked_scores[0])

masked_weights = F.softmax(masked_scores, dim=-1)

print("masked_weights shape:", tuple(masked_weights.shape))
print("Example masked attention weights:")
print(masked_weights[0])
print("Masked row sums for first example:")
print(masked_weights[0].sum(dim=-1))

masked_output = masked_weights @ V

print("masked_output shape:", tuple(masked_output.shape), "-> [batch_size, seq_len, d_model]")


print("\n=== Section D: Reflection ===")
print("Attention scores are [batch, seq_len, seq_len] because each position compares against every position.")
print("We scale by sqrt(d_k) to keep dot products from growing too large before softmax.")
print("We need a causal mask so position t cannot use information from future tokens.")


print("\n=== Next Lesson Preview ===")
print("Next we can wrap this math into one self-attention head,")
print("then build toward a minimal decoder-only Transformer.")
