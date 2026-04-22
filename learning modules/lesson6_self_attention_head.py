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
#
# Encode the text into integer ids.
#
# Build fixed-window next-token examples:
# - inputs should have shape [num_examples, block_size]
# - targets should have shape [num_examples]
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
        pass

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
        pass


class TinySingleHeadLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, head_size):
        super().__init__()

        # Create:
        # - token embedding table
        # - positional embedding table
        # - one SingleHeadSelfAttention module
        # - a final linear layer that maps from head_size to vocab_size
        pass

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
        pass


# Section C: Smoke Test
# After implementing the classes above:
#
# 1. Instantiate TinySingleHeadLM.
# 2. Pull one mini-batch from the DataLoader.
# 3. Run the model on that batch.
# 4. Verify:
#    - input batch shape is [batch, block_size]
#    - output logits shape is [batch, block_size, vocab_size]


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


# Reflection
# Why do we need positional embeddings if attention can compare tokens?
# Why does a single self-attention head still preserve one output vector per position?
# Why do logits end with vocab_size as the last dimension?


# Next Lesson Preview
# Add a feed-forward network, residual connections,
# and layer normalization to form a full decoder block.
