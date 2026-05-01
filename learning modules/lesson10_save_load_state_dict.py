"""
Lesson 10: Save + Load GPT-Style Decoder Weights

This lesson builds on Lesson 9.

Goal:
- reuse the tiny GPT-style decoder-only language model
- train it for character next-token prediction
- save model weights with state_dict
- save the small model config needed to rebuild the architecture
- load weights into a fresh model
- verify the original and reloaded model produce the same output in eval mode
- generate text from the reloaded model

Big idea:
A PyTorch state_dict stores learned parameters, not the Python class itself.
To load weights, you first rebuild the same model architecture, then call
load_state_dict.
"""

import math
import os

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
checkpoint_path = "checkpoints/lesson10_tiny_gpt_decoder.pt"


# Section A: Dataset Refresh
# Rebuild the same character next-token dataset.

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoded = [stoi[ch] for ch in text]

X = []
Y = []
for i in range(len(encoded) - block_size):
    X.append(encoded[i : i + block_size])
    Y.append(encoded[i + 1 : i + 1 + block_size])

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"vocab_size: {vocab_size}")
print(f"number of examples: {X.shape[0]}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


# Section B: GPT-Style Decoder Components
# Same model family as Lesson 9.

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.q = nn.Linear(d_model, head_size)
        self.k = nn.Linear(d_model, head_size)
        self.v = nn.Linear(d_model, head_size)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        _, T, head_size = q.shape
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadSelfAttention(d_model, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size, block_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, block_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, hidden_size, num_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, hidden_size, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        T = idx.shape[-1]
        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        token_embedding = self.token_embedding(idx)
        pos_id = torch.arange(T, device=idx.device)
        pos_embedding = self.pos_embedding(pos_id)

        x = token_embedding + pos_embedding
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)


# Section C: Helper Functions
# Keep training, loss, and generation reusable.

def build_model():
    return TinyTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def train_model(model, loader):
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, Y_batch in loader:
            logits = model(X_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                Y_batch.reshape(-1),
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, average loss: {avg_loss:.4f}")

    return losses


def greedy_generate(seed_text, model, stoi, itos, steps):
    generated = seed_text
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(steps):
            context = generated[-model.block_size :]
            context_id = [stoi[ch] for ch in context]
            X_context = torch.tensor([context_id], dtype=torch.long, device=device)

            logits = model(X_context)
            next_id = logits[:, -1, :].argmax(dim=-1).item()
            generated += itos[next_id]

    return generated


# Section D: Train A Model
# Train one model normally.

model = build_model()
losses = train_model(model, loader)

seed_text = text[:block_size]
print("Original model generation:")
print(greedy_generate(seed_text, model, stoi, itos, generate_steps))


# Section E: Save state_dict + Config
# A state_dict contains tensors for learned weights and buffers.
# The config lets us rebuild the same architecture before loading.

model_config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "d_model": d_model,
    "num_heads": num_heads,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "dropout": dropout,
}

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": model_config,
    "stoi": stoi,
    "itos": itos,
    "losses": losses,
}

os.makedirs("checkpoints", exist_ok=True)
torch.save(checkpoint, checkpoint_path)
print(f"Saved checkpoint to: {checkpoint_path}")


# Section F: Load Into A Fresh Model
# Rebuild the model with the saved config, then load the saved weights.

loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
loaded_config = loaded_checkpoint["model_config"]

reloaded_model = TinyTransformerLM(**loaded_config)
reloaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
reloaded_model.eval()


# Section G: Verify The Reloaded Model
# In eval mode, dropout is off, so the same weights and input should match.

model.eval()
with torch.no_grad():
    original_logits = model(X)
    reloaded_logits = reloaded_model(X)

print(f"Reloaded logits match original logits: {torch.allclose(original_logits, reloaded_logits)}")

print("Reloaded model generation:")
print(greedy_generate(seed_text, reloaded_model, stoi, itos, generate_steps))


# Reflection
# Why does loading require the same architecture?
# What is stored in model.state_dict()?
# What is not stored in model.state_dict()?
# Why should we call model.eval() before comparing loaded model outputs?
# Why is it useful to save model_config next to model_state_dict?


# Next Lesson Preview
# Build a validation split and track train loss vs validation loss.
