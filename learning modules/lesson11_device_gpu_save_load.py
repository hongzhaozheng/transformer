"""
Lesson 11: Device Handling + GPU Training + Cross-Device Loading

This lesson builds on Lesson 10.

Goal:
- choose GPU when it is available, otherwise use CPU
- move the model and mini-batches to the selected device
- keep generation device-safe
- save a checkpoint after training
- load the checkpoint with map_location so it works across devices

Big idea:
PyTorch tensors and models must live on the same device before they interact.
If your model is on GPU but your input batch is still on CPU, the forward pass
will fail. A good training loop moves each batch to the same device as the
model, and a good loading pattern uses map_location to avoid device surprises.
"""

import math
from pathlib import Path

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
epochs = 120
generate_steps = 40
learning_rate = 1e-2
checkpoint_path = Path("tiny_transformer_lm_device_checkpoint.pt")


# Section A: Pick A Device
# Use CUDA when it is available.
# Otherwise, fall back to CPU.
#
# Print the selected device so you know where training will run.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


# Section B: Dataset Refresh
# Build the same character next-token dataset.
#
# Important:
# Keep the full dataset tensors on CPU. The DataLoader will yield CPU batches,
# and the training loop will move each batch to the selected device.

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoded = [stoi[ch] for ch in text]

X = []
Y = []
for i in range(len(encoded) - block_size):
    x = encoded[i : i + block_size]
    y = encoded[i + 1 : i + 1 + block_size]
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"vocabulary: {''.join(chars)!r}")
print(f"vocab_size: {vocab_size}")
print(f"number of examples: {X.shape[0]}")
print(f"X device before training: {X.device}")


# Section C: Rebuild The Tiny Transformer
# This is the same model shape as Lesson 10.
#
# Notice:
# - the causal mask is registered as a buffer
# - buffers move with model.to(device)

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
        mask = self.mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        head_size = d_model // num_heads
        self.heads = nn.ModuleList(
            [
                SingleHeadSelfAttention(d_model, head_size, block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


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
    def __init__(self, d_model, num_heads, block_size, hidden_size, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, block_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, hidden_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model,
        num_heads,
        hidden_size,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, block_size, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_ids = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos_ids)

        x = self.dropout(token_emb + pos_emb)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)


# Section D: Move The Model To The Device
# model.to(device) moves all parameters and registered buffers.

config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "d_model": d_model,
    "num_heads": num_heads,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "dropout": dropout,
}

model = TinyTransformerLM(**config).to(device)
first_param = next(model.parameters())
print(f"model parameter device: {first_param.device}")


# Section E: Device-Safe Training Loop
# Move each batch to device inside the loop.
#
# Pattern:
# X_batch = X_batch.to(device)
# Y_batch = Y_batch.to(device)

def train_model(model, loader, epochs, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                Y_batch.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:03d}, average loss: {avg_loss:.4f}")


train_model(model, loader, epochs, learning_rate, device)


# Section F: Device-Safe Sampling
# Create context tensors on the same device as the model.
#
# A reliable way:
# model_device = next(model.parameters()).device

def get_model_device(model):
    return next(model.parameters()).device


def sample_text(seed_text, model, stoi, itos, steps, temperature):
    generated = seed_text
    model_device = get_model_device(model)
    model.eval()

    with torch.no_grad():
        for _ in range(steps):
            context = generated[-model.block_size :]
            context_ids = [[stoi[ch] for ch in context]]
            X_context = torch.tensor(context_ids, dtype=torch.long, device=model_device)

            logits = model(X_context)
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated += itos[next_id]

    return generated


seed_text = text[: model.block_size]
print(sample_text(seed_text, model, stoi, itos, generate_steps, temperature=0.8))


# Section G: Save A Device-Aware Checkpoint
# The checkpoint can contain tensors saved from CPU or GPU.
# Loading code should not assume which one was used.

checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": config,
    "stoi": stoi,
    "itos": itos,
    "text": text,
}

torch.save(checkpoint, checkpoint_path)
print(f"saved checkpoint to: {checkpoint_path}")


# Section H: Load With map_location
# map_location=device means:
# - load onto GPU if this run selected GPU
# - load onto CPU if this run selected CPU
#
# You can also force CPU loading with map_location="cpu".

loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
loaded_config = loaded_checkpoint["config"]

loaded_model = TinyTransformerLM(**loaded_config).to(device)
loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
loaded_model.eval()

print(f"loaded model device: {get_model_device(loaded_model)}")


# Section I: Verify The Loaded Model
# Compare logits from the original model and loaded model.
# Use the same batch and eval mode so dropout does not add randomness.

X_check, _ = next(iter(loader))
X_check = X_check.to(device)

model.eval()
loaded_model.eval()

with torch.no_grad():
    original_logits = model(X_check)
    loaded_logits = loaded_model(X_check)

print(f"logits match after checkpoint load: {torch.allclose(original_logits, loaded_logits)}")
print(f"max absolute difference: {(original_logits - loaded_logits).abs().max()}")

torch.manual_seed(0)
original_text = sample_text(seed_text, model, stoi, itos, generate_steps, temperature=0.8)

torch.manual_seed(0)
loaded_text = sample_text(
    seed_text,
    loaded_model,
    loaded_checkpoint["stoi"],
    loaded_checkpoint["itos"],
    generate_steps,
    temperature=0.8,
)

print(original_text)
print(loaded_text)
print(f"generated text matches: {original_text == loaded_text}")


# Reflection
# Why must inputs and model parameters be on the same device?
# Why do registered buffers move when you call model.to(device)?
# Why is map_location useful when loading checkpoints?
# Why should DataLoader batches usually be moved inside the training loop?
# What problems might happen if you save on GPU and load on a CPU-only machine?


# Next Lesson Preview
# Save optimizer state and resume training from a checkpoint.
