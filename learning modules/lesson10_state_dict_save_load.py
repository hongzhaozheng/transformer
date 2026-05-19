"""
Lesson 10: Saving + Loading Model Weights With state_dict

This lesson builds on Lesson 9.

Goal:
- reuse the tiny decoder-only Transformer language model
- train it briefly on the character next-token task
- save model weights with state_dict
- load those weights into a fresh model with the same architecture
- verify that the loaded model produces the same logits
- generate text from the reloaded model

Big idea:
The state_dict stores the learned tensors in a model, but it does not store the
Python class definition or the model hyperparameters. To load weights correctly,
you must recreate the same architecture first, then call load_state_dict.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
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
weights_path = Path("tiny_transformer_lm_state_dict.pt")
checkpoint_path = Path("tiny_transformer_lm_checkpoint.pt")


# Section A: Dataset Refresh
# Rebuild the same character next-token dataset from Lesson 9.
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
# - vocabulary
# - vocab_size
# - number of examples
# - input/target tensor shapes

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
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


# Section B: Rebuild The Lesson 9 Model
# Recreate the model classes from Lesson 9:
#
# - SingleHeadSelfAttention
# - MultiHeadSelfAttention
# - FeedForward
# - DecoderBlock
# - TinyTransformerLM
#
# Important:
# state_dict loading depends on matching parameter names and shapes.
# If you change a layer name, d_model, num_heads, vocab_size, or num_layers,
# an old state_dict may not load cleanly into the new model.

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
        weights = torch.softmax(scores, dim=-1)
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
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        out = torch.cat(heads, dim=-1)
        return self.dropout(self.linear(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


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
        self.pos_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, block_size, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        T = idx.shape[-1]
        if T > self.block_size:
            raise ValueError("idx length exceeds block_size")

        token_embedding = self.token_embedding(idx)
        pos_id = torch.arange(T, device=idx.device)
        pos_embedding = self.pos_embedding(pos_id)
        x = self.dropout(token_embedding + pos_embedding)

        for block in self.decoder:
            x = block(x)

        logits = self.linear(self.norm(x))
        return logits


# Section C: Training Helper
# Write a small train_model helper so training can be reused.
#
# Implement:
# - def train_model(model, loader, epochs, learning_rate)
#
# Inside:
# - call model.train()
# - use AdamW
# - use CrossEntropyLoss
# - reshape logits to [B*T, vocab_size]
# - reshape targets to [B*T]
# - print average loss every 20 epochs

def train_model(model, loader, epochs, learning_rate):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for X,Y in loader:
            logits = model(X)
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                Y.reshape(-1)
            )
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        avg_loss = total_loss/len(loader)
        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, average loss: {avg_loss}")
    

model = TinyTransformerLM(vocab_size,
        block_size,
        d_model,
        num_heads,
        hidden_size,
        num_layers,
        dropout)

train_model(model, loader, epochs, learning_rate)

# Section D: Save Only The Model Weights
# Save the learned tensors.
#
# Use:
# - torch.save(model.state_dict(), weights_path)
#
# Then inspect:
# - how many tensors are stored
# - a few state_dict keys
#
# Notice:
# The file contains weights and buffers, not the model class itself.

# switch the trained model to eval mode.
model.eval()

# get model.state_dict().
state_dict = model.state_dict()

# save the state_dict to weights_path.
torch.save(state_dict, weights_path)

# print the save path.
print(f"weights have been saved to {weights_path}")

# print how many tensors are in the state_dict.
print(f"number of tensors in the state_dict: {len(state_dict)}")

# print the first few state_dict keys and tensor shapes.
for key in list(state_dict)[:5]:
    print(f"  {key}: {tuple(state_dict[key].shape)}")



# Section E: Create A Fresh Untrained Model
# Instantiate a second model with the same hyperparameters.
#
# Before loading:
# - compare original model logits with fresh model logits
# - they should not match because the fresh model has random weights
#
# Important:
# Use eval mode before comparing so dropout does not create noise.

# instantiate a fresh TinyTransformerLM with the same hyperparameters.
fresh_model = TinyTransformerLM(vocab_size,
        block_size,
        d_model,
        num_heads,
        hidden_size,
        num_layers,
        dropout
        )

# pull one batch from loader.
X, Y = next(iter(loader))

# put both models in eval mode.
model.eval()
fresh_model.eval()

# use torch.no_grad() to compute logits from both models.
with torch.no_grad():
    logits1 = model(X)
    logits2 = fresh_model(X)

# print whether the logits match before loading.
print(f"the logits match: {torch.allclose(logits1, logits2)}")


# Section F: Load Weights Into The Fresh Model
# Load the saved state_dict into the fresh model.
#
# Use:
# - loaded_state_dict = torch.load(weights_path, map_location='cpu')
# - fresh_model.load_state_dict(loaded_state_dict)
#
# Then compare logits again.
# They should match because both models now have the same weights.

def load_torch_file(path):
    try:
        return torch.load(path, map_location='cpu', weights_only = True)
    except TypeError:
        return torch.load(path, map_location='cpu')

# load the saved state_dict from weights_path.
loaded_state_dict = load_torch_file(weights_path)

# call fresh_model.load_state_dict(...).
fresh_model.load_state_dict(loaded_state_dict)

# compute logits from the loaded model.
fresh_model.eval()
with torch.no_grad():
    logits = fresh_model(X)

# print whether the original and loaded logits match.
print(f"the logits match: {torch.allclose(logits1, logits)}")

# print the maximum absolute difference between the two logits tensors.
diff = (logits1 - logits).abs().sum()
print(f"the maximum absolute difference between the two logits tensors: {diff}")


# Section G: Generate From The Reloaded Model
# Reuse the sampling helper from Lesson 9.
#
# Implement:
# - def sample_text(seed_text, model, stoi, itos, steps, temperature)
#
# Make sure:
# - generation calls model.eval()
# - generation uses torch.no_grad()
# - context length is clipped with model.block_size

def sample_text(seed_text, model, stoi, itos, steps, temperature):
   generated = seed_text
   model.eval()
   with torch.no_grad():
       for step in range(steps):
           context = generated[-model.block_size:]
           context_id = torch.tensor([[stoi[ch] for ch in context]], dtype = torch.long)
           logits = model(context_id)
           last_logits = logits[:,-1,:] / temperature
           prob = torch.softmax(last_logits, dim = -1)
           next_id = torch.multinomial(prob, num_samples = 1)
           generated += itos[next_id.item()]
       
   return generated


# set seed_text.
seed_text = text[:model.block_size]

# generate from the original model.
torch.manual_seed(12)
original_text = sample_text(seed_text, model, stoi, itos, generate_steps, 0.8)

# reset torch.manual_seed(...) and generate from the reloaded model.
torch.manual_seed(12)
reloaded_text = sample_text(seed_text, fresh_model, stoi, itos, generate_steps, 0.8)

# confirm both models can generate text after loading.
print(original_text)
print(reloaded_text)

# Section H: Save A Training Checkpoint
# A raw state_dict is enough when you already know the model settings.
# A checkpoint is more convenient because it stores weights plus metadata.
#
# Save:
# - model_state_dict
# - config
# - stoi
# - itos
# - text
#
# This makes it easier to reconstruct the model later.

# create a config dictionary with the model hyperparameters.
config = {
    "vocab_size": vocab_size,
    "block_size": block_size,
    "d_model": d_model,
    "num_heads": num_heads,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "dropout": dropout
}

# create a checkpoint dictionary with model_state_dict, config, stoi, itos, and text.
cp_dic = {
    "model_state_dict": model.state_dict(),
    "config": config,
    "stoi": stoi,
    "itos": itos,
    "text": text
}

# save the checkpoint to checkpoint_path.
torch.save(cp_dic, checkpoint_path)

# print the checkpoint path.
print(f"checkpoint has been saved to {checkpoint_path}")

# Section I: Rebuild From The Checkpoint
# Load the checkpoint and rebuild the model using the saved config.
#
# This pattern is closer to what you will use in real projects:
# - load checkpoint
# - read config
# - instantiate the model
# - load model_state_dict
# - switch to eval mode for generation

# load the checkpoint from checkpoint_path.
checkpoint = torch.load(checkpoint_path)

# read the saved config.
loaded_config = checkpoint["config"]

# rebuild the model with TinyTransformerLM(**loaded_config).
rebuilt_model = TinyTransformerLM(**loaded_config)

# load the saved model_state_dict.
rebuilt_model.load_state_dict(checkpoint["model_state_dict"])

# switch the checkpoint model to eval mode.
rebuilt_model.eval()

# generate text from the checkpoint model.
torch.manual_seed(12)
with torch.no_grad():
    new_text = sample_text(seed_text, rebuilt_model, checkpoint["stoi"], checkpoint["itos"], generate_steps, 0.8)
print(f"model and rebuilt_model produce the same text: {original_text == new_text}")

# Reflection
# What is stored in model.state_dict()?
# What is not stored in model.state_dict()?
# Why must the model architecture match before loading weights?
# Why should you call model.eval() before comparing loaded model outputs?
# What extra information is useful to store in a checkpoint?
# When would you save only weights, and when would you save a checkpoint?


# Next Lesson Preview
# Move the training code to GPU when available, then save and load across devices.
