"""
Lesson 3: Tensors + Embeddings + A Tiny Next-Token Model

This lesson continues directly from lesson 2.
Work through the tasks below and implement each step yourself.
"""

import torch
import torch.nn as nn


torch.manual_seed(0)

text = "transformers are pattern learners"
block_size = 4
embed_dim = 8
hidden_dim = 64
epochs = 300


print("\n=== Lesson 3: Build Training Tensors ===")
print("Raw text:", repr(text))


# Task 1:
# Build the character vocabulary.
# Suggested variables:
# - chars: sorted unique characters from text
chars = sorted(set(text))


# - vocab_size: number of unique characters
vocab_size = len(chars)


# - stoi: char -> integer id
stoi = {ch:i for i, ch in enumerate(chars)}


# - itos: integer id -> char
itos = {i:ch for i, ch in enumerate(chars)}
print("itos: ", itos)

# Task 2:
# Encode the full text into integer ids.
# Suggested variable:
# - encoded: list of token ids for each character in text
encoded = [stoi[ch] for ch in text]


# Task 3:
# Print basic tokenization info:
# - Vocabulary size
print(vocab_size)
# - Character vocabulary
print("stoi: ", stoi)
# - Encoded text
print("encoded: ", encoded)

# Task 4:
# Build the training examples for next-character prediction.
# For each position i:
# - context = encoded[i:i + block_size]
# - target = encoded[i + block_size]
# Append contexts to X and targets to Y.

X = []
Y = []
for i in range(len(text) - block_size):
    context = encoded[i:i+block_size]
    target = encoded[i+block_size]
    X.append(context)
    Y.append(target)
print("X: ", X)
print("Y: ", Y)


# Task 5:
# Convert X and Y into torch tensors with dtype=torch.long.
# Expected shapes:
# - X: [num_examples, block_size]
# - Y: [num_examples]
X = torch.tensor(X, dtype = torch.long)
Y = torch.tensor(Y, dtype = torch.long)

# Task 6:
# Print tensor inspection info:
# - X shape as a tuple with a short explanation
# - Y shape as a tuple with a short explanation
# - First context ids
# - First context text
# - First target id
# - First target text
print("X has shape: ", tuple(X.shape))
print("Y has shape: ", tuple(Y.shape))
print("First context ids: ", X[0].tolist())
print("First context text: ", "".join(itos[i] for i in X[0].tolist()))
print("First target id: ", Y[0].item())
print("First target text: ", itos[Y[0].item()])


class CharMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, hidden_dim):
        super().__init__()

        # Task 7:
        # Define the model layers:
        # - embedding: nn.Embedding(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # - linear1: nn.Linear(block_size * embed_dim, hidden_dim)
        self.fc1 = nn.Linear(block_size*embed_dim, hidden_dim)
        # - relu: nn.ReLU()
        self.relu = nn.ReLU()
        # - linear2: nn.Linear(hidden_dim, vocab_size)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Task 8:
        # Implement the forward pass.
        # Suggested steps:
        # 1. Look up token embeddings
        # 2. Flatten from [batch, block_size, embed_dim] to [batch, block_size * embed_dim]
        # 3. Apply first linear layer
        # 4. Apply ReLU
        # 5. Apply final linear layer to get logits
        # 6. Return logits
        embedded = self.embedding(x)
        flat = embedded.view(x.shape[0], -1)
        hidden = self.relu(self.fc1(flat))
        logits = self.fc2(hidden)

        return logits


# Task 9:
# Instantiate the model using:
# - vocab_size
# - embed_dim
# - block_size
# - hidden_dim
model = CharMLP(vocab_size, embed_dim, block_size, hidden_dim)

# Task 10:
# Create:
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

print("\n=== Model Check ===")

# Task 11:
# Run a small batch through the model:
sample_logits = model(X[:3])
# Print:
# - input batch shape
print("input batch shape: ", tuple(X[:3].shape))
# - logits shape with a short explanation


print("\n=== Training ===")

# Task 12:
# Write the training loop for the requested number of epochs.
# Each epoch should:
# - compute logits = model(X)
# - compute loss = loss_fn(logits, Y)
# - zero gradients
# - backpropagate
# - optimizer step
#
# Every 50 epochs (and the final epoch), print:
# - epoch number
# - loss
# - train accuracy
#
# Hint:
# - predictions = logits.argmax(dim=1)
# - accuracy = (predictions == Y).float().mean().item()

for epoch in range(epochs):
    logits = model(X)
    loss = loss_fn(logits,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == epochs - 1:
        predictions = logits.argmax(dim = 1)
        acc = (predictions == Y).float().mean().item()
        print(f"Epoch: {epoch}, Loss: {loss.item():.3f}, Accuracy: {acc:.3f}")
        


print("\n=== Inspect A Few Predictions ===")

# Task 13:
# Evaluate the model without gradient tracking.
# Suggested steps:
# - logits = model(X)
# - predictions = logits.argmax(dim=1)
#
# Then inspect up to the first 10 examples.
# For each example:
# - get context_ids from X[i]
# - get target_id from Y[i]
# - get predicted_id from predictions[i]
# - decode them back to text using itos
# - print:
#   context='....' target='.' predicted='.'

with torch.no_grad():
    logits = model(X)
    predictions = logits.argmax(dim=1)

    for i in range(min(10, X.shape[0])):
        context_ids = X[i].tolist()
        target_id = Y[i].tolist()
        predicted_id = predictions[i].tolist()
        context = "".join(itos[i] for i in context_ids)
        target = itos[target_id]
        pred = itos[predicted_id]
        print(f"Context: {context}, Target: {target}, Prediction: {pred}")

print("\n=== Why This Matters ===")
print("The embedding layer turns token ids into learned vectors.")
print("The model uses the previous 4 characters to predict the next one.")
print("This is a small language model and a direct warm-up for Transformers.")


print("\n=== Next Lesson Preview ===")
print("Next we can switch from full-batch training to DataLoader mini-batches,")
print("then add autoregressive text generation from the trained model.")
