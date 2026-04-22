"""
Lesson 4: Mini-Batches + DataLoader + Simple Text Generation

This lesson continues from lesson 3.
Work through the tasks below and implement each step yourself.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


torch.manual_seed(0)

text = "transformers are pattern learners"
block_size = 4
embed_dim = 8
hidden_dim = 64
batch_size = 8
epochs = 300
generate_steps = 20


print("\n=== Lesson 4: Mini-Batches + Generation ===")
print("Raw text:", repr(text))


# Task 1:
# Build the character vocabulary and integer mappings.
# Suggested variables:
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}


# Task 2:
# Encode the text into integer ids.
# Suggested variable:
encoded = [stoi[ch] for ch in text]
print(f"encoded: {encoded}")

# Task 3:
# Build fixed-window next-character examples.
# For each position i:
# - context = encoded[i:i + block_size]
# - target = encoded[i + block_size]
# Append contexts to X and targets to Y.

X = []
Y = []
for i in range(len(encoded) - block_size):
    context = encoded[i:i + block_size]
    target = encoded[i + block_size]
    X.append(context)
    Y.append(target)

print(f"X: {X}")
print(f"Y: {Y}")

# Task 4:
# Convert X and Y into torch tensors with dtype=torch.long.
# Expected shapes:
# - X: [num_examples, block_size]
# - Y: [num_examples]
X = torch.tensor(X, dtype = torch.long)
Y = torch.tensor(Y, dtype = torch.long)

# Task 5:
# Create a TensorDataset and a DataLoader.
# Suggested steps:
# - dataset = TensorDataset(X, Y)
# - loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# Print:
# - number of examples
# - number of batches per epoch

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Number of examples: {len(dataset)}, number of batches per epoch: {len(loader)}")


class CharMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, hidden_dim):
        super().__init__()

        # Task 6:
        # Define the same model as lesson 3:
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim*block_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Task 7:
        # Implement the forward pass.
        # Suggested shapes:
        # - input x: [batch, block_size]
        # - embedded: [batch, block_size, embed_dim]
        # - flat: [batch, block_size * embed_dim]
        # - logits: [batch, vocab_size]
        embedded = self.embedding(x)
        flat = embedded.view(embedded.shape[0], -1)
        logits = self.fc2(self.relu(self.fc1(flat)))
        
        return logits


# Task 8:
# Instantiate:
# - model
# - loss_fn = nn.CrossEntropyLoss()
# - optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
model = CharMLP(vocab_size, embed_dim, block_size, hidden_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)


print("\n=== Batch Check ===")

# Task 9:
# Inspect one mini-batch from the DataLoader.
# Suggested steps:
# - take the first batch from loader
# - unpack it into xb, yb
# - print xb shape
# - print yb shape
# - run sample_logits = model(xb)
# - print sample_logits shape with a short explanation
xb, yb = next(iter(loader))
print(f"xb shape: {tuple(xb.shape)}, yb shape: {tuple(yb.shape)}")

sample_logits = model(xb)
print(f"The shape of sample_logits: {tuple(sample_logits.shape)}")


print("\n=== Training With Mini-Batches ===")

# Task 10:
# Write a mini-batch training loop.
# For each epoch:
# - set total_loss = 0.0
# - set total_correct = 0
# - set total_examples = 0
# - loop over xb, yb in loader
# - compute logits = model(xb)
# - compute loss = loss_fn(logits, yb)
# - zero gradients
# - backpropagate
# - optimizer step
# - accumulate total_loss using loss.item() * xb.size(0)
# - accumulate total_correct using predicted ids
# - accumulate total_examples using xb.size(0)
#
# Every 50 epochs (and the final epoch), print:
# - epoch number
# - average loss across the epoch
# - average accuracy across the epoch

for epoch in range(epochs):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    
    for xb,yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * xb.size(0)
        total_correct += (predictions==yb).sum().item()
        total_examples += xb.size(0)
    
    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}, Average Accuracy: {avg_acc}")

print("\n=== Greedy Text Generation ===")

# Task 11:
# Write a helper function:
# - generate_text(model, seed_text, steps, stoi, itos, block_size)
#
# Suggested generation logic:
# - start from a seed string such as "tran"
# - convert the seed characters to ids
# - keep only the most recent block_size ids as context
# - create a tensor of shape [1, block_size]
# - run the model
# - choose the next token with argmax(dim=1)
# - append the predicted id
# - repeat for the requested number of steps
# - return the decoded string
#
# Notes:
# - Assume the seed text is at least block_size characters long
# - Use model.eval() and torch.no_grad() during generation

def generate_text(model, seed_text, steps, stoi, itos, block_size):
    generated_ids = [stoi[ch] for ch in seed_text]

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            context_ids  = generated_ids[-block_size:]
            x = torch.tensor(context_ids, dtype = torch.long).unsqueeze(0)
            logits = model(x)
            next_id = logits.argmax(dim=1).item()
            generated_ids.append(next_id)

    return "".join(itos[i] for i in generated_ids)


# Task 12:
# Call your generation function with a seed such as "tran"
# and print the generated text.
seed_text = "tran"
steps = 10
print(generate_text(model, seed_text, steps, stoi, itos, block_size))

print("\n=== Why This Matters ===")
print("DataLoader introduces mini-batch training, which is the standard workflow in PyTorch.")
print("Generation shows how a next-token model can be used autoregressively.")
print("This sets up the transition to sequence models and Transformers.")


print("\n=== Next Lesson Preview ===")
print("Next we can add positional information and move toward self-attention.")
