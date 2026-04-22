"""
Lesson 2: Tokenization and fixed-window next-token data

This lesson is the next step toward building a language model.
We will:
1. Start with raw text
2. Build a character vocabulary
3. Map characters to integers and back
4. Create training examples for next-character prediction

This file intentionally shows both:
- the short Python version
- the expanded step-by-step version
"""

text = "transformers are pattern learners"

print("\n=== Lesson 2: Raw Text ===")
print(text)


print("\n=== Step 1: Build The Vocabulary ===")
print("Goal: collect every unique character from the text.")

# Short version:
chars = sorted(set(text))
vocab_size = len(chars)

print("Short version:")
print("chars = sorted(set(text))")
print("Result:", chars)
print("Vocabulary size:", vocab_size)

print("\nExpanded version:")
unique_chars = []
for ch in text:
    if ch not in unique_chars:
        unique_chars.append(ch)

print("Unique chars before sorting:", unique_chars)
unique_chars.sort()
print("Unique chars after sorting:", unique_chars)


print("\n=== Step 2: Build Lookup Tables ===")
print("Goal: map each character to an integer and back.")

# Short version:
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("Short version:")
print("stoi = {ch: i for i, ch in enumerate(chars)}")
print("itos = {i: ch for i, ch in enumerate(chars)}")

print("\nExpanded version for stoi:")
stoi_expanded = {}
for i, ch in enumerate(chars):
    print(f"At index {i}, character is {repr(ch)}")
    stoi_expanded[ch] = i

print("stoi_expanded:", stoi_expanded)

print("\nExpanded version for itos:")
itos_expanded = {}
for i, ch in enumerate(chars):
    itos_expanded[i] = ch

print("itos_expanded:", itos_expanded)

print("\nWhy this matters:")
print("Neural networks do not read raw characters directly.")
print("They work with numbers, so each character gets an integer id.")


print("\n=== Step 3: Encode Text Into Integers ===")
print("Goal: turn each character in the text into its token id.")

# Short version:
encoded = [stoi[ch] for ch in text]

print("Short version:")
print("encoded = [stoi[ch] for ch in text]")
print("Encoded text:", encoded)

print("\nExpanded version:")
encoded_expanded = []
for ch in text:
    token_id = stoi[ch]
    encoded_expanded.append(token_id)
    print(f"{repr(ch)} -> {token_id}")


print("\n=== Step 4: Decode Integers Back To Text ===")
print("Goal: recover the original text from token ids.")

# Short version:
decoded = "".join(itos[i] for i in encoded)

print("Short version:")
print("decoded = \"\".join(itos[i] for i in encoded)")
print("Decoded text:", decoded)

print("\nExpanded version:")
decoded_chars = []
for token_id in encoded:
    decoded_chars.append(itos[token_id])

decoded_expanded = "".join(decoded_chars)
print("Decoded text:", decoded_expanded)


print("\n=== Step 5: Build Fixed-Window Training Pairs ===")
print("Goal: use previous characters to predict the next character.")

block_size = 4
X = []
Y = []

print(f"Block size: {block_size}")
print("This means the model sees 4 previous characters and predicts the next one.")

for i in range(len(encoded) - block_size):
    context = encoded[i:i + block_size]
    target = encoded[i + block_size]
    X.append(context)
    Y.append(target)

print("Number of training examples:", len(X))


print("\n=== Step 6: Inspect The First Few Examples ===")
for i in range(min(8, len(X))):
    context_ids = X[i]
    target_id = Y[i]

    context_text = "".join(itos[idx] for idx in context_ids)
    target_text = itos[target_id]

    print(f"\nExample {i}")
    print("context ids  :", context_ids)
    print("context text :", repr(context_text))
    print("target id    :", target_id)
    print("target text  :", repr(target_text))
    print(f"Meaning      : given {repr(context_text)}, predict {repr(target_text)}")


print("\n=== Summary ===")
print("You started with raw text.")
print("Then you built a vocabulary.")
print("Then you created character <-> integer mappings.")
print("Then you encoded text into token ids.")
print("Then you built next-token training examples.")


print("\n=== Why This Matters For Transformers ===")
print("Transformers do not start from raw strings.")
print("They start from token ids.")
print("This lesson shows the data pipeline that comes before embeddings and attention.")


print("\n=== Next Lesson Preview ===")
print("Next we can convert X and Y into torch tensors")
print("and build a small nn.Embedding-based model for next-token prediction.")
