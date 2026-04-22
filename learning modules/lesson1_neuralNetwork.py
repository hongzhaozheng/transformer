"""
PyTorch nn practice task 2.

Goal:
Train a slightly bigger neural network on fake data that actually has a pattern.

The classification rule:
- Each sample has 10 input features.
- If feature_0 + feature_1 + feature_2 > 0, the label should be class 1.
- Otherwise, the label should be class 0.

Your tasks:
1. Import torch and torch.nn.
2. Set a random seed with torch.manual_seed(0).
3. Create x with shape (200, 10) using torch.randn.
4. Create y from x:
   - add columns 0, 1, and 2
   - if the sum is greater than 0, label is 1
   - otherwise, label is 0
   - y must contain integer class labels, not floats
5. Split the data:
   - first 160 samples for training
   - last 40 samples for testing
6. Create a class named TinyClassifier that inherits from nn.Module.
7. In __init__, build:
   - Linear layer: 10 -> 32
   - ReLU
   - Linear layer: 32 -> 16
   - ReLU
   - Linear layer: 16 -> 2
8. In forward(self, x), send x through those layers and return logits.
9. Create the model, CrossEntropyLoss, and Adam optimizer.
10. Train for 100 epochs using only the training data.
11. Every 10 epochs, print:
    - epoch number
    - training loss
    - test accuracy

Hints:
- You can create y with something like: (some_condition).long()
- Test accuracy needs torch.no_grad().
- Predictions come from logits.argmax(dim=1).
- Accuracy is the mean of correct predictions.
- Your final test accuracy should usually be above 85%.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

x = torch.randn(200,10)
temp = x[:,0] + x[:,1] + x[:,2]
y = (temp > 0).long()

x_train = x[:160,:]
x_test = x[160:,:]
y_train = y[:160]
y_test = y[160:]

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,32)
        self.relu = nn.ReLU();
        self.layer2 = nn.Linear(32,16)
        self.layer3 = nn.Linear(16,2)
        
    def forward(self, x):
        logits = self.layer1(x)
        logits = self.relu(logits)
        logits = self.layer2(logits)
        logits = self.relu(logits)
        logits = self.layer3(logits)
        return logits
    
model = TinyClassifier()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.01)

for i in range(100):
    logits = model(x_train)
    loss = loss_fn(logits, y_train)
    
    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 10 == 0:
        with torch.no_grad():
            test_logits = model(x_test)
            preds = test_logits.argmax(dim=1)
            accuracy = (preds == y_test).float().mean()

        print(i, loss.item(), accuracy.item())

    

    