import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class tttdata(Dataset):
    def __init__(self, input, output):
        self.x = torch.tensor([input], dtype=torch.float32)
        self.y = torch.tensor([output], dtype=torch.float32)
        self.length = len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.length


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_relu = nn.Sequential(
            nn.Linear(9,16)
        )
        
    def forward(self, input):
        res = self.linear_relu(input)
        return res


data = pd.read_csv('tic_tac_toe.csv')
x_values = []
char_map = {"x":1, "o":2, "-":-1}
length_of_data = len(data["String"])

for pos in data["String"]:
    x_values.append([char_map[x] for x in pos])

y_values = []
y_possibilities = set()
for n in range(length_of_data):
    y_possibilities.add(data["Value"][n] + str(data["Remoteness"][n]))

y_possibilities = list(y_possibilities)
y_possibilities.sort()

#['Lose0', 'Lose2', 'Lose4', 'Tie0', 'Tie1', 'Tie2', 'Tie3', 'Tie4', 'Tie5', 'Tie6', 'Tie7', 'Tie8', 'Tie9', 'Win1', 'Win3', 'Win5']

for n in range(length_of_data):
    y_values.append(np.asarray([1 if data["Value"][n] + str(data["Remoteness"][n]) == y_possibilities[k] else 0 for k in range(len(y_possibilities))]))


def accuracy_calc(corr, pred):
    c = 0
    for n in range(len(corr)):
        cl, pl = list(corr[n]), pred[n].tolist()
        cm, pm = max(cl), max(pl)
        cidx, pidx = cl.index(cm), pl.index(pm)
        if cidx == pidx:
            c += 1
    return c / len(corr)


learning_rate = 0.01
epochs = 10000

dl = tttdata(x_values, y_values)
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

losses = []
accuracy = []
best_acc = 0.0

for epoch in range(epochs):
    for __, (x_train, y_train) in enumerate(dl):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)
        acc = accuracy_calc(y_values, prediction)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        accuracy.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "tttModel.pth")
        
    if (epoch+1) % 100 == 0:  # Print every 10 epochs
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc}")
        
print(best_acc)