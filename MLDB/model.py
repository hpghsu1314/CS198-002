import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

path = "ttts.txt"

class tttdata(Dataset):
    def __init__(self, input, output):
        self.x = torch.tensor(input, dtype=torch.float32)
        self.y = torch.tensor(output, dtype=torch.float32)
        self.length = len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.length


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_relu = nn.Sequential(
            nn.Linear(5,16),
            nn.LogSigmoid(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,16)
        )
        
    def forward(self, input):
        res = self.linear_relu(input)
        return res


class Classification:
    
    def __init__(self, file_path):
        self.file = pd.read_csv(file_path)
        self.hashcode = self.file['Hash'].tolist()
        self.prim = self.file['Primitive'].tolist()
        self.remoteness = self.file['Remoteness'].tolist()
        self.length = len(self.hashcode)
        self.results = [self.hashcode, [str(self.prim[n]) + str(self.remoteness[n]) for n in range(self.length)]]
        self.legend = set()
        for i in range(self.length):
            self.legend.add(self.results[1][i])
        self.legend = list(self.legend)
        self.legend.sort()
        self.hashKeyMaxLength = len(str(self.results[0][-1]))
        self.classified = False
        self.vectorized = False
    
    def classifyResult(self):
        if self.classified:
            return
        for n in range(self.length):
            v = [1 if self.results[1][n] == self.legend[i] else 0 for i in range(len(self.legend))]
            self.results[1][n] = v
        self.classified = True
        return

    def vectorizeKeys(self):
        if self.vectorized:
            return
        for n in range(self.length):
            v = [0 if i > len(str(self.results[0][n])) else int(str(self.results[0][n])[-i]) for i in range(self.hashKeyMaxLength, 0, -1)]
            self.results[0][n] = v
        self.vectorized = True
        return
    
    def createCustomDataloader(self):
        self.classifyResult()
        self.vectorizeKeys()
        dataloader = DataLoader(tttdata(self.results[0], self.results[1]))
        return dataloader

learning_rate = 0.01
epochs = 700

data = Classification(path)
dl = data.createCustomDataloader()
model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

losses = []
accuracy = []

def accuracy_calc(corr, pred):
    c = 0
    for n in range(len(corr)):
        cl, pl = list(corr[n]), pred[n].tolist()
        cm, pm = max(cl), max(pl)
        cidx, pidx = cl.index(cm), pl.index(pm)
        if cidx == pidx:
            c += 1
    return c / len(corr)


for epoch in range(epochs):
    for __, (x_train, y_train) in enumerate(dl):
        output = model(x_train)
        loss = loss_fn(output, y_train)
        predicted = model(torch.tensor(data.results[0], dtype=torch.float32))
        #acc = (predicted.reshape(-1).detach().numpy().round()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss)
    accuracy.append(acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(epoch,loss,acc))