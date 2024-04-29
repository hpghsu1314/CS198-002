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
    

class tttRemotenessData(Dataset):
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
            nn.Sigmoid(),
            nn.Linear(in_features=5, out_features=9),
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

    def createRemotenessDataloader(self):
        self.vectorizeKeys()
        tempRemoteness = [[1 if self.remoteness[n] - 1 == p else 0 for p in range(9)] for n in range(len(self.remoteness))]
        dataloader = DataLoader(tttRemotenessData(self.results[0], tempRemoteness))
        return dataloader

learning_rate = 0.003
epochs = 700

best_acc = 0.0
data = Classification(path)
dl = data.createRemotenessDataloader()
model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

losses = []
accuracy = []

def accuracy_calc(corr, pred):
    cl, pl = corr.tolist()[0], pred.tolist()[0]
    cm, pm = max(cl), max(pl)
    cidx, pidx = cl.index(cm), pl.index(pm)
    return 1 if cidx == pidx else 0


for epoch in range(epochs):
    accurate = 0
    for __, (x_train, y_train) in enumerate(dl):
        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_fn(output, y_train)
        accurate += accuracy_calc(y_train, output)
        loss.backward()
        optimizer.step()
    acc = accurate / len(dl)
    losses.append(loss)
    accuracy.append(acc)
    print("epoch {}\tloss : {}\t accuracy : {}".format(epoch,loss,acc))
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "tttHashModel.pth")
print(best_acc)