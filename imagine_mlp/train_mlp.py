import os
import torch
from torch import nn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

class PhysionetDataset(Dataset):
    def __init__(self, dataset_path, train=True, debug=False, one_hot_encoding=True):
        """
        Args:
            dataset_dir (string): Path to Physionet MMI dataset with .csv extensions
        """
        self.label_map = {'imagine_both_feet': 0, 'imagine_both_fist': 1, 'imagine_left_fist': 2, 'imagine_right_fist': 3}
        self.X = pickle.load(open(dataset_path, 'rb'))
        if train:
            self.y = self.X['train_y']
            self.X = self.X['train_x'].astype(np.float32)
        else:
            self.y = self.X['test_y']
            self.X = self.X['test_x'].astype(np.float32)

        if debug:
            self.y = self.y[:1000]
            self.X = self.X[:1000]

        if one_hot_encoding:
            
            self.y = np.vectorize(self.label_map.__getitem__)(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        sample = self.X[idx], self.y[idx]
        return sample

    def get_labels(self):
        return np.unique(self.y)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return train_loss.item(), correct

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

if __name__ == '__main__':

    debug = False
    batch_size = 64
    
    training_data = PhysionetDataset('../dataset/train/cross_subject_data_0.pickle', train=True, debug=debug, one_hot_encoding=True)
    test_data = PhysionetDataset('../dataset/train/cross_subject_data_0.pickle', train=False, debug=debug, one_hot_encoding=True)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    model = MLP().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 200

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------------")
        loss, acc = train(train_dataloader, model, loss_fn, optimizer)
        train_loss.append(loss)
        train_acc.append(acc)

        loss, acc = test(test_dataloader, model, loss_fn)
        test_loss.append(loss)
        test_acc.append(acc)
    print("Done!")
    history = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}
    pickle.dump(history, open('history.pickle', 'wb'))
