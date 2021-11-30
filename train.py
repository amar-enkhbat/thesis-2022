import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import time
import copy
import random
import numpy as np
from params import PARAMS

def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    X_train, y_train = torch.tensor(X_train).to(PARAMS['DEVICE']), torch.tensor(y_train).to(PARAMS['DEVICE'])
    X_valid, y_valid = torch.tensor(X_valid).to(PARAMS['DEVICE']), torch.tensor(y_valid).to(PARAMS['DEVICE'])
    X_test, y_test = torch.tensor(X_test).to(PARAMS['DEVICE']), torch.tensor(y_test).to(PARAMS['DEVICE'])

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset), 'test': len(test_dataset)}
    dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
    return dataloaders, dataset_sizes

def init_model_params(model, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

# This func copied from 
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# and slightly edited
def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, num_epochs, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    since = time.time()

    # For saving history
    train_loss = torch.zeros(num_epochs)
    val_loss = torch.zeros(num_epochs)
    train_acc = torch.zeros(num_epochs)
    val_acc = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch}/{num_epochs-1}')
        # print('-' * 10)
        for phase in ['train', 'val']:
            # print('#' * len(phase))
            # print(phase)
            # print('#' * len(phase))
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(PARAMS['DEVICE'])
                labels = labels.to(PARAMS['DEVICE'])

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # Save history
            if phase == 'train':
                train_acc[epoch] = epoch_acc
                train_loss[epoch] = epoch_loss
            elif phase == 'val':
                val_acc[epoch] = epoch_acc
                val_loss[epoch] = epoch_loss
            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # print()

    time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s')

    
    return model, {'train_loss': train_loss.tolist(), 'train_acc': train_acc.tolist(), 'val_loss': val_loss.tolist(), 'val_acc': val_acc.tolist(), 'train_time': time_elapsed}
    
def train_model_best_acc(dataloaders, dataset_sizes, model, criterion, optimizer, num_epochs, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = torch.zeros(num_epochs)
    val_loss = torch.zeros(num_epochs)
    train_acc = torch.zeros(num_epochs)
    val_acc = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch}/{num_epochs-1}')
        # print('-' * 10)

        for phase in ['train', 'val']:
            # print('#' * len(phase))
            # print(phase)
            # print('#' * len(phase))
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(PARAMS['DEVICE'])
                labels = labels.to(PARAMS['DEVICE'])

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            if phase == 'train':
                train_acc[epoch] = epoch_acc
                train_loss[epoch] = epoch_loss
            elif phase == 'val':
                val_acc[epoch] = epoch_acc
                val_loss[epoch] = epoch_loss
            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    # print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s')
    # print(f'Best val acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    
    return model, {'train_loss': train_loss.tolist(), 'train_acc': train_acc.tolist(), 'val_loss': val_loss.tolist(), 'val_acc': val_acc.tolist(), 'train_time': time_elapsed}
