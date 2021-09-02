from convert_to_graphs import n_graph
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from convert_to_graphs import n_graph, d_graph, s_graph, normalize_adj
import mne
import pandas as pd
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(dataset_path):
    X = pickle.load(open(dataset_path, 'rb'))
    y = X['train_y']

    X = X['train_x'].astype(np.float32)

    label_map = {'imagine_both_feet': 0, 'imagine_both_fist': 1, 'imagine_left_fist': 2, 'imagine_right_fist': 3}
    y = np.vectorize(label_map.__getitem__)(y)
    


    return X, y, label_map

def prepare_data(X, y, seq_len, normalize):
    n_channels = X.shape[1]

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    print('X original shape:', X.shape)
    print('y original shape:', y.shape)
    print('Seq len:', seq_len)

    len_tail = X.shape[0] % seq_len
    if len_tail == 0:
        X = X.reshape(-1, 1, seq_len, n_channels)
        X = np.moveaxis(X, 2, -1)
        y = y.reshape(-1, seq_len)
    else:
        X = X[:-len_tail].reshape(-1, 1, seq_len, n_channels)
        X = np.moveaxis(X, 2, -1)
        y = y[:-len_tail].reshape(-1, seq_len)
    y = y[:, -1]
    print('X conversion shape:', X.shape)
    print('y conversion shape:', y.shape)
    return X, y

def compute_adj_matrices(type):
    ten_twenty_montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = pd.read_csv("../dataset/physionet.org_csv/S001/S001R01.csv")
    ch_names = ch_names.columns[2:]
    n_channels = 64

    ch_pos_1020 = ten_twenty_montage.get_positions()["ch_pos"]

    ch_pos_1010 = {}
    for ch_name_orig in ch_names:
        ch_name = ch_name_orig.upper().rstrip(".")
        if "Z" in ch_name:
            ch_name = ch_name.replace("Z", "z")
        if "P" in ch_name and len(ch_name) > 2:
            ch_name = ch_name.replace("P", "p")
        if "Cp" in ch_name:
            ch_name = ch_name.replace("Cp", "CP")
        if "Tp" in ch_name:
            ch_name = ch_name.replace("Tp", "TP")
        if "pO" in ch_name:
            ch_name = ch_name.replace("pO", "PO")
        ch_pos_1010[ch_name_orig] = ch_pos_1020[ch_name]
    print(len(ch_pos_1010))

    ch_pos_1010_names = []
    ch_pos_1010_dist = []
    for name, value in ch_pos_1010.items():
        ch_pos_1010_names.append(name)
        ch_pos_1010_dist.append(value)
    ch_pos_1010_dist = np.array(ch_pos_1010_dist)


    if type=='n':
        A = n_graph()
    elif type=='d':
        A = d_graph(n_channels, ch_pos_1010_dist)
    elif type=='s':
        A = s_graph(n_channels, ch_pos_1010_dist)

    A = normalize_adj(A)
    A = np.array(A, dtype=np.float32)
    return A
    
def get_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
    dataloaders = {'train': train_loader, 'val': test_loader}
    return dataloaders, dataset_sizes

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, num_epochs, writer):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch}/{num_epochs-1}')
        # print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model
    
def init_model_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

def model_predict(model, test_loader):
    y_preds = []
    y_true = []
    for inputs, labels in test_loader:
        _, y_pred = torch.max(model(inputs), 1)
        y_preds.append(y_pred)
        y_true.append(labels)
    y_preds = torch.cat(y_preds)
    y_true = torch.cat(y_true)
    return y_preds, y_true

def print_classification_report(y_true, y_preds, num_classes, writer):
    
    cr = classification_report(y_true.cpu().numpy(), y_preds.cpu().numpy(), digits=4)
    print(cr)

    cm = confusion_matrix(y_true.cpu().numpy(), y_preds.cpu().numpy())
    print(cm)

    y_preds_ohe = np.zeros((y_preds.size(0), num_classes))
    for i, j in enumerate(y_preds):
        y_preds_ohe[i, j] = 1

    y_true_ohe = np.zeros((y_true.size(0), num_classes))
    for i, j in enumerate(y_true):
        y_true_ohe[i, j] = 1
    auroc = roc_auc_score(y_true_ohe, y_preds_ohe, multi_class='ovo')
    writer.add_scalar('AUROC OvO', auroc)
    print('AUROC ovo:', auroc)
    auroc = roc_auc_score(y_true_ohe, y_preds_ohe, multi_class='ovr')
    writer.add_scalar('AUROC OvR', auroc)
    print('AUROC ovr:', auroc)
    return cr, cm

def plot_cm(cm, class_names, save_path):
    plt.figure(figsize=(7, 5))
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def plot_adj(adj, save_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(adj, fmt='g')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()