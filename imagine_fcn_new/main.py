import os
import random
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np 

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

from utils import load_data, prepare_data, get_dataloaders, init_model_params, train_model, model_predict, print_classification_report, plot_cm, plot_adj

from model import FCN

def run_auto_gnn_model(run_number, random_seed, summary_dir, num_epochs, batch_size, seq_len, hidden_sizes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    writer = SummaryWriter(summary_dir)

    X, y, label_map = load_data(f'../dataset/train/cross_subject_data_{run_number}.pickle')
    class_names = list(label_map.keys())

    X, y = prepare_data(X, y, seq_len, normalize=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    dataloaders, dataset_sizes = get_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    in_features = seq_len
    num_classes = 4
    n_channels = 64

    model = FCN(in_features=in_features, n_nodes=n_channels, num_classes=num_classes, hidden_sizes=hidden_sizes)

    model = init_model_params(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # writer.add_graph(model, torch.Tensor(X_test[:batch_size]))

    model = model.to(device)

    best_model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, num_epochs, writer)

    y_preds, y_test = model_predict(best_model, test_loader=dataloaders['val'])

    cr, cm = print_classification_report(y_test, y_preds, num_classes, writer)

    plot_cm(cm, class_names, os.path.join(summary_dir, 'cm.png'))
    
    print('Number of trainable parameters')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
if __name__=='__main__':
    random_seed = 1
    n_runs = 5
    n_runs = [i for i in range(n_runs)]
    num_epochs = 100
    batch_size = 32
    seq_len = 100
    hidden_sizes = [256, 512, 256]

    # for run_number in n_runs:
    #     text = f'Run Number:{run_number}'
    #     print('')
    #     print('#'*len(text))
    #     print(text)
    #     print('#'*len(text))
    #     print('')
    #     run_auto_gnn_model(run_number, random_seed, f'runs/gnn_auto_{run_number}', num_epochs, batch_size, seq_len, hidden_sizes)
    #     break
    for i in n_runs:
        text = f'Run Number:{i+1}'
        print('')
        print('#'*len(text))
        print(text)
        print('#'*len(text))
        print('')
        random_seed = i
        run_auto_gnn_model('5_subjects', random_seed, f'runs/gnn_auto_5_subjects', num_epochs, batch_size, seq_len, hidden_sizes)
