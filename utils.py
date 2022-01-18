import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from params import PARAMS

def load_data(dataset_path):
    dataset = pickle.load(open(dataset_path, 'rb'))
    y_train = dataset['y_train']
    X_train = dataset['X_train'].astype(np.float32)
    y_test = dataset['y_test']
    X_test = dataset['X_test'].astype(np.float32)

    y_train = np.vectorize(PARAMS['LABEL_MAP'].__getitem__)(y_train)
    y_test = np.vectorize(PARAMS['LABEL_MAP'].__getitem__)(y_test)

    return X_train, y_train, X_test, y_test

def prepare_data(X, y, seq_len):
    n_channels = X.shape[1]

    len_tail = X.shape[0] % seq_len
    if len_tail == 0:
        X = X.reshape(-1, seq_len, n_channels)
        X = np.moveaxis(X, 1, -1)
        y = y.reshape(-1, seq_len)
    else:
        X = X[:-len_tail].reshape(-1, seq_len, n_channels)
        X = np.moveaxis(X, 1, -1)
        y = y[:-len_tail].reshape(-1, seq_len)
    y = y[:, -1]

    return X, y

def prepare_data_cnn(X, y, seq_len):
    n_channels = X.shape[1]

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

    return X, y

def prepare_data_rnn(X, y, seq_len):
    n_channels = X.shape[1]

    len_tail = X.shape[0] % seq_len
    if len_tail == 0:
        X = X.reshape(-1, seq_len, n_channels)
        X = np.moveaxis(X, 1, -1)
        y = y.reshape(-1, seq_len)
    else:
        X = X[:-len_tail].reshape(-1, seq_len, n_channels)
        X = np.moveaxis(X, 1, -1)
        y = y[:-len_tail].reshape(-1, seq_len)
    y = y[:, -1]

    return X, y



def print_classification_report(y_true, y_preds, num_classes):
    # cr = classification_report(y_true, y_preds, output_dict=True, target_names=class_names)
    cr = classification_report(y_true, y_preds, output_dict=True)

    cm = confusion_matrix(y_true, y_preds)

    y_preds_ohe = np.zeros((len(y_preds), num_classes))
    for i, j in enumerate(y_preds):
        y_preds_ohe[i, j] = 1

    y_true_ohe = np.zeros((len(y_true), num_classes))
    for i, j in enumerate(y_true):
        y_true_ohe[i, j] = 1
    auroc = roc_auc_score(y_true_ohe, y_preds_ohe, multi_class='ovo')

    return cr, cm, auroc

def plot_history(history, save_path):
    df = pd.DataFrame(history)
    df["Epochs"] = range(len(df))
    
    fig = px.line(df, x='Epochs', y=['loss', 'val_loss'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_loss.html'))

    fig = px.line(df, x='Epochs', y=['acc', 'val_acc'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_acc.html'))

    # fig = px.line(df, x='Epochs', y=['lrs'])
    # fig.write_html(os.path.join(save_path, 'history_lrs.html'))

def plot_cm(cm, save_path):
    plt.figure(figsize=(7, 5))
    # cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    cm_df = pd.DataFrame(cm)
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cm.png'))
    # plt.show()
    plt.clf()
    plt.close()

def plot_adj(adj, save_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(adj, fmt='g')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    plt.close()