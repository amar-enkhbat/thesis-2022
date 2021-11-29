import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def load_data(dataset_path):
    X = pickle.load(open(dataset_path, 'rb'))
    y = X['train_y']

    X = X['train_x'].astype(np.float32)

    label_map = {'imagine_both_feet': 0, 'imagine_both_fist': 1, 'imagine_left_fist': 2, 'imagine_right_fist': 3}
    y = np.vectorize(label_map.__getitem__)(y)

    return X, y, label_map

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



def print_classification_report(y_true, y_preds, num_classes, class_names):
    cr = classification_report(y_true, y_preds, output_dict=True, target_names=class_names)

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
    fig = px.line(df, x='Epochs', y=['train_loss', 'val_loss'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_loss.html'))
    fig = px.line(df, x='Epochs', y=['train_acc', 'val_acc'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_acc.html'))

def plot_cm(cm, class_names, save_path):
    plt.figure(figsize=(7, 5))
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
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
    plt.savefig(os.path.join(save_path, 'adj.png'))
    # plt.show()
    plt.clf()
    plt.close()