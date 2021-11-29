from layers_batchwise_auto import GraphConvolutionAuto
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils import compute_adj_matrices

import numpy as np
from params import PARAMS

class FCN(nn.Module):
    def __init__(self, in_features, num_classes, n_nodes, hidden_sizes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)

        out = self.flatten(out)
        out = self.fc4(out)

        return out

class CNN(nn.Module):
    def __init__(self, kernel_size, seq_len, n_kernels, hidden_size, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=kernel_size)

        self.fc1 = nn.Linear(n_kernels * (seq_len - kernel_size[1] + 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.flatten(out)
        self.dropout = nn.Dropout()
        out = F.relu(self.fc1(out))
        self.dropout = nn.Dropout()

        out = self.flatten(out)
        out = self.fc2(out)

        return out


class RNN(nn.Module):
    def __init__(self, input_size, n_layers, hidden_size, n_classes):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(PARAMS['DEVICE'])
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(PARAMS['DEVICE'])

        out, _ = self.rnn(x, (h0, c0))
        # out: batch_size, seq_length, hidden_size
        # out: (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        return out

class GCN(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionAuto(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = GraphConvolutionAuto(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = GraphConvolutionAuto(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        # self.identity = torch.eye(n_nodes).to(PARAMS['DEVICE'])
        self.node_embeddings = torch.from_numpy(compute_adj_matrices('n') + np.eye(64, dtype=np.float32)).to(PARAMS['DEVICE'])
        # print(self.node_embeddings)
    def forward(self, x):
        # A = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # A = torch.add(A, self.identity)
        out = F.relu(self.gc1(x, self.node_embeddings))
        self.dropout = nn.Dropout()
        out = F.relu(self.gc2(out, self.node_embeddings))
        self.dropout = nn.Dropout()
        out = F.relu(self.gc3(out, self.node_embeddings))
        self.dropout = nn.Dropout()

        out = self.flatten(out)
        out = self.linear(out)

        return out

class GCNAuto(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GCNAuto, self).__init__()
        self.gc1 = GraphConvolutionAuto(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = GraphConvolutionAuto(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = GraphConvolutionAuto(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)
    def forward(self, x):
        # A = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # A = F.sigmoid(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))))
        # A = F.dropout(A, 0.5)
        # A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

        # x dim: [N, n_nodes, node_feat]
        # x_dim: [32, 64, 100]
        # print('Node embeddings:')
        # print(self.node_embeddings.shape)
        out = F.relu(self.gc1(x, self.node_embeddings))
        self.dropout = nn.Dropout()
        # [N, n_nodes, out_features]
        out = F.relu(self.gc2(out, self.node_embeddings))
        self.dropout = nn.Dropout()
        out = F.relu(self.gc3(out, self.node_embeddings))
        self.dropout = nn.Dropout()

        # out = F.relu(self.gc1(x, A))
        # # [N, n_nodes, out_features]
        # out = F.relu(self.gc2(out, A))
        # out = F.relu(self.gc3(out, A))

        out = self.flatten(out)
        out = self.linear(out)

        return out