from layers import BatchGraphConvolutionLayer, BatchGraphAttentionLayer, SelfAttentionLayer
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

        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = F.relu(self.fc3(out))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

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

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.flatten(out)
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])
        
        out = self.flatten(out)
        out = self.fc2(out)

        return out

class RNN(nn.Module):
    def __init__(self, input_size, n_layers, hidden_size, n_classes):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=PARAMS['DROPOUT_P'])

        self.fc = nn.Linear(hidden_size, n_classes)
    def forward(self, x):

        out, _ = self.lstm1(x)
        # out: batch_size, seq_length, hidden_size
        # out: (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        return out

class GCN(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes, graph_type='n'):
        super(GCN, self).__init__()
        self.gc1 = BatchGraphConvolutionLayer(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = BatchGraphConvolutionLayer(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = BatchGraphConvolutionLayer(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.identity = torch.eye(n_nodes).to(PARAMS['DEVICE'])
        self.node_embeddings = torch.from_numpy(compute_adj_matrices(graph_type)).to(PARAMS['DEVICE'])
    def forward(self, x):
        A = self.node_embeddings + self.identity
        out = F.relu(self.gc1(x, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = F.relu(self.gc2(out, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])
        
        out = F.relu(self.gc3(out, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = self.flatten(out)
        out = self.linear(out)

        return out

class GCNAuto(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GCNAuto, self).__init__()
        self.gc1 = BatchGraphConvolutionLayer(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = BatchGraphConvolutionLayer(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = BatchGraphConvolutionLayer(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.identity = torch.eye(n_nodes).to(PARAMS['DEVICE'])
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)
    def forward(self, x):
        # A = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        A = torch.mm(self.node_embeddings, self.node_embeddings.T)
        # A = F.dropout(A, 0.5)
        A = A + self.identity

        # x dim: [N, n_nodes, node_feat]
        # x_dim: [32, 64, 100]
        # print('Node embeddings:')
        # print(self.node_embeddings.shape)
        out = F.relu(self.gc1(x, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])
        # [N, n_nodes, out_features]
        out = F.relu(self.gc2(out, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])
        out = F.relu(self.gc3(out, A))
        out = F.dropout(out, p=PARAMS['DROPOUT_P'])

        out = self.flatten(out)
        out = self.linear(out)

        return out

class GCRAMAuto(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GCRAMAuto, self).__init__()
        self.gc1 = BatchGraphConvolutionLayer(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = BatchGraphConvolutionLayer(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = BatchGraphConvolutionLayer(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.conv1 = nn.Conv2d(1, 40, kernel_size=[64, 64])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 75), stride=10)

        self.lstm1 = nn.LSTM(input_size=1520, hidden_size=64, batch_first=True, bidirectional=True, num_layers=2, dropout=PARAMS['DROPOUT_P'])

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=2, batch_first=True)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, num_classes)

        self.identity = torch.eye(n_nodes).to(PARAMS['DEVICE'])
        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)
    def forward(self, x):
        A = torch.mm(self.node_embeddings, self.node_embeddings.T)
        A = A + self.identity

        out = F.relu(self.gc1(x, A))
        # out = self.dropout(out)
        # [N, n_nodes, out_features]
        out = F.relu(self.gc2(out, A))
        # out = self.dropout(out)
        
        out = out.unsqueeze(1)

        out = F.relu(self.conv1(out))

        out = self.maxpool1(out)

        out = self.flatten(out)

        out = out.view(out.shape[0], 1, -1)

        out, (h_T, c_T) = self.lstm1(out)
        out = out[:, -1, :]

        # out = out.unsqueeze(1)

        # out, attn_weights = self.attention(query=out, key=out, value=out)

        out = self.flatten(out)
        out = self.linear(out)

        return out


# class GCRAMAuto(nn.Module):
#     def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
#         super(GCRAMAuto, self).__init__()
#         self.gc1 = GraphConvolutionAuto(in_features, hidden_sizes[0], n_nodes)
#         self.gc2 = GraphConvolutionAuto(hidden_sizes[0], hidden_sizes[1], n_nodes)
#         self.gc3 = GraphConvolutionAuto(hidden_sizes[1], hidden_sizes[2], n_nodes)

#         self.conv1 = nn.Conv2d(1, 40, kernel_size=[64, 64])
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 75), stride=10)

#         self.lstm1 = nn.LSTM(input_size=1520, hidden_size=64, batch_first=True, bidirectional=True, num_layers=2, dropout=PARAMS['DROPOUT_P'])

#         self.dropout = nn.Dropout()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(128, num_classes)
#         self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=2, batch_first=True)

#         self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)
#     def forward(self, x):
#         A = torch.mm(self.node_embeddings, self.node_embeddings.T)
#         # A = F.dropout(A, 0.5)
#         A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

#         out = F.relu(self.gc1(x, A))
#         out = self.dropout(out)
#         # [N, n_nodes, out_features]
#         out = F.relu(self.gc2(out, A))
#         out = self.dropout(out)
#         out = out.unsqueeze(1)
#         print(out.shape)
#         out = F.relu(self.conv1(out))
#         print(out.shape)
#         out = self.maxpool1(out)
#         print(out.shape)
#         out = self.flatten(out)
#         print(out.shape)
#         out = out.reshape(out.shape[0], 1, -1)
#         print(out.shape)
#         out, (h_T, c_T) = self.lstm1(out)
#         out = out[:, -1, :]
#         print(out.shape)
#         out = out.unsqueeze(1)
#         print(out.shape)
#         out, attn_weights = self.attention(query=out, key=out, value=out)
#         print(out.shape)

#         out = self.flatten(out)
#         out = self.linear(out)

#         return out

class GCRAM(nn.Module):
    def __init__(self, seq_len, cnn_in_channels, cnn_n_kernels, cnn_kernel_size, cnn_stride, maxpool_kernel_size, maxpool_stride, lstm_hidden_size, is_bidirectional, lstm_n_layers, attn_embed_dim, n_classes):
        super(GCRAM, self).__init__()
        self.conv1 = nn.Conv2d(cnn_in_channels, cnn_n_kernels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.maxpool1 = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        cnn_output_size = (seq_len - cnn_kernel_size[1])//cnn_stride + 1
        maxpool_output_size = (cnn_output_size-maxpool_kernel_size[1])//maxpool_stride+1
        lstm_input_size = maxpool_output_size * cnn_in_channels * cnn_n_kernels
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=is_bidirectional, num_layers=lstm_n_layers)

        if is_bidirectional:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size*2, attention_size=attn_embed_dim, return_alphas=True)
        else:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size, attention_size=attn_embed_dim, return_alphas=True)
        
        self.flatten = nn.Flatten()

        if is_bidirectional:
            self.linear = nn.Linear(lstm_hidden_size*2, n_classes)
        else:
            self.linear = nn.Linear(lstm_hidden_size, n_classes)

        self.node_embeddings = torch.from_numpy(compute_adj_matrices('n')).to(PARAMS['DEVICE'])
    def forward(self, x):
        out = torch.einsum("ij,kjl->kil", self.node_embeddings, x)

        out = out.unsqueeze(1)

        out = F.relu(self.conv1(out))
        out = self.maxpool1(out)
        
        out = self.flatten(out)
        out = out.unsqueeze(1)

        out, (h_T, c_T) = self.lstm1(out)
        out = out[:, -1, :]

        out = out.unsqueeze(1)

        out, attn_weights = self.attention(out)

        out = self.flatten(out)
        out = self.linear(out)

        return out
class GATAuto(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GATAuto, self).__init__()
        self.gat1 = BatchGraphAttentionLayer(in_features, hidden_sizes[0], alpha=0.2, dropout=PARAMS['DROPOUT_P'])
        self.gat2 = BatchGraphAttentionLayer(hidden_sizes[0], hidden_sizes[1], alpha=0.2, dropout=PARAMS['DROPOUT_P'])
        self.gat3 = BatchGraphAttentionLayer(hidden_sizes[1], hidden_sizes[2], alpha=0.2, dropout=PARAMS['DROPOUT_P'])

        self.dropout = nn.Dropout(p=PARAMS['DROPOUT_P'])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.node_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)
    def forward(self, x):
        # A = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        A = torch.mm(self.node_embeddings, self.node_embeddings.T)
        # A = F.dropout(A, 0.5)
        A = A + torch.eye(A.shape[0]).to(PARAMS['DEVICE'])

        # x dim: [N, n_nodes, node_feat]
        # x_dim: [32, 64, 100]
        # print('Node embeddings:')
        # print(self.node_embeddings.shape)
        out = F.relu(self.gat1(x, A))
        out = self.dropout(out)
        # [N, n_nodes, out_features]
        out = F.relu(self.gat2(out, A))
        out = self.dropout(out)
        out = F.relu(self.gat3(out, A))
        out = self.dropout(out)

        # out = F.relu(self.gc1(x, A))
        # # [N, n_nodes, out_features]
        # out = F.relu(self.gc2(out, A))
        # out = F.relu(self.gc3(out, A))

        out = self.flatten(out)
        out = self.linear(out)

        return out