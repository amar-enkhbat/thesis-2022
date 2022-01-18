from layers import BatchGraphConvolutionLayer, BatchGraphAttentionLayer, SelfAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils import compute_adj_matrices

import math

class FCN(nn.Module):
    def __init__(self, in_features, num_classes, n_nodes, hidden_sizes, dropout_p):
        super(FCN, self).__init__()
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.fc3(out))
        out = F.dropout(out, p=self.dropout_p)

        out = self.flatten(out)
        out = self.fc4(out)

        return out

class CNN(nn.Module):
    def __init__(self, kernel_size, seq_len, n_kernels, hidden_size, n_classes, dropout_p):
        super(CNN, self).__init__()

        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=kernel_size)

        self.fc1 = nn.Linear(n_kernels * (seq_len - kernel_size[1] + 1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.flatten(out)
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=self.dropout_p)
        
        out = self.flatten(out)
        out = self.fc2(out)

        return out

class RNN(nn.Module):
    def __init__(self, input_size, n_layers, hidden_size, n_classes, dropout_p):
        super(RNN, self).__init__()

        self.dropout_p = dropout_p

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

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
    def __init__(self, graph_type, in_features, n_nodes, num_classes, hidden_sizes, dropout_p, device):
        super(GCN, self).__init__()

        self.dropout_p = dropout_p

        self.gc1 = BatchGraphConvolutionLayer(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = BatchGraphConvolutionLayer(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = BatchGraphConvolutionLayer(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.adj = torch.from_numpy(compute_adj_matrices(graph_type)).to(device)

    def forward(self, x):
        out = F.relu(self.gc1(x, self.adj))
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.gc2(out, self.adj))
        out = F.dropout(out, p=self.dropout_p)
        
        out = F.relu(self.gc3(out, self.adj))
        out = F.dropout(out, p=self.dropout_p)

        out = self.flatten(out)
        out = self.linear(out)

        return out

class GCNAuto(nn.Module):
    def __init__(self, kernel_type, in_features, n_nodes, num_classes, hidden_sizes, dropout_p, device):
        super(GCNAuto, self).__init__()

        self.kernel_type = kernel_type
        self.dropout_p = dropout_p
        

        self.gc1 = BatchGraphConvolutionLayer(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = BatchGraphConvolutionLayer(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = BatchGraphConvolutionLayer(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.eye = torch.eye(n_nodes, device=device)
        self.adj = nn.Parameter(torch.randn(n_nodes, n_nodes))

    def forward(self, x):
        if self.kernel_type == 'a':
            adj = self.kernel_a()
        if self.kernel_type == 'b':
            adj = self.kernel_b()
        if self.kernel_type == 'c':
            adj = self.kernel_c()
        if self.kernel_type == 'd':
            adj = self.kernel_d()

        out = F.relu(self.gc1(x, adj))
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.gc2(out, adj))
        out = F.dropout(out, p=self.dropout_p)

        out = F.relu(self.gc3(out, adj))
        out = F.dropout(out, p=self.dropout_p)

        out = self.flatten(out)
        out = self.linear(out)

        return out

    def kernel_a(self):
        return self.adj

    def kernel_b(self):
        return torch.mm(self.adj, self.adj.T)

    def kernel_c(self):
        return torch.mm(self.adj, self.adj.T) + self.eye

    def kernel_d(self):
        return F.softmax(torch.mm(self.adj, self.adj.T) + self.eye) 
    
    def init_node_embeddings(self):
        stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        self.adj.data.fill_diagonal_(1)
        
class GCRAM(nn.Module):
    def __init__(self, graph_type, seq_len, cnn_in_channels, cnn_n_kernels, cnn_kernel_size, cnn_stride, maxpool_kernel_size, maxpool_stride, lstm_hidden_size, is_bidirectional, lstm_n_layers, attn_embed_dim, n_classes, lstm_dropout_p, dropout1_p, dropout2_p, device):
        super(GCRAM, self).__init__()

        self.dropout1_p = dropout1_p
        self.dropout2_p = dropout2_p

        self.conv1 = nn.Conv2d(cnn_in_channels, cnn_n_kernels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.maxpool1 = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        cnn_output_size = (seq_len - cnn_kernel_size[1])//cnn_stride + 1
        maxpool_output_size = (cnn_output_size-maxpool_kernel_size[1])//maxpool_stride+1
        lstm_input_size = maxpool_output_size * cnn_in_channels * cnn_n_kernels
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=is_bidirectional, num_layers=lstm_n_layers, dropout=lstm_dropout_p)

        if is_bidirectional:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size*2, attention_size=attn_embed_dim, return_alphas=True)
        else:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size, attention_size=attn_embed_dim, return_alphas=True)
        
        self.flatten = nn.Flatten()

        if is_bidirectional:
            self.linear = nn.Linear(lstm_hidden_size*2, n_classes)
        else:
            self.linear = nn.Linear(lstm_hidden_size, n_classes)

        self.adj = torch.from_numpy(compute_adj_matrices(graph_type)).to(device)

    def forward(self, x):
        out = torch.einsum("ij,kjl->kil", self.adj, x)

        out = out.unsqueeze(1)

        out = F.relu(self.conv1(out))
        out = self.maxpool1(out)

        out = self.flatten(out)
        out = out.unsqueeze(1)
        out = F.dropout(out, p=self.dropout1_p)

        out, (h_T, c_T) = self.lstm1(out)
        out = out[:, -1, :]

        out = out.unsqueeze(1)

        out, attn_weights = self.attention(out)
        out = F.dropout(out, p=self.dropout2_p)

        out = self.flatten(out)
        out = self.linear(out)

        return out

class GCRAMAuto(nn.Module):
    def __init__(self, seq_len, cnn_in_channels, cnn_n_kernels, cnn_kernel_size, cnn_stride, maxpool_kernel_size, maxpool_stride, lstm_hidden_size, is_bidirectional, lstm_n_layers, attn_embed_dim, n_classes, lstm_dropout_p, dropout1_p, dropout2_p, device):
        super(GCRAMAuto, self).__init__()

        self.dropout1_p = dropout1_p
        self.dropout2_p = dropout2_p

        self.conv1 = nn.Conv2d(cnn_in_channels, cnn_n_kernels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.maxpool1 = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        cnn_output_size = (seq_len - cnn_kernel_size[1])//cnn_stride + 1
        maxpool_output_size = (cnn_output_size-maxpool_kernel_size[1])//maxpool_stride+1
        lstm_input_size = maxpool_output_size * cnn_in_channels * cnn_n_kernels
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=is_bidirectional, num_layers=lstm_n_layers, dropout=lstm_dropout_p)

        if is_bidirectional:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size*2, attention_size=attn_embed_dim, return_alphas=True)
        else:
            self.attention = SelfAttentionLayer(hidden_size=lstm_hidden_size, attention_size=attn_embed_dim, return_alphas=True)
        
        self.flatten = nn.Flatten()

        if is_bidirectional:
            self.linear = nn.Linear(lstm_hidden_size*2, n_classes)
        else:
            self.linear = nn.Linear(lstm_hidden_size, n_classes)

        self.adj = nn.Parameter(torch.randn(64, 64), requires_grad=True)
        
    def forward(self, x):
        out = torch.einsum("ij,kjl->kil", self.adj, x)

        out = out.unsqueeze(1)

        out = F.relu(self.conv1(out))
        out = self.maxpool1(out)
        
        out = self.flatten(out)
        out = out.unsqueeze(1)
        out = F.dropout(out, p=self.dropout1_p)

        out, (h_T, c_T) = self.lstm1(out)
        out = out[:, -1, :]

        out = out.unsqueeze(1)

        out, attn_weights = self.attention(out)
        out = F.dropout(out, p=self.dropout2_p)

        out = self.flatten(out)
        out = self.linear(out)

        return out

    def init_node_embeddings(self):
        stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        self.adj.data.fill_diagonal_(1)

class GATAuto(nn.Module):
    """DOES NOT WORK!!!"""
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes, dropout_p):
        super(GATAuto, self).__init__()

        self.dropout_p = dropout_p

        self.gat1 = BatchGraphAttentionLayer(in_features, hidden_sizes[0], alpha=0.2, dropout=dropout_p)
        self.gat2 = BatchGraphAttentionLayer(hidden_sizes[0], hidden_sizes[1], alpha=0.2, dropout=dropout_p)
        self.gat3 = BatchGraphAttentionLayer(hidden_sizes[1], hidden_sizes[2], alpha=0.2, dropout=dropout_p)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        self.adj = nn.Parameter(torch.randn(n_nodes, n_nodes))
    def forward(self, x):
        out = F.relu(self.gat1(x, self.adj))
        out = F.relu(self.gat2(out, self.adj))
        out = F.relu(self.gat3(out, self.adj))

        out = self.flatten(out)
        out = self.linear(out)

        return out

    def init_node_embeddings(self):
        stdv = 1. / math.sqrt(self.adj.size(1))
        self.adj.data.uniform_(-stdv, stdv)
        self.adj.data.fill_diagonal_(1)