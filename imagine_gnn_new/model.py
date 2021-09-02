from layers_batchwise_2 import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import compute_adj_matrices

class GCN(nn.Module):
    def __init__(self, in_features, n_nodes, num_classes, hidden_sizes):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_sizes[0], n_nodes)
        self.gc2 = GraphConvolution(hidden_sizes[0], hidden_sizes[1], n_nodes)
        self.gc3 = GraphConvolution(hidden_sizes[1], hidden_sizes[2], n_nodes)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)

        # self.identity = torch.eye(n_nodes).to(device)
        self.node_embeddings = torch.from_numpy(compute_adj_matrices('n') + np.eye(64, dtype=np.float32)).to(device)
        # print(self.node_embeddings)
    def forward(self, x):
        # A = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # A = torch.add(A, self.identity)
        out = F.relu(self.gc1(x, self.node_embeddings))
        out = F.relu(self.gc2(out, self.node_embeddings))
        out = F.relu(self.gc3(out, self.node_embeddings))

        out = self.flatten(out)
        out = self.linear(out)

        return out