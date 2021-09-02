from layers_batchwise_2 import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCN(nn.Module):
    def __init__(self, in_features, num_classes, n_nodes, hidden_sizes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2]*n_nodes, num_classes)
        self.flatten = nn.Flatten()
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = self.flatten(out)
        out = self.fc4(out)

        return out