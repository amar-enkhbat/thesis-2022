from layers_batchwise_2 import GraphConvolution
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, input_height, input_width, seq_len, n_kernels, hidden_size_1, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kernels, kernel_size=(input_height, input_height))

        self.fc1 = nn.Linear(n_kernels * (seq_len - input_width + 1), hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, n_classes)
        self.flatten = nn.Flatten()
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out