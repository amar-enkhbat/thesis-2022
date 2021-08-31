import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batch_size, n_channels, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('input shape:', input.shape)
        batch_size = input.shape[0]
        n_channels = input.shape[1]
        # print('weight shape:', self.weight.shape)
        # print('')
        input = input.reshape(batch_size*n_channels, self.in_features)
        # print('input shape:', input.shape)
        # print('weight shape:', self.weight.shape)
        # print('')
        support = torch.mm(input, self.weight)
        # print('adj shape:', adj.shape)
        # print('support shape:', support.shape)
        output = torch.mm(adj, support)
        # print('output shape:', output.shape)
        output = output.reshape(batch_size, n_channels, self.out_features)
        # print('output shape:', output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
