import math

import torch

from torch import nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    """
    Original GC Layer from: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, hidden_size, attention_size, return_alphas=False):
        super(SelfAttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.return_alphas = return_alphas

        self.w_omega = nn.Parameter(torch.empty(hidden_size, attention_size).normal_(mean=0.0, std=0.1))
        self.b_omega = nn.Parameter(torch.empty(attention_size).normal_(mean=0.0, std=0.1))
        self.u_omega = nn.Parameter(torch.empty(attention_size).normal_(mean=0.0, std=0.1))

    def forward(self, inputs):
        v = torch.tanh(torch.tensordot(inputs, self.w_omega, dims=1) + self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alphas = F.softmax(vu, dim=1)
        output = torch.sum(inputs * alphas.unsqueeze(-1), 1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.hidden_size) + ' -> ' \
               + str(self.attention_size) + ')'



class GraphConvolutionLayer(nn.Module):
    """
    Original GC Layer from: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchGraphConvolutionLayer(nn.Module):
    """
    GCN Layer for batch of graphs. 
    """
    def __init__(self, in_features, out_features, n_channels, bias=True):
        super(BatchGraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input [N, n_nodes, node_features]
        # adj [n_nodes, n_nodes]
        # weights [in_features, out_features]
        support = torch.einsum("ijk,kl->ijl", input, self.weight)
        # [N, n_nodes, node_features] x [in_features, out_features] = [N, n_nodes, out_features]
        output = torch.einsum("ij,kjl->kil", adj, support)
        # [n_nodes, n_nodes][N, n_nodes, out_features] = [N, n_nodes, out_features]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class BatchGraphAttentionLayer(nn.Module):
    """
    Batchwise
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(BatchGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # print('h shape:', h.shape)
        # print('weight shape:', self.W.shape)
        Wh = torch.einsum("ijk,kl->ijl", h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print('Wh shape:', Wh.shape)
        e = self._prepare_attentional_mechanism_input(Wh)
        # print('e shape:', e.shape)

        zero_vec = -9e15*torch.ones_like(e)
        # print('zero vec shape:', zero_vec.shape)
        
        adj_batch = torch.cat([adj for i in range(32)], axis=0).view(32, 64, 64)
        # print('adj_batch shape:', adj_batch.shape)
        attention = torch.where(adj_batch > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # print('wh shape:', Wh.shape)
        # print('self a shape:', self.a[:self.out_features, :].shape)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # print('Wh1 shape:', Wh1.shape)
        
        # print('wh shape:', Wh.shape)
        # print('self a shape:', self.a[self.out_features:, :].shape)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # print('Wh2 shape:', Wh2.shape)
        # print('Wh2.T shape:', Wh2.T.shape)
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'