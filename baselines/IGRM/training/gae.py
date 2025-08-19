from builtins import print
from numpy.core.numeric import NaN
import torch
from torch._C import get_num_interop_threads
import torch.nn as nn
import torch.nn.functional as F
import pyro
import scipy.sparse as sp
import numpy as np
import math
from sklearn import preprocessing
import time

import pdb

# New VGAE
class VGAE(nn.Module):
    """ GAE/VGAE as edge prediction model """
    def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
        super(VGAE, self).__init__()
        self.gae = gae
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False))
        self.layers.append(GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False))

    def forward(self, adj, features):
        # GCN encoder
        hidden = self.layers[0](adj, features)
        self.mean = self.layers[1](adj, hidden)
        if self.gae:
            # GAE (no sampling at bottleneck)
            Z = self.mean
        else:
            # VGAE
            self.logstd = self.layers[2](adj, hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
            Z = sampled_Z
        # inner product decoder
        #adj_logits = Z @ Z.T
        #return adj_logits

        return Z

# class VGAE(nn.Module):
#     """ GAE/VGAE as edge prediction model """
#     def __init__(self, dim_feats, dim_h, dim_z, activation, gae=False):
#         super(VGAE, self).__init__()
#         self.gae = gae
#         self.layers = nn.ModuleList()
#         self.layers.append(GCNLayer(dim_feats, dim_h, 1, None, 0, bias=False))
#         self.layers.append(GCNLayer(dim_h, dim_z, 1, activation, 0, bias=False))

#     def forward(self, adj, features):
#         # GCN encoder
#         hidden = self.layers[0](adj, features)
#         self.mean = self.layers[1](adj, hidden)
#         if self.gae:
#             # GAE (no sampling at bottleneck)
#             Z = self.mean
#         else:
#             # VGAE
#             self.logstd = self.layers[2](adj, hidden)
#             gaussian_noise = torch.randn_like(self.mean)
#             sampled_Z = gaussian_noise*torch.exp(self.logstd) + self.mean
#             Z = sampled_Z
#         # inner product decoder
#         adj_logits = Z @ Z.T
#         return adj_logits

class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


from sklearn.metrics.pairwise import cosine_similarity
def sample_adj(adj_logits):
    """ sample an adj from the predicted edge probabilities of ep_net """
    relu = torch.nn.ReLU()
    adj_logits = relu(adj_logits)
    adj_logits_ = (adj_logits / torch.max(adj_logits)) / 20
    # sampling
    adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=0.2, probs=adj_logits_).rsample()
    # making adj_sampled symmetric
    adj_sampled = adj_sampled.triu(1)
    return adj_sampled, adj_logits_
    #return adj_sampled, adj_sampled + adj_sampled.T, adj_logits_
