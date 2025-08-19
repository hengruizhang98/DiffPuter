from networkx.readwrite import adjlist
from numpy.core.numeric import roll
import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
import pdb
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F

def get_adj_matrix(df_X, edge_index):
    row = edge_index[0]
    col = edge_index[1]
    adj_mat = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(df_X.shape[0]+df_X.shape[1], df_X.shape[0]+df_X.shape[1]))
    return adj_mat

def get_obob_adj_matrix(df_X, edge_index):
    row = edge_index[0]
    col = edge_index[1]
    adj_mat = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(df_X.shape[0], df_X.shape[0]))
    return adj_mat

def normalize_adj_(adj):
    adj_ = sp.coo_matrix(adj)
    adj_.setdiag(1)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_norm_tuple = sparse_to_tuple(adj_norm)
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_tuple[0].T),
                                                torch.FloatTensor(adj_norm_tuple[1]),
                                                torch.Size(adj_norm_tuple[2]))
    return adj_norm

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx