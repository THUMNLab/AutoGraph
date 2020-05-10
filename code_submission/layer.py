import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv, Node2Vec, SGConv, SAGEConv
from torch.nn import BatchNorm1d
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.sparse as ssp
import networkx as nx

class GCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, hidden2=16, dropout=0.5, features_num=16, num_class=2, agg='concat', withbn=True, act='leaky_relu', **args):
        super(GCN, self).__init__()
        self.args = args
        self.dropout = dropout
        self.agg = agg
        self.withbn = withbn
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        if self.withbn:
            self.bn1 = BatchNorm1d(hidden)
            self.bns = torch.nn.ModuleList()
        hd = [hidden]
        for i in range(num_layers - 1):
            hd.append(hidden2)
            self.convs.append(GCNConv(hidden, hidden2))
            self.bns.append(BatchNorm1d(hidden2))
        if agg == 'concat':
            outdim = sum(hd)
        elif agg == 'self':
            outdim = hd[-1]
        if act == 'leaky_relu':
            self.act = F.leaky_relu
        elif act == 'tanh':
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        self.lin2 = Linear(outdim, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.leaky_relu(self.first_lin(x))
        if self.withbn:
            x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs = [x]
        for conv, bn in zip(self.convs, self.bns):
            x = self.act(conv(x, edge_index, edge_weight=edge_weight))
            if self.withbn:
                x = bn(x)
            xs.append(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        if self.agg == 'concat':
            x = torch.cat(xs, dim=1)
        elif self.agg == 'self':
            x = xs[-1]
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def train_predict(self, data, train_mask, val_mask=None, return_train=False, debug=False):
        """
            hyperparams:
                lr
                weight_decay
                epoches
        """
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        for epoch in range(1, self.args['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        test_mask = data.test_mask if val_mask is None else val_mask
        self.eval()
        res = self.forward(data)
        with torch.no_grad():
            pred = res[test_mask]
            if return_train:
                pred = (pred, res[train_mask])
        return pred
 
    def __repr__(self):
        return self.__class__.__name__


class GAT(torch.nn.Module):
    def __init__(self, features_num=16, num_class=2, dropout=0.5, **args):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(features_num, 8, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(8*8, num_class, dropout=self.dropout)

    def forward(self, data):
        x = data.x
        #x = F.dropout(data.x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=-1)

    def train_predict(self, data, train_mask, val_mask=None, return_train=False, **hyperparams):
        """
            hyperparams:
                lr
                weight_decay
                epoches
        """
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
        for epoch in range(1, hyperparams['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        test_mask = data.test_mask if val_mask is None else val_mask
        self.eval()
        res = self.forward(data)
        with torch.no_grad():
            pred = res[test_mask]
            if return_train:
                pred = (pred, res[train_mask])
        return pred
    def __repr__(self):
        return self.__class__.__name__

class LocalDegreeProfile(object):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """

    def __call__(self, data, norm=True):
        row, col = data.edge_index
        N = data.num_nodes

        deg = degree(row, N, dtype=torch.float)
        if norm:
            deg = deg / deg.max()
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class MatrixFactorization(object):
    def __init__(self):
        pass

    def normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt = ssp.diags(d_inv_sqrt)
        return adj.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)

    def forward(self, adj, d, use_eigenvalues = 0, adj_norm = 1):
        G = nx.from_scipy_sparse_matrix(adj)
        comp = list(nx.connected_components(G))
        results = np.zeros((adj.shape[0],d))
        #print(len(comp))
        #print(comp)
        for i in range(len(comp)):
            node_index = np.array(list(comp[i]))
            d_temp = min(len(node_index) - 2, d)
            if d_temp <= 0:
                continue
            temp_adj = adj[node_index,:][:,node_index].asfptype()
            if adj_norm == 1:
                temp_adj = self.normalize_adj(temp_adj)
            lamb, X = ssp.linalg.eigs(temp_adj, d_temp)
            lamb, X = lamb.real, X.real
            temp_order = np.argsort(lamb)
            lamb, X = lamb[temp_order], X[:,temp_order]
            for i in range(X.shape[1]):
                if np.sum(X[:,i]) < 0:
                    X[:,i] = -X[:,i]
            if use_eigenvalues == 1:
                X = X.dot(np.diag(np.sqrt(np.absolute(lamb))))
            elif use_eigenvalues == 2:
                X = X.dot(np.diag(lamb))
            results[node_index,:d_temp] = X
        return results


