import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv, Node2Vec, SGConv, SAGEConv, GINConv
from torch.nn import BatchNorm1d
import time
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.sparse as ssp
import pprint
import networkx as nx

from utils import set_default

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = set_default(args, {
                    'num_layers': 2,
                    'hidden': 64,
                    'hidden2': 32,
                    'dropout': 0.5,
                    'lr': 0.005,
                    'epoches': 300,
                    'weight_decay': 5e-4,
                    'agg': 'concat',
                    'act': 'leaky_relu',
                    'withbn': True,
                        })
        self.timer = self.args['timer']
        self.dropout = self.args['dropout']
        self.agg = self.args['agg']
        self.withbn = self.args['withbn']
        self.conv1 = GCNConv(self.args['hidden'], self.args['hidden'])
        self.convs = torch.nn.ModuleList()
        if self.withbn:
            self.bn1 = BatchNorm1d(self.args['hidden'])
            self.bns = torch.nn.ModuleList()
        hd = [self.args['hidden']]
        for i in range(self.args['num_layers'] - 1):
            hd.append(self.args['hidden2'])
            self.convs.append(GCNConv(self.args['hidden'], self.args['hidden2']))
            self.bns.append(BatchNorm1d(self.args['hidden2']))
        if self.args['agg'] == 'concat':
            outdim = sum(hd)
        elif self.args['agg'] == 'self':
            outdim = hd[-1]
        if self.args['act'] == 'leaky_relu':
            self.act = F.leaky_relu
        elif self.args['act'] == 'tanh':
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        self.lin2 = Linear(outdim, self.args['num_class'])
        self.first_lin = Linear(self.args['features_num'], self.args['hidden'])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        #x = self.act(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.leaky_relu(self.first_lin(x))
        if self.withbn:
            x = self.bn1(x)
        #x = self.act(self.conv1(x, edge_index, edge_weight=edge_weight))
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

    def train_predict(self, data, train_mask=None, val_mask=None, return_out=True):
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        flag_end = False
        st = time.time()
        for epoch in range(1, self.args['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                cost = (time.time()-st)/epoch*50
                if max(cost*10, 5) > self.timer.remain_time():
                    flag_end = True
                    break

        test_mask = data.test_mask
        self.eval()
        with torch.no_grad():
            res = self.forward(data)
            if return_out:
                pred = res
            else:
                pred = res[test_mask]
            if val_mask is not None:
                return pred, res[val_mask], flag_end
        return pred, flag_end
 
    def __repr__(self):
        return self.__class__.__name__

class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = set_default(args, {
                    'hidden': 64,
                    'hidden2': 32,
                    'dropout': 0.5,
                    'lr': 0.005,
                    'epoches': 300,
                    'weight_decay': 5e-4,
                    'agg': 'self',
                    'act': 'leaky_relu',
                    'withbn': True,
                        })
        self.timer = self.args['timer']
        self.dropout = self.args['dropout']
        self.agg = self.args['agg']
        self.withbn = self.args['withbn']
        self.conv1 = GATConv(self.args['hidden'], self.args['hidden'], self.args['heads'], dropout=self.args['dropout'])
        self.conv2 = GATConv(self.args['hidden']*self.args['heads'], self.args['hidden2'], dropout=self.args['dropout'])
        hd = [self.args['hidden'], self.args['hidden']*self.args['heads'], self.args['hidden2']]
        if self.withbn:
            self.bn1 = BatchNorm1d(self.args['hidden']*self.args['heads'])
            self.bn2 = BatchNorm1d(self.args['hidden2'])
        if self.args['agg'] == 'concat':
            outdim = sum(hd)
        elif self.args['agg'] == 'self':
            outdim = hd[-1]
        if self.args['act'] == 'leaky_relu':
            self.act = F.leaky_relu
        elif self.args['act'] == 'tanh':
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        self.lin2 = Linear(outdim, self.args['num_class'])
        self.first_lin = Linear(self.args['features_num'], self.args['hidden'])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.act(self.first_lin(x))
        xs = [x]
        x = self.act(self.conv1(x, edge_index))
        if self.withbn:
            x = self.bn1(x)
        xs.append(x)
        x = self.act(self.conv2(x, edge_index))
        if self.withbn:
            x = self.bn2(x)
        xs.append(x)
        if self.agg == 'concat':
            x = torch.cat(xs, dim=1)
        elif self.agg == 'self':
            x = xs[-1]
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def train_predict(self, data, train_mask=None, val_mask=None, return_out=True):
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        flag_end = False
        st = time.time()
        for epoch in range(1, self.args['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                cost = (time.time()-st)/epoch*50
                if max(cost*10, 5) > self.timer.remain_time():
                    flag_end = True
                    break

        test_mask = data.test_mask
        self.eval()
        with torch.no_grad():
            res = self.forward(data)
            if return_out:
                pred = res
            else:
                pred = res[test_mask]
            if val_mask is not None:
                return pred, res[val_mask], flag_end
        return pred, flag_end
 
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

class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.args = set_default(args, {
                    'num_layers': 2,
                    'hidden': 64,
                    'hidden2': 32,
                    'dropout': 0.5,
                    'lr': 0.005,
                    'epoches': 300,
                    'weight_decay': 5e-4,
                    'act': 'leaky_relu',
                    'withbn': True,
                        })
        self.timer = self.args['timer']
        self.dropout = self.args['dropout']
        self.agg = self.args['agg']
        self.withbn = self.args['withbn']
        self.conv1 = SAGEConv(self.args['hidden'], self.args['hidden'])
        self.convs = torch.nn.ModuleList()
        if self.withbn:
            self.bn1 = BatchNorm1d(self.args['hidden'])
            self.bns = torch.nn.ModuleList()
        hd = [self.args['hidden'], self.args['hidden']]
        for i in range(self.args['num_layers'] - 1):
            hd.append(self.args['hidden2'])
            self.convs.append(SAGEConv(self.args['hidden'], self.args['hidden2']))
            self.bns.append(BatchNorm1d(self.args['hidden2']))
        if self.args['agg'] == 'concat':
            outdim = sum(hd)
        elif self.args['agg'] == 'self':
            outdim = hd[-1]
        if self.args['act'] == 'leaky_relu':
            self.act = F.leaky_relu
        elif self.args['act'] == 'tanh':
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        self.lin2 = Linear(outdim, self.args['num_class'])
        self.first_lin = Linear(self.args['features_num'], self.args['hidden'])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.act(self.first_lin(x))
        xs = [x]
        x = self.act(self.conv1(x, edge_index, edge_weight=edge_weight))
        if self.withbn:
            x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
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

    def train_predict(self, data, train_mask=None, val_mask=None, return_out=True):
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        flag_end = False
        st = time.time()
        for epoch in range(1, self.args['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                cost = (time.time()-st)/epoch*50
                if max(cost*10, 5) > self.timer.remain_time():
                    flag_end = True
                    break

        test_mask = data.test_mask
        self.eval()
        with torch.no_grad():
            res = self.forward(data)
            if return_out:
                pred = res
            else:
                pred = res[test_mask]
            if val_mask is not None:
                return pred, res[val_mask], flag_end
        return pred, flag_end
 
    def __repr__(self):
        return self.__class__.__name__

class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.args = set_default(args, {
                    'num_layers': 2,
                    'hidden': 64,
                    'hidden2': 32,
                    'dropout': 0.5,
                    'lr': 0.005,
                    'epoches': 300,
                    'weight_decay': 5e-4,
                    'act': 'leaky_relu',
                    'withbn': True,
                        })
        self.timer = self.args['timer']
        self.dropout = self.args['dropout']
        self.agg = self.args['agg']
        self.conv1 = GINConv(
                        Sequential(
                            Linear(self.args['features_num'], self.args['hidden']),
                            ReLU(),
                            BatchNorm1d(self.args['hidden']),
                        ),
                        train_eps=True)
        self.convs = torch.nn.ModuleList()
        hd = [self.args['hidden']]
        for i in range(self.args['num_layers'] - 1):
            hd.append(self.args['hidden2'])
            self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(self.args['hidden'], self.args['hidden2']),
                            ReLU(),
                            BatchNorm1d(self.args['hidden2']),
                        ),
                        train_eps=True))
        if self.args['agg'] == 'concat':
            outdim = sum(hd)
        elif self.args['agg'] == 'self':
            outdim = hd[-1]
        if self.args['act'] == 'leaky_relu':
            self.act = F.leaky_relu
        elif self.args['act'] == 'tanh':
            self.act = torch.tanh
        else:
            self.act = lambda x: x
        self.lin2 = Linear(outdim, self.args['num_class'])

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.act(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
            xs.append(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)
        if self.agg == 'concat':
            x = torch.cat(xs, dim=1)
        elif self.agg == 'self':
            x = xs[-1]
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def train_predict(self, data, train_mask=None, val_mask=None, return_out=True):
        if train_mask is None:
            train_mask = data.train_mask
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        flag_end = False
        st = time.time()
        for epoch in range(1, self.args['epoches']):
            self.train()
            optimizer.zero_grad()
            res = self.forward(data)
            loss = F.nll_loss(res[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                cost = (time.time()-st)/epoch*50
                if max(cost*10, 5) > self.timer.remain_time():
                    flag_end = True
                    break

        test_mask = data.test_mask
        self.eval()
        with torch.no_grad():
            res = self.forward(data)
            if return_out:
                pred = res
            else:
                pred = res[test_mask]
            if val_mask is not None:
                return pred, res[val_mask], flag_end
        return pred, flag_end
 
    def __repr__(self):
        return self.__class__.__name__
