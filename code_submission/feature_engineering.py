import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
import torch
import copy
from torch_geometric.data import Data
from collections import namedtuple
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pprint
import time

from layer import LocalDegreeProfile, MatrixFactorization
from utils import norm_minmax, norm_z, norm_max, setx, timeit, label_distribution
from utils import Timer
from gfe import gbdt_gen, scale, degree_gen, DeepGL
from functools import partial
Numpy_Data = namedtuple('Data', 'x num_nodes train_ind y test_ind edge_index edge_weight train_mask test_mask')

class Feature_Engineering:
    def __init__(self, data, timer):
        self.timer = timer
        self.data = self.generate_data(data)
        self.num_nodes = self.data.x.shape[0]
        self.num_edges = self.data.edge_index.shape[0]
        self.unweighted = np.all(self.data.edge_weight == 1.0)
        print('nodes: {}, edges: {}, unweighted graph: {}'.format(self.num_nodes, self.num_edges, self.unweighted))

        has_feature = self.data.x.shape[1] > 2000
        self.data = setx(self.data, self.generate_feature(self.data))
        if not has_feature:
            self.data = setx(self.data, self.feature_selection(self.data))
        print('total feature: {}'.format(self.data.x.shape[1]))

    def generate_data(self, data):
        x = data['fea_table']
        node_index = x['node_index']
        x = x.drop('node_index', axis=1)
        num_nodes = x.shape[0]

        x = x.loc[:, x.max() != x.min()] #删除全相同列
        if x.shape[1] == 0:
            print("$$$$$$$$$$$$$$$$$$$$  No Feature")
            x = np.empty((num_nodes, 0))
        else:
            x = x.to_numpy(dtype=float)
            d1, d2 = x.shape
            # 删掉one-hot vector
            if d2 == d1 and np.allclose(x[:, :d1], np.eye(d1)):
                x = np.empty((num_nodes, 0))

        train_ind = data['train_label'][['node_index']].to_numpy().reshape(-1) # == data['train_indices']???
        train_y = data['train_label'][['label']].to_numpy().reshape(-1)
        test_ind = np.array(data['test_indices'])

        edge_index = data['edge_file'][['src_idx', 'dst_idx']].to_numpy()
        edge_weight = data['edge_file']['edge_weight'].to_numpy()

        ### delete edges with zero weight
        ind = np.where(edge_weight != 0.0)[0]
        edge_index = edge_index[ind]
        edge_weight = edge_weight[ind]
        
        train_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[train_ind] = True
        y = np.zeros(num_nodes, dtype=int)
        y[train_ind] = train_y
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[test_ind] = True
        return Numpy_Data(x=x, num_nodes=num_nodes, train_ind=train_ind, y=y, test_ind=test_ind, edge_index=edge_index, edge_weight=edge_weight, train_mask=train_mask, test_mask=test_mask)

    @timeit
    def generate_feature(self, data):
        x = [
                gbdt_gen(data, fixlen=2000, early_stopping_rounds=10, num_boost_round=100),
                #data.x,
                scale(degree_gen(data))
                ]

        if self.num_edges <= 1000000:
            x.append(norm_z(self._get_mf_feature(data)))
        size = [i.shape[1] for i in x]
        print("feature generation: {}=({})".format(sum(size), '+'.join(map(str, size))))
        return np.hstack(x)

    @timeit
    def _get_mf_feature(self, data, size=32):
        adj = csr_matrix((data.edge_weight, (data.edge_index[:, 0], data.edge_index[:,1])), shape = (self.num_nodes,self.num_nodes))
        if np.max(adj - adj.T) > 1e-5:
            adj = adj + adj.T 
        mf = MatrixFactorization()
        return mf.forward(adj, size)

    @timeit
    def feature_selection(self, data, K=1000, seclection_type='dgl'):
        if seclection_type == 'f_classif':
            tx = data.x
            x = data.x[data.train_ind]
            ind = x.max(axis=0) != x.min(axis=0)
            x = x[:, ind]
            tx = tx[:, ind]
            y = data.y[data.train_ind]
            k = min(K, x.shape[1])

            #sk = SelectKBest(chi2, k=k)
            #sk = SelectKBest(mutual_info_classif, k=k)
            sk = SelectKBest(f_classif, k=k)
            res = sk.fit(x, y).get_support(indices=True)
            rx = tx[:, res]
            return tx[:, res]
        elif seclection_type == 'gbdt':
            rx = gbdt_gen(data, fixlen=K)
        elif seclection_type == 'dgl':
            K = 200
            dgl = DeepGL(data)
            remain_time = self.timer.remain_time()
            rx = dgl.gen(max_epoch=5, fixlen=K, y_sel_func=partial(gbdt_gen,num_boost_round=50,early_stopping_rounds=10,is_val=True),timebudget=remain_time/3)
            ### add
            data=setx(data,rx)
            rx=gbdt_gen(data,fixlen=2000)
            ### 
        print('after selection, feature: {}->{}'.format(data.x.shape[1], rx.shape[1]))
        return rx

    def generate_pyg_data(self, data=None):
        if data is None:
            data = self.data
        x = torch.tensor(data.x, dtype=torch.float)

        edge_index = sorted(data.edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(data.edge_index, dtype=torch.long).transpose(0, 1)
        edge_weight = torch.tensor(data.edge_weight, dtype=torch.float32)

        num_nodes = self.num_nodes

        pyg_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pyg_data.num_nodes = num_nodes

        pyg_data.train_mask = torch.BoolTensor(data.train_mask)
        pyg_data.y = torch.LongTensor(data.y)
        pyg_data.test_mask = torch.BoolTensor(data.test_mask)
        return pyg_data

    def get_data(self, model_name):
        if model_name in ['AutoGCN', 'AutoGAT', 'AutoSAGE', 'AutoGIN']:
            return self.generate_pyg_data()
        else:
            return self.data

    def split(self, n_splits, test_size=0.2, return_tensor=True):
        """
        y = self.data.y[self.data.train_ind]
        for _ in range(n_splits):
            train_ind, val_ind, y_train, y_val = train_test_split(self.data.train_ind, y, test_size=test_size, shuffle=True, stratify=y)
            train_mask = np.zeros(self.num_nodes, dtype=bool)
            train_mask[train_ind] = True
            val_mask = np.zeros(self.num_nodes, dtype=bool)
            val_mask[val_ind] = True
            if return_tensor:
                train_mask = torch.BoolTensor(train_mask)
                val_mask = torch.BoolTensor(val_mask)
            yield train_mask, val_mask


        """
        ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        for train_index, test_index in ss.split(self.data.train_ind, self.data.y[self.data.train_ind]):
            train_ind = self.data.train_ind[train_index]
            val_ind = self.data.train_ind[test_index]
            #pprint.pprint(label_distribution(self.data.y[train_ind]), width=1)
            #pprint.pprint(label_distribution(self.data.y[val_ind]), width=1)
            #y_test = np.loadtxt('data/e/test_label.tsv', dtype=int, skiprows=1)[:, 1]
            #pprint.pprint(label_distribution(y_test), width=1)
            train_mask = np.zeros(self.num_nodes, dtype=bool)
            train_mask[train_ind] = True
            val_mask = np.zeros(self.num_nodes, dtype=bool)
            val_mask[val_ind] = True
            if return_tensor:
                train_mask = torch.BoolTensor(train_mask)
                val_mask = torch.BoolTensor(val_mask)
            yield train_mask, val_mask
        return 0
