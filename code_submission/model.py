"""the simple baseline for autograph"""

import numpy as np
import pandas as pd
import os, time
import lightgbm as lgb

import torch

from feature_engineering import Feature_Engineering
from automl import AutoGCN, AutoGBDT, AutoGAT, AutoGBM

import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)


class Model:

    def __init__(self):
        self.start_time = time.time()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #select model : ['AutoGCN', 'AutoGBDT']
        #self.model_name = 'AutoGCN'
        #self.model_name = 'AutoGAT'
        #self.model_name = 'AutoGBDT'
        #self.model_name = 'AutoGBM'
        self.model_name = 'Ensemble'
    def generate_pyg_data(self):
        data = self.fe.pyg_data
        return data

    def train_predict(self, data, time_budget,n_class,schema):
        self.time_budget = time_budget
        self.n_class = n_class
        self.schema = schema
        self.fe = Feature_Engineering(data, self.start_time, time_budget)

        if self.model_name == 'AutoGCN':
            data = self.fe.generate_pyg_data()
            model = AutoGCN(data, self.device, self.start_time, time_budget, self.n_class, self.fe.data.train_ind)
        elif self.model_name == 'AutoGAT':
            data = self.fe.generate_pyg_data()
            model = AutoGAT(data, self.device, self.start_time, time_budget, self.n_class, self.fe.data.train_ind)
        elif self.model_name == 'AutoGBDT':
            data = self.fe.data
            model = AutoGBDT(data, self.start_time, time_budget, self.n_class)
        elif self.model_name == 'AutoGBM':
            data = self.fe.data
            model = AutoGBM(data, self.start_time, time_budget, self.n_class)
        elif self.model_name == 'Ensemble':
            data = self.fe.generate_pyg_data()
            #model_gcn = AutoGCN(data, self.device, time.time(), time_budget/2, self.n_class, self.fe.data.train_ind)
            #gcn_rep = model_gcn.fit(rep_sign=True)
            model_gat = AutoGAT(data, self.device,  time.time(), time_budget*2/3, self.n_class, self.fe.data.train_ind)
            gat_rep = model_gat.fit(rep_sign=True)
            lightgbm = lgb.LGBMClassifier()
            data = self.fe.data
            #x = data.x
            x = gat_rep
            #x = np.concatenate((gcn_rep, gat_rep), axis=1)
            if self.fe.mf.any():
                x = np.concatenate((x, self.fe.mf), axis=1)
            x_train = x[data.train_mask]
            y_train = data.y[data.train_mask]
            x_test = x[data.test_mask]
            print("start gbm training: ")
            lightgbm.fit(x_train, y_train)
            return lightgbm.predict(x_test)
        pred = model.fit()

        return pred
