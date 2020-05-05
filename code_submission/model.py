"""the simple baseline for autograph"""

import numpy as np
import pandas as pd
import os, time

import torch

from feature_engineering import Feature_Engineering
from automl import AutoGCN, AutoGBDT

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
        self.model_name = 'AutoGCN'

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
        elif self.model_name == 'AutoGBDT':
            data = self.fe.data
            model = AutoGBDT(data, self.start_time, time_budget, self.n_class)
        pred = model.fit()

        return pred
