"""the simple baseline for autograph"""

import numpy as np
import pandas as pd
import os, time
import gc
import traceback

# a puzzle: importing lightgbm here will cost 30x times in gbdt_gen
import torch

from feature_engineering import Feature_Engineering
from automl import AutoGCN, AutoGAT, AutoGBM, AutoSAGE, AutoGIN
from utils import Timer, setx

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
        self.timer = Timer()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.models = ['AutoGCN', 'AutoSAGE', 'AutoGAT']
        self.model_mapping = {
                'AutoGCN': AutoGCN,
                'AutoGAT': AutoGAT,
                'AutoSAGE': AutoSAGE,
                'AutoGIN': AutoGIN,
                'AutoGBM': AutoGBM,}

    def train_predict(self, data, time_budget,n_class,schema):
        self.timer.time_budget = time_budget
        self.n_class = n_class
        self.schema = schema
        self.fe = Feature_Engineering(data, self.timer)

        stacking = False
        n_round = 20
        iter_num = 0
        preds, scores, model_names = [], [], []
        flag_end = False
        black_lists = []
        stack_pre_model = ['AutoGCN','AutoGAT', 'AutoSAGE', 'AutoGIN']
        stack_after_model = ['AutoGBM']
        if (not self.fe.unweighted) or self.fe.num_edges > 1000000:
            black_lists.append('AutoGAT')
        scores_model = dict(zip(self.models, [[] for _ in range(len(self.models))]))
        for train_mask, val_mask in self.fe.split(n_round):
            for i in range(1):
                preds_all = []
                scores_per_data = []
                preds_per_data = []
                model_names_per_data = []
                for model_name in self.models:
                    if model_name in black_lists:
                        continue
                    try:
                        if stacking and (model_name in stack_after_model):
                            x = self.fe.data.x
                            self.fe.data = setx(self.fe.data, np.hstack(preds_all+[x]))
                        data = self.fe.get_data(model_name)
                        model_f = self.model_mapping[model_name]
                        model = model_f(data, self.device, iter_num, self.timer, self.n_class)
                        pred, score = model.fit(train_mask, val_mask)
                        print("Round: {}-{}, model: {}, score: {:.4f}, remain_time: {:.2f}".format(iter_num, i, model_name, score, self.timer.remain_time()))
                        flag_end = model.flag_end
                        if model_name != 'AutoGBM':
                            preds_all.append(pred)
                            preds_per_data.append(pred[self.fe.data.test_mask])
                        else:
                            preds_per_data.append(pred)
                        scores_per_data.append(score)
                        model_names_per_data.append(model_name)
                        scores_model[model_name].append(score)
                        if stacking and (model_name in stack_after_model):
                            self.fe.data = setx(self.fe.data, x)
                    except Exception as e:
                        print("Error!!!!!!!!!")
                        print(e)
                        traceback.print_exc()
                        print("Add {} to black_lists".format(model_name))
                        black_lists.append(model_name)
                    if flag_end:
                        if len(scores_per_data) > 0:
                            ind = np.argmax(scores_per_data)
                            scores.append(scores_per_data[ind])
                            preds.append(preds_per_data[ind])
                            model_names.append(model_names_per_data[ind])
                        return self.get_results(preds, scores, model_names)
                    gc.collect()
                if len(scores_per_data) > 0:
                    ind = np.argmax(scores_per_data)
                    scores.append(scores_per_data[ind])
                    preds.append(preds_per_data[ind])
                    model_names.append(model_names_per_data[ind])
                for k, v in scores_model.items():
                    print("Score: {} {:.4f} {}".format(k, np.mean(v) if len(v) > 0 else 0.0, len(v)))
                if len(model_names) >= 5:
                    for model_name in self.models:
                        if model_name not in model_names and model_name not in black_lists:
                            print("Add {} to black_lists".format(model_name))
                            black_lists.append(model_name)
            iter_num += 1
        return self.get_results(preds, scores, model_names)

    def get_results(self, preds, scores, model_names):
        ensemble_std_threshold = 1e-2

        scores = np.array(scores)
        ind = np.argsort(scores)[::-1]
        scores = scores[ind]

        num = 3
        for i in range(len(scores), 3, -1):
            std = np.std(scores[:i])
            num = i
            if std < ensemble_std_threshold:
                break

        num = min(10, num)
        ind = ind[:num]
        scores = scores[:num]
        print(scores)
        model_names = [model_names[i] for i in ind]
        print(model_names)
        scores = scores + 20*(scores - scores.mean())
        scores = np.array([max(0.01, i) for i in scores])
        scores = scores / scores.sum()
        print(scores)
        preds = [preds[i] for i in ind]
        res = sum(map(lambda x: x[0] * x[1], zip(preds, scores)))
        return res.argmax(1)
