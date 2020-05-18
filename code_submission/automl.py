import numpy as np
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin, rand
import hyperopt.pyll.stochastic
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll import scope
import time
from sklearn.metrics import accuracy_score
import torch
import lightgbm as lgb
import copy
import pprint

from layer import GCN, GAT, GIN, GraphSAGE
from utils import train_val_split, softmax

class AutoGCN:
    def __init__(self, data, device, iter_num, timer, n_class, **args):
        self.data = data.to(device)
        self.device = device
        self.timer = timer
        self.n_class = n_class
        self.iter_num = iter_num

        self.flag_end = False

        self.params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class,
                #'epoches': 150,
            }
        self.space = {
                'num_layers': scope.int(hp.choice('num_layers', [1, 2])),
                #'agg': hp.choice('agg', ['concat', 'self']),
                'hidden': scope.int(hp.quniform('hidden', 4, 128, 1)),
                'hidden2': scope.int(hp.quniform('hidden2', 4, 64, 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(1.0)),
                'epoches': scope.int(hp.quniform('epoches', 100, 300, 10)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        self.points = [{
                'num_layers': 2,
                #'agg': 'concat',
                'hidden': 64,
                'hidden2': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-3,
                },]


    def fit(self, train_mask, val_mask):
        if self.iter_num < len(self.points):
            hyperparams = self.points[self.iter_num]
        else:
            hyperparams = hyperopt.pyll.stochastic.sample(self.space) 
        #pprint.pprint(hyperparams, width=1)

        self.model = GCN({**self.params, **hyperparams, 'timer': self.timer}).to(self.device) 
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        pred, pred_val, flag = self.model.train_predict(self.data, train_mask=train_mask, val_mask=val_mask)
        if flag:
            self.flag_end = True
        score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred_val.max(1)[1]).cpu().numpy())
        return pred.cpu().numpy(), score

class AutoGAT:
    def __init__(self, data, device, iter_num, timer, n_class, **args):
        self.data = data.to(device)
        self.device = device
        self.timer = timer
        self.n_class = n_class
        self.iter_num = iter_num

        self.flag_end = False

        self.params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class,
            }
        self.space = {
                'hidden': scope.int(hp.quniform('hidden', 4, 64, 1)),
                'heads': scope.int(hp.quniform('heads', 2, 4, 1)),
                'hidden2': scope.int(hp.quniform('hidden2', 4, 64, 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(1.0)),
                'epoches': scope.int(hp.quniform('epoches', 100, 300, 10)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        self.points = [{
                'hidden': 32,
                'heads': 4,
                'hidden2': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-3,
                },]

    def fit(self, train_mask, val_mask):
        if self.iter_num < len(self.points):
            hyperparams = self.points[self.iter_num]
        else:
            hyperparams = hyperopt.pyll.stochastic.sample(self.space) 
        #pprint.pprint(hyperparams, width=1)

        self.model = GAT({**self.params, **hyperparams, 'timer': self.timer}).to(self.device) 
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        pred, pred_val, flag = self.model.train_predict(self.data, train_mask=train_mask, val_mask=val_mask)
        if flag:
            self.flag_end = True
        score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred_val.max(1)[1]).cpu().numpy())
        return pred.cpu().numpy(), score

class AutoSAGE:
    def __init__(self, data, device, iter_num, timer, n_class, **args):
        self.data = data.to(device)
        self.device = device
        self.timer = timer
        self.n_class = n_class
        self.iter_num = iter_num

        self.flag_end = False

        self.params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class,
                #'epoches': 150,
            }
        self.space = {
                'num_layers': scope.int(hp.choice('num_layers', [1, 2])),
                'agg': hp.choice('agg', ['concat', 'self']),
                'hidden': scope.int(hp.quniform('hidden', 4, 128, 1)),
                'hidden2': scope.int(hp.quniform('hidden2', 4, 64, 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(1.0)),
                'epoches': scope.int(hp.quniform('epoches', 100, 300, 10)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        self.points = [{
                'num_layers': 2,
                'agg': 'concat',
                'hidden': 64,
                'hidden2': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-3,
                },]


    def fit(self, train_mask, val_mask):
        if self.iter_num < len(self.points):
            hyperparams = self.points[self.iter_num]
        else:
            hyperparams = hyperopt.pyll.stochastic.sample(self.space) 
        #pprint.pprint(hyperparams, width=1)

        self.model = GraphSAGE({**self.params, **hyperparams, 'timer': self.timer}).to(self.device) 
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        pred, pred_val, flag = self.model.train_predict(self.data, train_mask=train_mask, val_mask=val_mask)
        if flag:
            self.flag_end = True
        score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred_val.max(1)[1]).cpu().numpy())
        return pred.cpu().numpy(), score

class AutoGIN:
    def __init__(self, data, device, iter_num, timer, n_class, **args):
        self.data = data.to(device)
        self.device = device
        self.timer = timer
        self.n_class = n_class
        self.iter_num = iter_num

        self.flag_end = False

        self.params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class,
                #'epoches': 150,
            }
        self.space = {
                'num_layers': scope.int(hp.choice('num_layers', [1, 2])),
                'agg': hp.choice('agg', ['concat', 'self']),
                'hidden': scope.int(hp.quniform('hidden', 4, 128, 1)),
                'hidden2': scope.int(hp.quniform('hidden2', 4, 64, 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(1.0)),
                'epoches': scope.int(hp.quniform('epoches', 100, 300, 10)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        self.points = [{
                'num_layers': 2,
                'agg': 'concat',
                'hidden': 64,
                'hidden2': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-3,
                },]

    def fit(self, train_mask, val_mask):
        if self.iter_num < len(self.points):
            hyperparams = self.points[self.iter_num]
        else:
            hyperparams = hyperopt.pyll.stochastic.sample(self.space) 
        #pprint.pprint(hyperparams, width=1)

        self.model = GIN({**self.params, **hyperparams, 'timer': self.timer}).to(self.device) 
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        pred, pred_val, flag = self.model.train_predict(self.data, train_mask=train_mask, val_mask=val_mask)
        if flag:
            self.flag_end = True
        score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred_val.max(1)[1]).cpu().numpy())
        return pred.cpu().numpy(), score



class AutoGBM:
    def __init__(self, data, device, iter_num, timer, n_class):
        self.data = data
        self.timer = timer
        self.n_class = n_class
        self.iter_num = iter_num

        self.flag_end = False
        self.params = {
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 0,
            'silent': -1,
            'subsample': 0.9,
            'subsample_freq': 1,
            'n_jobs': 10,
            'objective': 'multiclass',
            'num_class': n_class
        }
        self.space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'num_leaves': hp.choice('num_leaves', np.linspace(16, 32, 64, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.7, 0.9, 0.1)),
        }

    def fit(self, train_mask, val_mask):
        x_train = self.data.x[train_mask]
        y_train = self.data.y[train_mask]
        x_val = self.data.x[val_mask]
        y_val = self.data.y[val_mask]
        hyperparams, score = self.hyper_optimization(self.params, x_train, y_train, x_val, y_val)
        self.model = lgb.LGBMClassifier(**self.params, **hyperparams, n_estimators=20)
        self.model.fit(x_train, y_train)
        pred_val = self.model.predict(self.data.x[val_mask])
        score = accuracy_score(pred_val, y_val)
        pred = self.model.predict(self.data.x[self.data.test_mask], raw_score=True)
        return pred, score

    def hyper_optimization(self, params, x_train, y_train, x_val, y_val):
        eval_set = [(x_train, y_train), (x_val, y_val)]

        def objective(hyperparams):
            model = lgb.LGBMClassifier(**params, **hyperparams, n_estimators=5)
            model.fit(x_train, y_train, eval_set=eval_set, verbose=False, early_stopping_rounds=20)
            pred = model.predict(x_val)
            score = accuracy_score(y_val , pred)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()#generate_trials_to_calculate(points)
        best = fmin(fn=objective, space=self.space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=0, timeout=self.timer.remain_time()*1/3,
                    )
        remain_time = self.timer.remain_time()
        hyperparams = space_eval(self.space, best)
        best_score = -trials.best_trial['result']['loss']
        #acc = trials.best_trial['result']['acc']
        #print('score: {:.4f}, hyperparams: {}'.format(best_score, hyperparams))
        return hyperparams, best_score
