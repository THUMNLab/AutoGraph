import numpy as np
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll import scope
import time
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from layer import GCN
from utils import train_val_split

class AutoGCN:
    def __init__(self, data, device, start_time, time_budget, n_class, train_ind):
        self.data = data.to(device)
        self.device = device
        self.start_time = start_time
        self.time_budget = time_budget
        self.n_class = n_class
        self.train_ind = train_ind

    def fit(self):
        params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class
                }
        preds = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget-(st-self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt/(i+1e-6)))
            if 3*pt/(i+1e-6) > remain_time:
                break
            train_mask, val_mask = train_val_split(self.train_ind, self.data.num_nodes)
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)

            hyperparams = self.hyper_optimization(params, train_mask, val_mask)
            hyperparams['epoches'] = 3*hyperparams['epoches']

            self.model = GCN(**{**params, **hyperparams}).to(self.device)
            pred = self.model.train_predict(self.data, train_mask=None, val_mask=None, **hyperparams)
            preds.append(pred)
            pt += time.time()-st
        return sum(preds).max(1)[1].cpu().numpy().flatten()

    def hyper_optimization(self, params, train_mask, val_mask, trials=None):
        space = {
                'num_layers': hp.choice('num_layers', [1, 2]), 
                'hidden': scope.int(hp.qloguniform('hidden', np.log(4), np.log(128), 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(0.5)),
                'epoches': scope.int(hp.quniform('epoches', 100, 200, 20)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        points = [{
                'num_layers': 1, #### warning!!!: 这里的1是上面hp.choice的数组下标。不是值。。
                'hidden': 32,
                'dropout': 0.5,
                'lr': 0.025,
                'epoches': 200,
                'weight_decay': 5e-3,
                },]
        def objective(hyperparams):
            model = GCN(**{**params, **hyperparams}).to(self.device)
            pred = model.train_predict(self.data, train_mask, val_mask, **hyperparams).max(1)[1]
            score = accuracy_score(self.data.y[val_mask].cpu().numpy(), pred.cpu().numpy())
            return {'loss': -score, 'status': STATUS_OK}

        trials = generate_trials_to_calculate(points)
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        hyperparams = space_eval(space, best)
        print('score: {:.4f}, hyperparams: {}'.format(-trials.best_trial['result']['loss'], hyperparams))
        return hyperparams

class AutoGBDT:
    def __init__(self, data, start_time, time_budget, n_class):
        self.data = data
        self.start_time = start_time
        self.time_budget = time_budget
        self.n_class = n_class
        self.train_ind = data.train_ind

    def fit(self):
        params = {
                'objective': 'multiclass',
                'num_class': self.n_class,
                'metric': 'multi_logloss',
                'num_threads': 4,
                'verbose': -1,
                }
        preds = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget-(st-self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt/(i+1e-6)))
            if 3*pt/(i+1e-6) > remain_time:
                break
            train_mask, val_mask = train_val_split(self.train_ind, self.data.num_nodes)

            train_data = lgb.Dataset(self.data.x[train_mask], label=self.data.y[train_mask], free_raw_data=False)
            valid_data = lgb.Dataset(self.data.x[val_mask], label=self.data.y[val_mask], free_raw_data=False)

            hyperparams = self.hyper_optimization(params, train_data, valid_data)
            print("best hyperparams: ", hyperparams)
            hyperparams['num_iterations'] = max(100, hyperparams['num_iterations'])
            self.model = lgb.train({**params, **hyperparams}, train_data, 100, 
                                    valid_sets=valid_data, early_stopping_rounds=100, verbose_eval=100)
            pred = self.model.predict(self.data.x[self.data.test_mask], num_iterations=self.model.best_iteration)
            preds.append(pred)
            pt += time.time()-st
        return sum(preds).argmax(1)

    def hyper_optimization(self, params, train_data, val_data):
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.1)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 150, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
            "num_iterations": scope.int(hp.quniform('num_iterations', 100, 200, 20))
            }
        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 50, valid_sets=val_data, early_stopping_rounds=30, verbose_eval=100)
            #score = model.best_score["valid_0"][params["metric"]]
            pred = model.predict(val_data.get_data(), num_iterations=model.best_iteration).argmax(1)
            score = accuracy_score(val_data.get_label() , pred)

            return {'loss': -score, 'status': STATUS_OK}
        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        hyperparams = space_eval(space, best)
        print('score: {:.4f}, hyperparams: {}'.format(-trials.best_trial['result']['loss'], hyperparams))
        return hyperparams


