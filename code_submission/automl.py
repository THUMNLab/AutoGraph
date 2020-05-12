import numpy as np
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin, rand
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll import scope
import time
from sklearn.metrics import accuracy_score
import torch
import lightgbm as lgb
import copy

from layer import GCN, GAT
from utils import train_val_split, softmax

class AutoGCN:
    def __init__(self, data, device, start_time, time_budget, n_class, train_ind):
        self.data = data.to(device)
        self.device = device
        self.start_time = start_time
        self.time_budget = time_budget
        self.n_class = n_class
        self.train_ind = train_ind

        self.params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class,
                #'epoches': 150,
            }
        self.space = {
                'num_layers': scope.int(hp.choice('num_layers', [1, 2])),
                'hidden': scope.int(hp.quniform('hidden', 4, 128, 1)),
                'hidden2': scope.int(hp.quniform('hidden2', 4, 64, 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(0.5)),
                'epoches': scope.int(hp.quniform('epoches', 100, 200, 10)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        self.points = [{
                'num_layers': 1, #### warning!!!: 这里的1是上面hp.choice的数组下标。不是值。。
                'hidden': 64,
                'hidden2': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-4,
                },]


    def fit(self, rep_sign=False):
        preds = []
        scores = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget-(st-self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt/(i+1e-6)))
            if 2*pt/(i+1e-6) > remain_time:
                break
            train_mask, val_mask = train_val_split(self.train_ind, self.data.y.cpu().numpy()[self.train_ind], self.data.num_nodes)
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)

            hyperparams, score = self.hyper_optimization(train_mask, val_mask)
            #params = copy.copy(self.params)
            hyperparams['epoches'] *= 2

            self.model = GCN(**{**self.params, **hyperparams}).to(self.device)
            if rep_sign:
                val_mask = torch.BoolTensor(np.ones(self.data.num_nodes))
            else:
                val_mask = None
            pred = self.model.train_predict(self.data, train_mask=None, val_mask=val_mask, return_train=False)
            preds.append(pred.cpu().numpy())
            scores.append(score)
            pt += time.time()-st
        scores = np.array(scores)
        ind = np.where(scores>=max(scores)-0.05)[0]
        preds = [preds[i] for i in ind]
        scores = scores[ind]
        print(scores)
        res = sum(map(lambda x: x[0]*x[1], zip(preds, softmax(scores))))
        #print(res)
        if rep_sign:
            return res
        return res.argmax(1)


    def hyper_optimization(self, train_mask, val_mask, trials=None):
        st = time.time()
        def objective(hyperparams):
            model = GCN(**{**self.params, **hyperparams}).to(self.device)
            pred, pred_train = model.train_predict(self.data, train_mask, val_mask, return_train=True)
            score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred.max(1)[1]).cpu().numpy())
            score_train = accuracy_score(self.data.y[train_mask].cpu().numpy(), (pred_train.max(1)[1]).cpu().numpy())
            #score = -F.nll_loss(pred, self.data.y[val_mask]).item()
            return {'loss': -score+0.1*(score_train-score), 'status': STATUS_OK, 'acc': [score_train, score]}

        trials = generate_trials_to_calculate(self.points)
        best = fmin(fn=objective, space=self.space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        pt = time.time()-st
        remain_time = self.time_budget - (time.time()-self.start_time)
        if remain_time/pt > 6:
            best = fmin(fn=objective, space=self.space, trials=trials,
                    algo=tpe.suggest, max_evals=10, verbose=1,
                    )
        hyperparams = space_eval(self.space, best)
        best_score = -trials.best_trial['result']['loss']
        acc = trials.best_trial['result']['acc']
        print('score: {:.4f}, acc: {:.2f} {:.2f}, hyperparams: {}'.format(best_score, acc[0], acc[1], hyperparams))
        return hyperparams, best_score

class AutoGAT:
    def __init__(self, data, device, start_time, time_budget, n_class, train_ind):
        self.data = data.to(device)
        self.device = device
        self.start_time = start_time
        self.time_budget = time_budget
        self.n_class = n_class
        self.train_ind = train_ind

    def fit(self, rep_sign=False):
        params = {
                'features_num': self.data.x.size()[1],
                'num_class': self.n_class
                }
        preds = []
        scores = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget - (st - self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt / (i + 1e-6)))
            if 2 * pt / (i + 1e-6) > remain_time:
                break
            train_mask, val_mask = train_val_split(self.train_ind, self.data.y.cpu().numpy()[self.train_ind],
                                                   self.data.num_nodes)
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)

            hyperparams, score = self.hyper_optimization(params, train_mask, val_mask)
            hyperparams['epoches'] = 2 * hyperparams['epoches']

            self.model = GAT(**{**params, **hyperparams}).to(self.device)
            if rep_sign:
                val_mask = torch.BoolTensor(np.ones(self.data.num_nodes))
            else:
                val_mask = None
            pred = self.model.train_predict(self.data, train_mask=None, val_mask=val_mask, return_train=False,
                                            **hyperparams)
            preds.append(pred.cpu().numpy())
            scores.append(score)
            pt += time.time() - st
        scores = np.array(scores)
        ind = np.where(scores >= max(scores) - 0.1)[0]
        preds = [preds[i] for i in ind]
        scores = scores[ind]
        print(scores)
        res = sum(map(lambda x: x[0] * x[1], zip(preds, softmax(scores))))
        if rep_sign:
            return res
        return res.argmax(1)

    def hyper_optimization(self, params, train_mask, val_mask, trials=None):
        st = time.time()
        space = {
                #'num_layers': hp.choice('num_layers', [1, 2]),
                #'hidden': scope.int(hp.qloguniform('hidden', np.log(4), np.log(128), 1)),
                'dropout': hp.uniform('dropout', 0.1, 0.9),
                'lr': hp.loguniform('lr', np.log(0.001), np.log(0.5)),
                'epoches': scope.int(hp.quniform('epoches', 100, 200, 20)),
                'weight_decay': hp.loguniform('weight_decay', np.log(1e-4), np.log(1e-2))
                }
        points = [{
                #'num_layers': 1, #### warning!!!: 这里的1是上面hp.choice的数组下标。不是值。。
                #'hidden': 32,
                'dropout': 0.5,
                'lr': 0.005,
                'epoches': 200,
                'weight_decay': 5e-4,
                },]
        def objective(hyperparams):
            model = GAT(**{**params, **hyperparams}).to(self.device)
            pred, pred_train = model.train_predict(self.data, train_mask, val_mask, return_train=True, **hyperparams)
            score = accuracy_score(self.data.y[val_mask].cpu().numpy(), (pred.max(1)[1]).cpu().numpy())
            score_train = accuracy_score(self.data.y[train_mask].cpu().numpy(), (pred_train.max(1)[1]).cpu().numpy())
            # score = -F.nll_loss(pred, self.data.y[val_mask]).item()
            return {'loss': -score + 0.1 * (score_train - score), 'status': STATUS_OK, 'acc': [score_train, score]}

        trials = generate_trials_to_calculate(points)
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        pt = time.time() - st
        remain_time = self.time_budget - (time.time() - self.start_time)
        if remain_time / pt > 6:
            best = fmin(fn=objective, space=space, trials=trials,
                        algo=tpe.suggest, max_evals=10, verbose=1,
                        )
        hyperparams = space_eval(space, best)
        best_score = -trials.best_trial['result']['loss']
        acc = trials.best_trial['result']['acc']
        print('score: {:.4f}, acc: {:.2f} {:.2f}, hyperparams: {}'.format(best_score, acc[0], acc[1], hyperparams))
        return hyperparams, best_score

class AutoGBM:
    def __init__(self, data, start_time, time_budget, n_class):
        self.data = data
        self.start_time = start_time
        self.time_budget = time_budget
        self.n_class = n_class
        self.train_ind = data.train_ind

    def fit(self):
        params = {
            'boosting_type': 'gbdt',
            'num_threads': 4,
            'verbose': -1,
            'random_state': 0,
            'silent': -1,
        }
        preds = []
        scores = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget - (st - self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt / (i + 1e-6)))
            if 2 * pt / (i + 1e-6) > remain_time:
                break
            train_mask, val_mask = train_val_split(self.train_ind, self.data.y[self.train_ind],
                                                   self.data.num_nodes)
            x_train = self.data.x[train_mask]
            y_train = self.data.y[train_mask]
            x_val = self.data.x[val_mask]
            y_val = self.data.y[val_mask]
            hyperparams, score = self.hyper_optimization(params, x_train, y_train, x_val, y_val)
            self.model = lgb.LGBMClassifier(**params, **hyperparams)
            self.model.fit(self.data.x[self.data.train_mask], self.data.y[self.data.train_mask], verbose=100)
            pred = self.model.predict(self.data.x[self.data.test_mask], raw_score=True)
            preds.append(pred)
            scores.append(score)
            pt += time.time() - st
        scores = np.array(scores)
        ind = np.where(scores >= max(scores) - 0.1)[0]
        preds = [preds[i] for i in ind]
        scores = scores[ind]
        print(scores)
        res = sum(map(lambda x: x[0] * x[1], zip(preds, softmax(scores))))
        return res.argmax(1)

    def hyper_optimization(self, params, x_train, y_train, x_val, y_val):
        st = time.time()
        space = {
            #'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'num_leaves': hp.choice('num_leaves', np.linspace(16, 32, 64, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample': hp.uniform('subsample', 0.8, 1),
        }
        eval_set = [(x_train, y_train), (x_val, y_val)]

        def objective(hyperparams):
            model = lgb.LGBMClassifier(**params, **hyperparams, num_boost_round=20)
            model.fit(x_train, y_train, eval_set=eval_set, verbose=100, early_stopping_rounds=20)
            pred = model.predict(x_val)
            score = accuracy_score(y_val , pred)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()#generate_trials_to_calculate(points)
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        pt = time.time() - st
        remain_time = self.time_budget - (time.time() - self.start_time)
        if remain_time / pt > 6:
            best = fmin(fn=objective, space=space, trials=trials,
                        algo=tpe.suggest, max_evals=10, verbose=1,
                        )
        hyperparams = space_eval(space, best)
        best_score = -trials.best_trial['result']['loss']
        #acc = trials.best_trial['result']['acc']
        print('score: {:.4f}, hyperparams: {}'.format(best_score, hyperparams))
        return hyperparams, best_score



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
        scores = []
        pt = 0
        for i in range(5):
            st = time.time()
            remain_time = self.time_budget - (st - self.start_time)
            print('remain time: {:.4f}s, per iter: {:.4f}s'.format(remain_time, pt / (i + 1e-6)))
            if 2 * pt / (i + 1e-6) > remain_time:
                break

            train_mask, val_mask = train_val_split(self.train_ind, self.data.y[self.train_ind],
                                                   self.data.num_nodes)
            train_data = lgb.Dataset(self.data.x[train_mask], label=self.data.y[train_mask], free_raw_data=False)
            valid_data = lgb.Dataset(self.data.x[val_mask], label=self.data.y[val_mask], free_raw_data=False)
            hyperparams, score = self.hyper_optimization(params, train_data, valid_data)

            train_data = lgb.Dataset(self.data.x[self.data.train_mask], label=self.data.y[self.data.train_mask], free_raw_data=False)
            self.model = lgb.train({**params, **hyperparams}, train_data, verbose_eval=100)
            pred = self.model.predict(self.data.x[self.data.test_mask], num_iterations=self.model.best_iteration)
            preds.append(pred)
            scores.append(score)
            pt += time.time() - st
        scores = np.array(scores)
        ind = np.where(scores >= max(scores) - 0.1)[0]
        preds = [preds[i] for i in ind]
        scores = scores[ind]
        print(scores)
        res = sum(map(lambda x: x[0] * x[1], zip(preds, softmax(scores))))
        return res.argmax(1)

    def hyper_optimization(self, params, train_data, val_data):
        st = time.time()
        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            "num_leaves": hp.choice("num_leaves", np.linspace(16, 32, 64, dtype=int)),
            'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
            'subsample': hp.uniform('subsample', 0.8, 1),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 20, valid_sets=val_data,
                              early_stopping_rounds=100, verbose_eval=100)
            #score = model.best_score["valid_0"][params["metric"]]
            pred = model.predict(val_data.get_data(), num_iterations=model.best_iteration).argmax(1)
            score = accuracy_score(val_data.get_label() , pred)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()#generate_trials_to_calculate(points)
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=5, verbose=1,
                    )
        pt = time.time() - st
        remain_time = self.time_budget - (time.time() - self.start_time)
        if remain_time / pt > 6:
            best = fmin(fn=objective, space=space, trials=trials,
                        algo=tpe.suggest, max_evals=10, verbose=1,
                        )
        hyperparams = space_eval(space, best)
        best_score = -trials.best_trial['result']['loss']
        #acc = trials.best_trial['result']['acc']
        print('score: {:.4f}, hyperparams: {}'.format(best_score, hyperparams))
        return hyperparams, best_score
