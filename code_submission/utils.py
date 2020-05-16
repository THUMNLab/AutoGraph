import numpy as np
import time
import torch
from typing import Any
from functools import wraps
from sklearn.model_selection import train_test_split

nesting_level = 0

def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(str(space) + " " + str(entry))

def timeit(method, start_log=None):
    @wraps(method)
    def wrapper(*args, **kw):
        global nesting_level

        log("Start [" + str(method.__name__) + "]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End   [" + str(method.__name__) + "]. Time elapsed: "+ "%.2f" % ((end_time - start_time)) + " sec.")
        return result
    return wrapper

def norm_minmax(x):
    return (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))

def norm_z(x):
    return (x-x.mean(axis=0)) / x.std(axis=0)

def norm_max(x):
    return x/x.max(axis=0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def train_val_split(train_ind, y, num_nodes, ratio=0.8, perm=True):
    train_ind, val_ind, y_train, y_val = train_test_split(train_ind, y, test_size=1-ratio, shuffle=perm, stratify=y)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_ind] = 1
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_ind] = 1
    return train_mask, val_mask

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def setx(data, x):
    return data._replace(x=x)

def set_default(args, d):
    for k, v in d.items():
        if k not in args:
            args[k] = v
    return args

class Timer:
    def __init__(self, time_budget=0):
        self.start_time = time.time()
        self.time_budget = time_budget

    def remain_time(self):
        return self.time_budget-(time.time()-self.start_time)

def label_distribution(y):
    res = {}
    for i in y:
        if i not in res:
            res[i] = 0
        res[i]+= 1
    t = sum(res.values())
    for k, v in res.items():
        res[k] = v*1.0/t
    return res
