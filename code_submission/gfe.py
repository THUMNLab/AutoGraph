import torch
import time
from typing import Any
from config import *
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pandas as pd
from functools import wraps
#os.system('pip install tabulate')
from tabulate import tabulate

from utils import log, timeit

class Graphlet:
    r"""refer to 
    Ahmed, N. K., Willke, T. L., & Rossi, R. A. (2016). 
    Estimation of local subgraph counts. Proceedings - 2016 IEEE International Conference on Big Data, Big Data 2016, 586–595. 
    https://doi.org/10.1109/BigData.2016.7840651
    """
    def __init__(self,data,sample_error=0.1,sample_confidence=0.1):
        self._data=data
        self._init()
        
        self._sample_error=sample_error
        self._sample_confidence=sample_confidence
        self._dw=int(np.ceil(0.5*(self._sample_error**-2)*np.log(2/self._sample_confidence)))
        print("sample error {} , confidence {},num {}".format(self._sample_error,self._sample_confidence,self._dw))

   
    def _init(self):
        self._edges=list(self._data.edge_index)
        self._edges=[self._edges[0],self._edges[1]]
        self._num_nodes=self._data.num_nodes
        self._num_edges=len(self._edges[0])
        self._neighbours=[[] for _ in range(self._num_nodes)]
        for i in range(len(self._edges[0])):
            u,v=self._edges[0][i],self._edges[1][i]
            self._neighbours[u].append(v)
        
        print("nodes {} , edges {}".format(self._num_nodes,self._num_edges))
        
        # sorting
        self._node_degrees=np.array([len(x) for x in self._neighbours])
        self._nodes=np.argsort(self._node_degrees)
        for i in self._nodes:
            self._neighbours[i]=[x for _,x in sorted(zip(self._node_degrees[self._neighbours[i]],self._neighbours[i]),reverse=True)]
        self._neighbours=[np.array(x) for x  in self._neighbours]
    def _get_gdv(self,v,u):
        if self._node_degrees[v]>=self._node_degrees[u]:
            pass
        else:
            u,v=v,u
        Sv,Su,Te=set(),set(),set()
        sigma1,sigma2=0,0
        nb=self._neighbours
        N=self._num_nodes
        M=self._num_edges
        phi=np.zeros(self._num_nodes,dtype=int)
        c1,c2,c3,c4=1,2,3,4
        x=np.zeros(16,dtype=int)
        # p1
        for w in nb[v]:
            if w!=u:
                Sv.add(w)
                phi[w]=c1
        # p2
        for w in nb[u]:
            if w!=v:
                if phi[w]==c1:
                    Te.add(w)
                    phi[w]=c3
                    Sv.remove(w)
                else:
                    Su.add(w)
                    phi[w]=c2
        # p3
        for w in Te:
            for r in nb[w]:
                if phi[r]==c3:
                    x[5]+=1
            phi[w]=c4
            sigma2=sigma2+len(nb[w])-2
        # p4
        for w in Su:
            for r in nb[w]:
                if phi[r]==c1:
                    x[8]+=1
                if phi[r]==c2:
                    x[7]+=1
                if phi[r]==c4:
                    sigma1+=1
            phi[w]=0
            sigma2=sigma2+len(nb[w])-1
        # p5
        for w in Sv:
            for r in nb[w]:
                if phi[r]==c1:
                    x[7]+=1
                if phi[r]==c4:
                    sigma1+=1
            phi[w]=0
            sigma2=sigma2+len(nb[w])-1
            
        lsv,lsu,lte,du,dv=len(Sv),len(Su),len(Te),len(nb[u]),len(nb[v])
        # 3-graphlet
        x[1]=lte
        x[2]=du+dv-2-2*x[1]
        x[3]=N-x[2]-x[1]-2
        x[4]=N*(N-1)*(N-2)/6-(x[1]+x[2]+x[3])
        # 4 connected graphlets
        x[6]=x[1]*(x[1]-1)/2 -x[5]
        x[10]=lsv*lsu-x[8]
        x[9]=lsv*(lsv-1)/2+lsu*(lsu-1)/2 -x[7]
        # 4 diconnected graphlets
        t1=N-(lte+lsu+lsv+2)
        x[11]=x[1]*t1
        x[12]=M-(du+dv-1)-(sigma2-sigma1-x[5]-x[8]-x[7])
        x[13]=(lsu+lsv)*t1
        x[14]=t1*(t1-1)/2-x[12]
        x[15]=N*(N-1)*(N-2)*(N-3)/24-np.sum(x[5:15])
        
        return x
    
    def _get_gdv_sample(self,v,u):
        if self._node_degrees[v]>=self._node_degrees[u]:
            pass
        else:
            u,v=v,u
        Sv=set()
        sigma1,sigma2=0,0
        nb=self._neighbours
        N=self._num_nodes
        M=self._num_edges
        phi=np.zeros(self._num_nodes,dtype=int)
        c1,c2,c3,c4=1,2,3,4
        x=np.zeros(16)
        dw=self._dw
        
        # p1
        Sv=set(nb[v][nb[v]!=u])
        phi[list(Sv)]=c1
        # p2
        p2w=nb[u][nb[u]!=c1]
        p2w1=p2w[phi[p2w]==c1]
        p2w2=p2w[phi[p2w]!=c1]
        Te=p2w1
        phi[p2w1]=c3
        Sv-=set(list(p2w1))
        Su=p2w2
        phi[p2w2]=c2
        # p3
        for w in Te:
            if dw>=len(nb[w]):
                region=nb[w]
                inc=1
            else:
                region=np.random.choice(nb[w],dw,replace=False)
                inc=self._node_degrees[w]/dw
            phir=phi[region]
            x[5]+=inc*np.sum(phir==c3)
            phi[w]=c4
            sigma2=sigma2+len(nb[w])-2
        # p4
        for w in Su:
            if dw>=len(nb[w]):
                region=nb[w]
                inc=1
            else:
                region=np.random.choice(nb[w],dw,replace=False)
                inc=self._node_degrees[w]/dw
            phir=phi[region]
            x[8]+=inc*np.sum(phir==c1)
            x[7]+=inc*np.sum(phir==c2)
            sigma1+=inc*np.sum(phir==c4)
            phi[w]=0
            sigma2=sigma2+len(nb[w])-1
        # p5
        for w in Sv:
            if dw>=len(nb[w]):
                region=nb[w]
                inc=1
            else:
                region=np.random.choice(nb[w],dw,replace=False)
                inc=self._node_degrees[w]/dw
            phir=phi[region]
            x[7]+=inc*np.sum(phir==c1)
            sigma1+=inc*np.sum(phir==c4)
            phi[w]=0
            sigma2=sigma2+len(nb[w])-1
            
        lsv,lsu,lte,du,dv=len(Sv),len(Su),len(Te),len(nb[u]),len(nb[v])
        # 3-graphlet
        x[1]=lte
        x[2]=du+dv-2-2*x[1]
        x[3]=N-x[2]-x[1]-2
        x[4]=N*(N-1)*(N-2)/6-(x[1]+x[2]+x[3])
        # 4 connected graphlets
        x[6]=x[1]*(x[1]-1)/2 -x[5]
        x[10]=lsv*lsu-x[8]
        x[9]=lsv*(lsv-1)/2+lsu*(lsu-1)/2 -x[7]
        # 4 diconnected graphlets
        t1=N-(lte+lsu+lsv+2)
        x[11]=x[1]*t1
        x[12]=M-(du+dv-1)-(sigma2-sigma1-x[5]-x[8]-x[7])
        x[13]=(lsu+lsv)*t1
        x[14]=t1*(t1-1)/2-x[12]
        x[15]=N*(N-1)*(N-2)*(N-3)/24-np.sum(x[5:15])
        
        return x
    def get_gdvs(self,sample=True):
        res=np.zeros((self._num_nodes,15))
        for u in tqdm(range(self._num_nodes)):
            vs=self._neighbours[u]
            if len(vs)!=0:
                gdvs=[]
                for v in tqdm(vs,disable=len(vs)<100):
                    if sample:
                        gdvs.append(self._get_gdv_sample(u,v))
                    else:
                        gdvs.append(self._get_gdv(u,v))
                res[u,:]=np.mean(gdvs,axis=0)[1:]
        return res
    def get_gdvs_cp(self,workers='max'):
        r"""
        c++ parallel , same function as get_gdvs
        """
        tmpfile='tmp.mtx'
        tmpmicro='tmp.micro'
        self._save(tmpfile)
#         os.system("{} -f {} --micro {} -v -w {}".format(pgd_path,tmpfile,tmpmicro,workers))
        os.system("{} -f {} --micro {} -w {}".format(pgd_path,tmpfile,tmpmicro,workers))
        return self._load(tmpmicro)
    def _save(self,filename):
        with open(filename,'w') as file:
            file.write('{} {} {}\n'.format(self._num_nodes,self._num_nodes,self._num_edges))
            for u in self._nodes:
                for v in self._neighbours[u]:
                    file.write("{} {}\n".format(u+1,v+1))
    def _load(self,filename):
        df=pd.read_csv(filename)
        edges=df[['% src','dst']].values
        egdvs=df.values[:,2:]

        num_nodes=np.max(edges)
        ngdvs=np.zeros((num_nodes,8))

        nbs=[[] for _ in range(num_nodes)]
        for i,(u,v) in enumerate(edges):
            u-=1
            v-=1
            nbs[u].append(i)
            nbs[v].append(i)

        for i in range(num_nodes):
            if len(nbs[i])!=0:
                ngdvs[i,:]=np.mean(egdvs[nbs[i]],axis=0)
        return ngdvs

from sklearn import preprocessing
mms=preprocessing.MinMaxScaler()
ss=preprocessing.StandardScaler()

def scale(x):
#     return mms.fit_transform(x)
    return ss.fit_transform(x)
def identity_gen(data):
    r"""return original features
    """
    res=data.x
    return res
@timeit
def identity_fil(data):
    r"""remove id one hot and constant features
    """
    d1,d2=data.x.shape
    xx=data.x
    if d2>=d1:
        if np.allclose(xx[:,:d1],np.eye(d1)):
            return np.empty((d1,0))
    return xx[:,np.where(np.all(xx == xx[0,:], axis = 0)==False)[0]]
def identity_gen_simple_sel(data):
    r"""return features selected by similarity between labels-features and features-features
    """
    tx=data.x[data.train_mask]
    ty=data.y[data.train_mask]
    ty_sel=sim_y_sel(tx,ty)
    print('y sel {}'.format(ty_sel.shape))
    if len(ty_sel)<=1:
        return np.empty((data.x.shape[0],0))
    tx=tx[:,ty_sel]
    tx_sel=sim_x_sel(tx)
    print('x sel {}'.format(tx_sel.shape))
    if len(tx_sel)<=1:
        return np.empty((data.x.shape[0],0))
    res=data.x[:,ty_sel[tx_sel]]
    return res
def zero_gen(data):
    return np.zeros_like(data.x)

@timeit 
def gdv_gen(data):
    r"""graphlet degree vectors
    """
    gl=Graphlet(data)
    res=gl.get_gdvs_cp(workers)
    return res

#os.system('pip install networkx')
import networkx as nx
@timeit
def pagerank_gen(data):
    graph=nx.DiGraph()
    w=data.edge_weight
    eg=[(u,v,w[i]) for i,(u,v) in enumerate(data.edge_index.T)]
    graph.add_weighted_edges_from(eg)
    pagerank=nx.pagerank(graph)
    pr=np.zeros((data.num_nodes,1))
    for i,v in pagerank.items():
        pr[i]=v
    return pr

from collections import Counter
@timeit
def degree_gen(data):
    egs=data.edge_index.T
    num_nodes=data.num_nodes
    res=np.zeros((num_nodes,2))
    cin,cout=Counter(egs[:,0]),Counter(egs[:,1])
    res[list(cin.keys()),0]=list(cin.values())
    res[list(cout.keys()),1]=list(cout.values())
    return res

#os.system('pip install featuretools')
import featuretools as ft
@timeit
def featuretools_gen(data):
    
    num_nodes,num_feas=data.x.shape
    feas=pd.DataFrame(
        np.concatenate([np.arange(num_nodes).reshape((-1,1)),data.x],axis=1),
        index=None,
        columns=['node_index']+['f{}'.format(i) for i in range(num_feas)]
    )
    es=ft.EntitySet()
    es.entity_from_dataframe(
        entity_id='data',
        dataframe=feas,
        index='node_index'
    )
    fm,f_def=ft.dfs(
        entityset=es,
        target_entity='data',
        verbose=True,
    )
    res=fm.values
    print(res)
    return res

#os.system('pip install lightgbm')
import lightgbm as lgb
@timeit
def gbdt_gen(data,params={
        'verbosity':-1,
        'random_state':47,
    
    },fixlen=1000): 
    x=data.x[data.train_mask]
    _,num_feas = x.shape
    if num_feas < fixlen :
        return data.x
    label=data.y[data.train_mask]
    fnames=np.array(['f{}'.format(i) for i in range(num_feas)])
    train_x=pd.DataFrame(x,columns=fnames,index=None)
    
    dtrain=lgb.Dataset(train_x,label=label)
    
    clf=lgb.train(train_set=dtrain,params=params)
    imp=np.array(list(clf.feature_importance()))

    res=data.x[:,np.argsort(imp)[-fixlen:]]
    return res

#os.system('pip install lightgbm')
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures as pf
@timeit
def gbdt_gen2(data):

    x=data.x
    poly=pf(2,interaction_only=True)
    x=poly.fit_transform(x)
    tmp=data.x
    data.x=torch.FloatTensor(x)
    res=gbdt_gen(data)
    data.x=tmp
    return res

from torch import nn
@timeit 
def deepfm_gen(data,emb_size=100,epochs=200):
    class DeepFm(nn.Module):
        def __init__(self,featnum,embsize,num_class):
            nn.Module.__init__(self)
            self._lat_embed=nn.Parameter(torch.ones(featnum,embsize))
            self._one_embed=nn.Parameter(torch.ones(featnum,1))
            nn.init.normal_(self._lat_embed,std=0.01)
            nn.init.normal_(self._one_embed,std=0.01)
            self._field_size=featnum
            self._emb_size=embsize
            self._dense_dim=self._field_size*emb_size
            self._num_class=num_class
            self._fc1=nn.Sequential(nn.Linear(self._dense_dim,emb_size),nn.ReLU(inplace=True))
            self._fc2=nn.Sequential(nn.Linear(emb_size,emb_size),nn.ReLU(inplace=True))
            self._output=nn.Sequential(nn.Linear(1+emb_size*2,num_class))
        def forward(self,x):
            '''
            x : feature matrix , ( batchnum * featurenum)
            '''
            x=x.unsqueeze(dim=-1)
            f1=self._one_embed*x
            f1=torch.sum(f1,dim=1)
            
            x=self._lat_embed*x
            emb_dnn=torch.reshape(x,(-1,self._dense_dim))
            dnn=self._fc2(self._fc1(emb_dnn))
            
            fm1=torch.sum(x,dim=1)
            fm1=torch.mul(fm1,fm1)
            fm2=torch.mul(x,x)
            fm2=torch.sum(fm2,dim=1)
            fm=0.5*(fm1-fm2)
            
            h=torch.cat([f1,fm,dnn],dim=1)
            output= self._output(h)
            return F.log_softmax(output, dim=-1)
        def fit(self,train_x,train_y,epochs=500):
            model=self
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            
            for epoch in tqdm(range(1,epochs)):
                model.train()
                optimizer.zero_grad()
                loss = F.nll_loss(model(train_x), train_y)
                loss.backward()
                optimizer.step()

        def pred(self,x):
            self.eval()
            with torch.no_grad():
                pred=self.forward(x).max(1)[1]
            return pred
        def get_feat(self,x):
            self.eval()
            with torch.no_grad():
                x=x.unsqueeze(dim=-1)
                e1=self._one_embed*x
                e2=self._lat_embed*x
                emb_one=e1.squeeze(-1)
                emb_dnn=torch.reshape(e2,(-1,self._dense_dim))
                emb=torch.cat([emb_one,emb_dnn],dim=-1)
            return emb.cpu().numpy()
    # data
    
    x=data.x
    num_nodes,feat_num = x.shape
    label=data.y[data.train_mask]
    train_x=torch.FloatTensor(x[data.train_mask]).to(device)
    train_y =torch.LongTensor(label).to(device)
    # model
    num_class=int(np.max(label)+1)
    model=DeepFm(feat_num,emb_size,num_class).to(device)
    # train
    model.fit(train_x,train_y,epochs)
    x=model.get_feat(torch.Tensor(x).to(device))
    return x





verbose=0
def op_sum(x,nbs):
    res=np.zeros_like(x)
    for u in tqdm(range(len(nbs)),disable=not verbose):
        nb=nbs[u]
        if len(nb!=0):
            res[u]=np.sum(x[nb],axis=0)
    return res
def op_mean(x,nbs):
    res=np.zeros_like(x)
    for u in tqdm(range(len(nbs)),disable=not verbose):
        nb=nbs[u]
        if len(nb!=0):
            res[u]=np.mean(x[nb],axis=0)
    return res
def op_max(x,nbs):
    res=np.zeros_like(x)
    for u in tqdm(range(len(nbs)),disable=not verbose):
        nb=nbs[u]
        if len(nb!=0):
            res[u]=np.max(x[nb],axis=0)
    return res
def op_min(x,nbs):
    res=np.zeros_like(x)
    for u in tqdm(range(len(nbs)),disable=not verbose):
        nb=nbs[u]
        if len(nb!=0):
            res[u]=np.min(x[nb],axis=0)
    return res
def op_prod(x,nbs):
    res=np.zeros_like(x)
    for u in tqdm(range(len(nbs)),disable=not verbose):
        nb=nbs[u]
        if len(nb!=0):
            res[u]=np.prod(x[nb],axis=0)
    return res


from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn import preprocessing
def sim_y_sel(datax,datay,valve=0.05):
    x=datax.T
    y=datay.T.reshape((1,-1))
    sims=np.abs(cos_sim(x,y))
    return np.where(sims>valve)[0]
def sim_y_sel_n(datax,datay,fixlen=200):
    x=datax.T
    y=datay.T.reshape((1,-1))
    sims=np.abs(cos_sim(x,y))
    return np.argsort(sims)[-fixlen:].reshape(-1)
def sim_y_gen(data,fixlen=200):
    if data.x.shape[1]<fixlen:
        return data.x
    sels=sim_y_sel_n(data.x[data.train_mask],data.y[data.train_mask],fixlen)
    return data.x[:,sels]
def sim_x_sel(datax,valve=0.9):
    x=datax.T
    y=np.mean(x,axis=0,keepdims=True)
    sims=np.abs(cos_sim(x,y))
    return np.where(sims<valve)[0]

import networkx as nx
import random
from tabulate import tabulate
import copy
from utils import setx

class Timer:
    def __init__(self,timebudget=None):
        self._timebudget=timebudget
        self._esti_time=0
        self._g_start=time.time()
    def start(self):
        self._start=time.time()
    def end(self):
        time_use=time.time()-self._start
        self._esti_time=(self._esti_time+time_use)/2
    def is_timeout(self):
        timebudget=self._timebudget
        if timebudget:
            timebudget=self._timebudget-(time.time()-self._g_start)
            if timebudget<self._esti_time:
                return True
        return False
class DeepGL:
    r"""
    reference to
        Rossi, R. A., Zhou, R., & Ahmed, N. K. (2020). 
        Deep Inductive Graph Representation Learning. 
        IEEE Transactions on Knowledge and Data Engineering, 32(3), 438–452. 
        https://doi.org/10.1109/TKDE.2018.2878247
    """
    def __init__(self,data):
        self._data=copy.deepcopy(data)
        self._num_nodes=data.num_nodes
        self._x=data.x
        self._edges=data.edge_index
        self._neighbours=[[] for _ in range(self._num_nodes)]
        for u,v in self._edges:
            self._neighbours[u].append(v)
        self._neighbours=[np.array(v) for v in self._neighbours]
        self._ops=[op_sum,op_mean,op_max,op_min]
        self._sim=cos_sim
#     @timeit
    def _gen(self,x):
        res=[]
        for i,op in enumerate(self._ops):
            res.append(op(x,self._neighbours))
        res=np.concatenate(res,axis=1)
        return res
    
#     @timeit
    def _sel(self,x,valve=0.1):
        x=x.T
        sims=np.abs(self._sim(x,x))
        adjm=sims>valve
        fg=nx.from_numpy_matrix(adjm)
        ccs=[list(_) for _ in nx.algorithms.components.connected_components(fg)]
        fsel_idxs=[random.choice(cc) for cc in ccs]
        return x[fsel_idxs].T
    @timeit
    def gen(self,max_epoch=3,fixlen=200,y_sel_func=gbdt_gen,timebudget=None):
        x=self._x.copy()
        gx=x.copy()
        verbs=[]
        data=self._data
        soft_timer=Timer(timebudget)
        for epoch in tqdm(range(max_epoch)):
            soft_timer.start()
            verb=[epoch,gx.shape[1]]
            gx=self._gen(gx)
            gx=scale(gx)
            verb.append(gx.shape[1])
            data = setx(data, gx)
            gx=y_sel_func(data,fixlen=fixlen)
            verb.append(gx.shape[1])   
            x=np.concatenate([x,gx],axis=1)
            verbs.append(verb)
            soft_timer.end()
            if soft_timer.is_timeout():
                break
        print(tabulate(verbs,headers='epoch origin after-gen after-sel'.split()))
        return x

@timeit
def deepgl_gen(data):
    dgl=DeepGL(data)
    return dgl.gen()
       
def gendata(data,funcs):
    print('before gen , feature num : {}'.format(data.x.shape))
    res=[torch.FloatTensor(func(data)) for func in funcs]
    print("every part of gen :")
    print(tabulate([[func.__name__,list(res[i].shape)[1]] for i,func in enumerate(funcs)],headers=['func name','feature num']))
    data.x=torch.cat(res,dim=1)
    print('after gen , feature num : {}'.format(data.x.shape))
    return data

"""
class AFE:
    def __init__(self,data):
        self._data=data
        print("data nodes : {} ,edges : {} ,features {} ".format(data.num_nodes,data.edge_index.shape[1],data.x.shape[1]))
    def tonumpy(self,data):
        data.x=data.x.numpy()
        data.y=data.y.numpy()
        data.train_mask=data.train_mask.numpy()
        data.edge_index=data.edge_index.numpy()
    def totensor(self,data):
        data.edge_index=torch.LongTensor(data.edge_index)
        data.train_mask=torch.BoolTensor(data.train_mask)
        data.y=torch.LongTensor(data.y)
        data.x=torch.FloatTensor(data.x)
    def zipdata(self,x):
        self._data.x=x
        return self._data
    def gen(self):
        data=self._data
        self.tonumpy(data)
        x= [
            gbdt_gen(self.zipdata(identity_fil(data)),fixlen=2000),
            scale(degree_gen(data)),
#             scale(pagerank_gen(data)),
#             scale(gdv_gen(data))
            ]
        data.x=np.concatenate(x,axis=1)
        dgl=DeepGL(data)
        data.x=dgl.gen(max_epoch=5,fixlen=200,y_sel_func=gbdt_gen)
        self.totensor(data)
        print("gen done features -> {}".format(data.x.shape[1]))
        return data
"""
