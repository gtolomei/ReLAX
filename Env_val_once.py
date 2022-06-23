# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:51:53 2021

@author: Albert
"""

from copy import deepcopy
from torch.autograd import Variable
import torch as torch
import numpy as np
from args import *
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

class Env_val():
    def __init__(self,observation_dim,action_dim,model,balance_factor,max_step,feature_selector):
        self.act_dim=action_dim
        self.obs_dim=observation_dim
        self.f=model
        self.timestep=0
        self.gamma=balance_factor
        self.max_step=max_step
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.feature_selector=feature_selector
        self.candidate=[]
    def reset(self,x,y):
        self.x_ori=deepcopy(x)
        self.y_ori=deepcopy(y)
        self.state=deepcopy(x)
        self.action_available=np.arange(self.act_dim)
        #print("action_ava",self.action_available)
        return self.state
    def reset_local(self,x,y,i):
        self.x_ori=deepcopy(x)
        self.y_ori=deepcopy(y)
        self.state=deepcopy(x)
        self.action_available=self.select_feature_local(i)
        #print("here")
        #print("action_ava",self.action_available)
        return self.state
    def select_feature_nn(self):
        x_var=Variable(deepcopy(self.x_ori),requires_grad=True).type(torch.FloatTensor)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            x_var = x_var.cuda()
        fs = self.f.forward(x_var)
        fs[0].backward(retain_graph=True)
        grads = x_var.grad.data.cpu().numpy().copy()
        top_feat_idx = np.argsort(grads)[::-1]
        top_feat_idx = self.feature_selector.select(top_feat_idx, args.num_feat)
        return top_feat_idx
    def build_feature_local(self,X,Y,n):
        nbrs = NearestNeighbors(n_neighbors=n, algorithm="kd_tree").fit(X)  
        distances, indices = nbrs.kneighbors(X) 
        X = torch.FloatTensor(deepcopy(X))
        label_matrix=np.zeros((Y.shape[0],n))
        for i in range(Y.shape[0]):
            for j in range(n):
                label_matrix[i,j]=Y[indices[i,j]]
        candidate=[]
        for i in range(X.shape[0]):
            clf = DecisionTreeClassifier(random_state=0,max_depth=6)
            X_loc=X[indices[i],:]
            Y_loc=label_matrix[indices[i],:]
            clf.fit(X_loc,Y_loc)
            importance=clf.feature_importances_
            importance=np.argsort(importance)[::-1]
            candidate.append(importance[:args.num_feat])
        self.candidate=candidate
    def select_feature_local(self,i):
        return self.candidate[i]
        
    def step(self,action):
        self.timestep+=1
        
        i=action[0]
        #x=action[1][i][0]
        x=min(action[1][i][0],args.action_bound)
        #print("i",i)
        #print("self.ava",self.action_available)
        if(i in self.action_available):
            #print("TRUE")
            t=np.arange(len(self.action_available))[np.where(self.action_available==i)][0]
            self.action_available=np.delete(self.action_available,t)
            self.state[i]=self.state[i]+x
            #diff=-self.gamma*np.abs(self.state[i]-self.x_ori[i])
            #print("diff",diff,"new",self.state[i],"ori",self.x_ori[i])
            label,change=self.reward()
            if(change):
                terminal=True
            else:
                terminal=False
            #print("REWARD",reward)
            return ((self.state,self.timestep),terminal)
            
    def reward(self):
        state = torch.from_numpy(deepcopy(np.array(self.state,dtype=np.float32))).to(self.device)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            state = state.cuda()
        f = self.f.forward(state
                        ).data.cpu().numpy().flatten()
        I = f.argsort()[::-1]
        fk_hat = I[0]
        if(fk_hat==self.y_ori):
            return 0, False
        else:
            return 1, True
    
    def get_act_ava(self):
        act_ava=deepcopy(self.action_available)
        return act_ava