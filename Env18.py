# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:53:44 2021

@author: Albert
"""
from copy import deepcopy
from torch.autograd import Variable
import torch as torch
import numpy as np
from args import *
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import warnings

class Env():
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
        self.gap=0
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
        prob=np.squeeze(self.f.predict_proba(self.state.reshape(1,-1)))
        #print("ori_prob",prob)
        self.gap=np.abs(prob[0]-prob[1])
        #print("here")
        #print("action_ava",self.action_available)
        return self.state
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
            clf = DecisionTreeClassifier(random_state=0,max_depth=10)
            X_loc=X[indices[i],:]
            Y_loc=label_matrix[indices[i],:]
            clf.fit(X_loc,Y_loc)
            importance=clf.feature_importances_
            importance=np.argsort(importance)[::-1]
            candidate.append(importance[:args.num_feat])
        self.candidate=candidate
    def build_meta_local(self,X,Y,n):
        candidate=[]
        common=np.array([1,2,9,10,11,12,13,15])
        for i in range(X.shape[0]):   
            candidate.append(common)
        self.candidate=candidate
    def select_feature_local(self,i):
        #print(self.candidate)
        return self.candidate[i]
        
    def step(self,action):
        self.timestep+=1
        
        i=action[0]
        #print("x",action[1][i][0])
        x=min(action[1][i][0],args.action_bound)
        #print("i",i)
        #print("self.ava",self.action_available)
        if(i in self.action_available):
            #print("TRUE")
            t=np.arange(len(self.action_available))[np.where(self.action_available==i)][0]
            self.action_available=np.delete(self.action_available,t)
            state_last=deepcopy(self.state[i])
            self.state[i]=self.state[i]+x
            diff=self.gamma*(np.abs(state_last-self.x_ori[i])-np.abs(self.state[i]-self.x_ori[i]))
            #print("diff",diff,"new",self.state[i],"ori",self.x_ori[i])
            label,change=self.reward_shape()
            #print("label")
            reward=label*18+diff
            #print("reward",reward)
            if(change):
                terminal=True
            else:
                terminal=False
            #print("REWARD",reward)
            return ((self.state,self.timestep),reward,terminal)
            
    def reward(self):
        I = self.f.predict(self.state)
        fk_hat = I
        if(fk_hat==self.y_ori):
            return 0, False
        else:
            return 1, True
    def reward_shape(self):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        I = self.f.predict(self.state.reshape(1,-1))
        fk_hat = I
        #print("newlabel",fk_hat)
        if(fk_hat!=self.y_ori):
            return 1, True
        else:
            prob=np.squeeze(self.f.predict_proba(self.state.reshape(1,-1)))
            gap_t=np.abs(prob[0]-prob[1])
            r=self.gap-gap_t
            self.gap=gap_t
            #print("prob",prob)
            return r,False
            
        
        
    def get_act_ava(self):
        act_ava=deepcopy(self.action_available)
        return act_ava
    
    
        
    