# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 01:39:06 2022

@author: Albert
"""
from args import *
from src.trainer import *
from src.utils import *
from src.selector import *
from Pre_train import pretrain
from agents.pdqn_bound import PDQNAgent
from agents.pdqn_split import SplitPDQNAgent
from agents.pdqn_multipass import MultiPassPDQNAgent
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from copy import deepcopy

#from Env import Env
import numpy as np
from Env18 import Env
from Env_val_once import Env_val
from tqdm import tqdm
from Validate import validate
import joblib


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32) for i in range(action_dim)]
    params[act][:] = act_param
    return (act, params)

seed=np.random.randint(100000)
evaluation_episodes=1000
episodes=650000
batch_size=64
gamma=0.2
inverting_gradients=True,            
initial_memory_threshold=18250
use_ornstein_noise=False         
replay_memory_size=80000
epsilon_steps=60000
epsilon_final=0.35
tau_actor=0.1
tau_actor_param=0.0005
learning_rate_actor=1e-3
learning_rate_actor_param=1e-3
learning_rate_actor_min=1e-5
learning_rate_actor_param_min=1e-5


scale_actions=True
initialise_params=True
clip_grad=10.
split=False
multipass=True
indexed=False
weighted=False
average=False
random_weighted=False
zero_index_gradients=False
action_input_layer=0
layers=[128]
save_freq=0
save_dir="results/platform"
render_freq=100
save_frames=False
visualise=True
title="PDDQN"
balance_factor=0.36
n_neighbour=20
norm_noise=True

#pretrain()


###Data Processing

"""
scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)

model=joblib.load(args.model_temp_path)
num_action = train_data.getX().shape[1]
bound_min, bound_max, bound_type = get_constraints(train_data.getX())

from agents.pdqn_bound import PDQNAgent
observation_dim=train_data.getX().shape[1]
action_dim=train_data.getX().shape[1]
action_high=bound_max
action_low=bound_min
X_tr=train_data.getX()
Y_tr=model.predict(X_tr)
tr_size=X_tr.shape[0]
X_val=val_data.getX()
Y_val=model.predict(X_val)
val_size=X_val.shape[0]
X_test=test_data.getX()
Y_test=model.predict(X_test)
test_size=X_test.shape[0]

X_tr=np.vstack((X_tr,np.vstack((X_val,X_test))))
Y_tr=np.concatenate((Y_tr,np.concatenate((Y_val,Y_test))))
tr_size=X_tr.shape[0]

X_tr1=X_tr[Y_tr==1]
X_test1=X_test[Y_test==1]
X1=np.vstack([X_tr1,X_test1])
Y1=np.array([1]*X1.shape[0])

"""
index=6
x=np.expand_dims(X1[index,:],axis=0)
x[0,11]-=0.6
x[0,2]+=1.15
print(model.predict(x))




