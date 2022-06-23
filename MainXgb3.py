# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 00:38:44 2021

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

pretrain()


###Data Processing
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



np.random.seed(seed)
max_steps = 3
total_reward = 0.
returns = []
label=[]
acc_tr=[]
reward_record_tr=[]
acc_val=[]
acc_test=[]
acc_val_best=float("-inf")
acc_test_best=float("-inf")
#reward_record_val=[]

##agent
agent_class = PDQNAgent

agent = agent_class(
                   observation_dim, action_dim,action_high,action_low,
                   batch_size=batch_size,
                   learning_rate_actor=learning_rate_actor,
                   learning_rate_actor_param=learning_rate_actor_param,
                   epsilon_steps=epsilon_steps,
                   gamma=gamma,
                   tau_actor=tau_actor,
                   tau_actor_param=tau_actor_param,
                   clip_grad=clip_grad,
                   indexed=indexed,
                   weighted=weighted,
                   average=average,
                   random_weighted=random_weighted,
                   initial_memory_threshold=initial_memory_threshold,
                   use_ornstein_noise=use_ornstein_noise,
                   replay_memory_size=replay_memory_size,
                   epsilon_final=epsilon_final,
                   inverting_gradients=inverting_gradients,
                   actor_kwargs={'hidden_layers': layers,
                                 'action_input_layer': action_input_layer,},
                   actor_param_kwargs={'hidden_layers': layers,
                                       'squashing_function': False,
                                       'output_layer_init_std': 0.0001,},
                   zero_index_gradients=zero_index_gradients,
                   norm_noise=norm_noise,
                   seed=seed)
agent_val=deepcopy(agent)
agent_val.epsilon=0
agent_val.trainpurpose=False
#控制测试不加noise
feature_selector = FeatureSelector(X_tr, args.gen_gamma) if args.gen_gamma > 0.0 else None
env=Env(observation_dim,action_dim,model,balance_factor,max_steps,feature_selector)
#env.build_feature_local(X_tr,Y_tr,n_neighbour)
env.build_meta_local(X_tr,Y_tr,n_neighbour)
count=0
previous_reward=float("-inf")

#episodes=5000
for k in range(episodes):
    episode_reward=0
    if(np.random.uniform(0,1)<0.5):
        i=np.random.randint(tr_size)
        x=X_tr[i,:]
        y=Y_tr[i]
    else:
        i=0
        x=np.random.randn(X_tr.shape[1])  
        I = model.predict(x.reshape(1,-1))
        y=I
    
    x = torch.FloatTensor(x)
    x_var = Variable(x,requires_grad=True).type(torch.FloatTensor)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        x_var = x_var.cuda()
    env.reset_local(x,y,i)
    
    #print("original",y)
    #env.reset(x,y)
    state = np.array(x, dtype=np.float32, copy=False)
    act_ava=env.get_act_ava()
    
    
    act, act_param, all_action_parameters = agent.act(state,act_ava)
    #print("act",act)
    #print("all_action_parameters",all_action_parameters[act])
    #act_param=min(act_param,args.action_bound)
    action = pad_action(act, act_param)
    for j in range(max_steps):
        #print("act",act)
        #print("all_action_parameters",all_action_parameters[act])
        ret = env.step(action)
        act_ava=env.get_act_ava()
        #print("act_ava",act_ava)
        #print("act_ava",act_ava)
        (next_state, steps), reward, terminal = ret
        next_state = np.array(next_state, dtype=np.float32, copy=False)
        next_act, next_act_param, next_all_action_parameters = agent.act(next_state,act_ava)
        #next_act_param=min(next_act_param,args.action_bound)
        next_action = pad_action(next_act, next_act_param)
        agent.step(state, (act, all_action_parameters), reward, next_state,
                   (next_act, next_all_action_parameters), terminal, steps)
        act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
        action = next_action
        #print("action",len(action)
        #print("")
        state = next_state
        episode_reward+=reward
        #print("terminal",terminal)
        #print("episode",episode_reward)
        if terminal:
            break
    if(terminal):
        label.append(1)
    else:
        label.append(0)
    agent.end_episode()
    returns.append(episode_reward)
    total_reward+=episode_reward
    if k % 100 == 0:
        agent_val.copy_models(agent.actor,agent.actor_param)
        acc_tr.append(np.array(label[-100:]).mean())
        reward_record_tr.append(np.array(returns[-100:]).mean())
        if(previous_reward>np.array(returns[-100:]).mean()+0.3):
            count+=1
        previous_reward=np.array(returns[-100:]).mean()
        if(count>=5 and k>30000):
            agent.learning_rate_actor=max(agent.learning_rate_actor/2,learning_rate_actor_min)
            agent.learning_rate_actor_param=max(agent.learning_rate_actor_param/2,learning_rate_actor_param_min)
            count=0
        np.savetxt("acc_tr0.08_step3.txt",acc_tr)
        np.savetxt("reward_tr0.08_step3.txt",reward_record_tr)
        if(k%300==0 and k>5600):
            #print("K",k)
            r_val=validate(agent_val,observation_dim,action_dim,balance_factor,max_steps,model,X_val,Y_val,val_size)
            acc_val.append(r_val)
            np.savetxt("acc_val0.08_step3.txt",acc_val)
            r_test=validate(agent_val,observation_dim,action_dim,balance_factor,max_steps,model,X_test,Y_test,val_size)
            if(r_test>acc_test_best):
                agent_val.save_models("best0.08_step3")
                acc_test_best=r_test
            if(r_test>0.76):
                agent_val.save_models("model0.08_step3"+str(np.round(r_test,2))+str(np.round(r_val,2)))
            acc_test.append(r_test)
            np.savetxt("acc_test0.08_step3.txt",acc_test)
            print('{0:5s} R:{1:.4f} r100:{2:.4f} rtr{3:.2f} rval{4:.2f} rtest{5:.2f}'.format(str(k), total_reward / (k + 1), np.array(returns[-100:]).mean(),np.array(label[-100:]).mean(),r_val,r_test))
        elif(k%100==0):
            print('{0:5s} R:{1:.4f} r100:{2:.4f} rtr{3:.2f}'.format(str(k), total_reward / (k + 1), np.array(returns[-100:]).mean(),np.array(label[-100:]).mean()))
        else:
            pass









