# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:21:28 2021

@author: Albert
"""

from args import *
from src.utils import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib
import warnings

def pretrain():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
            args.csv, args.seed, scaler=args.pre_scaler)
    
    train_X=train_data.getX()
    print(train_X.shape)
    train_y=train_data.gety()
    val_X=val_data.getX()
    val_y=val_data.gety()
    test_X=test_data.getX()
    test_y=test_data.gety()
    clf = XGBClassifier(max_depth=args.max_depth,n_estimators=args.n_estimators)#,learning_rate=0.8)
    #scores = cross_val_score(clf,train_X,train_y, cv=10)
    clf.fit(train_X,train_y)
    acc_train=clf.score(train_X,train_y)
    acc_test=clf.score(test_X,test_y)
    acc_val=clf.score(val_X,val_y)
    print("acc_train",acc_train)
    print("acc_test",acc_test)
    print("acc_val",acc_val)
    joblib.dump(clf, args.model_temp_path)

#pretrain()