# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 19:36:19 2018

@author: MUKHESH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import seaborn as sd

import boston_2 as vs
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score,make_scorer
from sklearn.cross_validation import train_test_split,ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as tree
data=pd.read_csv("D:/boston_housing-master/housing.csv")
prices=data["MEDV"]
features=data.drop("MEDV",axis=1)
#print("the data points is {} and the variables is {}".format(*data.shape))
mine=np.min(prices)
maxe=np.max(prices)
std=np.std(prices)
median=np.median(prices)
mean=np.mean(prices)
#print("the mean_prices is {:,.2f}".format(mean))
#print("the median_prices is {:,.2f}".format(median))
#print("the minimum_prices is {:,.2f}".format(mine))
#print("the maximum_prices is {:,.2f}".format(maxe))
#print("the standard deviation_prices is {:,.2f}".format(std))
def performance_metric(true,predict):
    score=r2_score(true,predict)
    return score
#the below gives the total data points and variables such as there are only 2 indexes
#print(features.shape[])
#print(performance_metric([2.0,3.4,5.6,1.2,3.6],[1.5,2.8,5.0,1.1,3.0]))
#d=[]
#
#for i in range(1000):
#    (training_input,testing_input,training_class,testing_class)=train_test_split(features,prices,test_size=0.2,random_state=0)
#    decision_tree_classifier=DecisionTreeClassifier()
#    decision_tree_classifier.fit(training_input,training_class)
#    d.append(decision_tree_classifier.score(testing_input,testing_class))
#sd.distplot(d)
(train_in,test_in,train_pr,test_pr)=train_test_split(features,prices,train_size=0.8,random_state=0)
#vs.Modellearning(features,prices)
#vs.Modelcomplexity(features,prices)

def filter_model(x,y):
    regressor=DecisionTreeRegressor()
    param={'max_depth':np.arange(1,11)}
    cv=ShuffleSplit(x.shape[0],n_iter=10,test_size=0.2,random_state=0)
    score=make_scorer(performance_metric)
    grid=GridSearchCV(regressor,param_grid=param,scoring=score,cv=cv)
    grid=grid.fit(x,y)
    return grid.best_estimator_
reg=filter_model(features,prices)
#the below function will retrive the params of the column max_depth
#print("the max_depth of optimal is {}".format(reg.get_params()['max_depth']))
client_data = [[5, 17, 15],
               [4, 32, 22], 
               [8, 3, 12]]
#for i,price in enumerate(reg.predict(client_data)):
#    print("the client{} the corresponding price is {:,.2f}".format(i,price))
#the below function is for the checking the model is working better or not
#vs.predictprices(features,prices,filter_model,client_data)
with open('C:/Users/MUKHESH/Documents/PYTHON/boston_housing.dot','w') as open_file:
    open_file=tree.export_graphviz(reg,out_file=open_file)