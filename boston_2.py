# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:01:53 2018

@author: MUKHESH
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sd
import sklearn.learning_curve as curves
from sklearn.cross_validation import train_test_split,ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore",category=UserWarning,module="matplotlib")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
def Modellearning(x,y):
    cv=ShuffleSplit(x.shape[0],n_iter=10,test_size=0.2,random_state=0)
    train_size=np.rint(np.linspace(1,x.shape[0]*0.8-1,9)).astype(int)
    fig=plt.figure(figsize=(10,10))
    for k,depth in enumerate([1,3,6,10]):
        #print(k,depth)
        classifier=DecisionTreeRegressor(max_depth=depth)
        (sizes,train_scores,test_scores)=curves.learning_curve(classifier,x,y,train_sizes=train_size,cv=cv,scoring='r2')
        ax=plt.subplot(2,2,k+1)
        train_mean=np.mean(train_scores,axis=1)
        test_mean=np.mean(test_scores,axis=1)
        train_std=np.std(train_scores,axis=1)
        test_std=np.std(test_scores,axis=1)
        ax.plot(sizes,test_mean,'o-',color='g',label='testing scores')
        ax.plot(sizes,train_mean,'o-',color='r',label='training scores')
        ax.fill_between(sizes,train_mean-train_std,train_mean+train_std,color='r',alpha=0.8)
        ax.fill_between(sizes,test_mean-test_std,test_mean+test_std,color='g',alpha=0.8)
        ax.set_title('maxdepth= %s'%(depth))
        ax.set_xlim([0,x.shape[0]*0.8])
        ax.set_ylim([-0.05,1.05])
        ax.set_xlabel('sizes')
        ax.set_ylabel('scores')
    ax.legend(bbox_to_anchor=(1.05,2.05),loc='lower left',borderaxespad=0.)
    fig.suptitle('DecisionTreeClassifier',fontsize=16,color='g',y=1.05)
    fig.tight_layout()
    fig.show()
    return True
def Modelcomplexity(x,y):
    cv=ShuffleSplit(x.shape[0],n_iter=10,test_size=0.2,random_state=0)
    max_depth=np.arange(1,11)
    plt.figure(figsize=(10,10))
    classifier=DecisionTreeRegressor()
    (train_scores,test_scores)=curves.validation_curve(classifier,x,y,param_name="max_depth",param_range=max_depth,cv=cv,scoring='r2')
    train_mean=np.mean(train_scores,axis=1)
    test_mean=np.mean(test_scores,axis=1)
    train_std=np.std(train_scores,axis=1)
    test_std=np.std(test_scores,axis=1)
    plt.plot(max_depth,test_mean,'o-',color='g',label='testing scores')
    plt.plot(max_depth,train_mean,'o-',color='r',label='training scores')
    plt.fill_between(max_depth,train_mean-train_std,train_mean+train_std,color='r',alpha=0.8)
    plt.fill_between(max_depth,test_mean-test_std,test_mean+test_std,color='g',alpha=0.8)

    plt.xlim([0,11])
    plt.ylim([-0.05,1.05])
    plt.xlabel('maximum depth')
    plt.ylabel('scores')    
        #print(k,depth)

    plt.legend(loc='upper right',borderaxespad=0.)
    plt.subtitle('DecisionTreeClassifier',fontsize=16,color='g',y=1.05)
    plt.tight_layout()
    plt.show()
    return True
def predictprices(x,y,fitter,data):
    p=[]
    for k in range(10):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=10,random_state=k)
        #r=DecisionTreeRegressor()
        reg=fitter(x_train,y_train)
        pred=reg.predict([data[0]])[0]
        p.append(pred)
        print("the prices for {}  is {:,.2f}".format(k+1,pred))
    print("the range of prices is {:,.2f}".format(max(p)-min(p)))    