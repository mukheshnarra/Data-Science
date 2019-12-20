# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:50:52 2018

@author: MUKHESH
"""
import numpy as np
import pandas as pd
import matplotlib as plt
import untitled14 as vs
from IPython.display import display
data=pd.read_csv("C:/Users/MUKHESH/Documents/PYTHON/titanic_modified.csv")
#display(data.head())
#display(data.describe())
outcomes=data['Survived']
data=data.drop('Survived',axis=1)
#data['Fare']=data.Fare.str.replace(",","").astype(int)
def accuracy_test(truth,pred):
    display(truth)
    display(pred)
    if len(truth)==len(pred):
        return "the accuracy is {:.2f}".format((truth==pred).mean()*100)
    else:
        return "no. of outcomes is not predicted"
def prediction(data):
    predictions=[]
    for _,passenger in data.iterrows():
        if ((((passenger['Age']>=20)and(passenger['Age']<=35)))and((passenger['Class/Dept']=='1st Class')) and (passenger['Sex']=='female')):#and (passenger['Job'] ==('Personal Maid','Quartermaster','Lookout'))):#and(passenger['Joined']=='Cherbourg')):
            
            predictions.append(1)
        elif((passenger['Age']<=10) and passenger['Sex']=='male'):
            predictions.append(1)
        elif((passenger['Class/Dept']=='Deck')and((passenger['Age']>=20)and(passenger['Age']<=50))and (passenger['Sex']=='male')):
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)
vs.survival_stats(data,outcomes,'Sex',["Class/Dept == '1st_Class'"])
print(accuracy_test(outcomes,prediction(data)))
#print(accuracy_test(outcomes,pd.Series(np.ones(5,dtype=bool))))