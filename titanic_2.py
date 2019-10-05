# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 13:11:32 2018

@author: MUKHESH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=UserWarning,module="matplotlib")
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','inline')
#%matplotlib inline
def filter_condition(data,condition):
    field, op, value = condition.split(" ")
    
    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")
        value=value.replace("_"," ")
        #print(value)
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    #print(matches.head())
    return data[matches].reset_index(drop=True)

def survival_stat(data,outcomes,key,filters=[]):
    if key not in data.columns.values:
        print("entered wrong features try else")
        return False
    all_data=pd.concat([data,outcomes],axis=1)
    for condition in filters:
        all_data=filter_condition(all_data,condition)    
    all_data=all_data[[key,'Survived']]
    plt.figure(figsize=(10,10))
    if (key=='Age') or (key=='Fare'):
        if (key=='Age'):
            all_data=all_data[~np.isnan(all_data[key])]
        else:
            #all_data=all_data[~all_data['Fare'].isnull()]
            #all_data['Fare']=all_data['Fare'].apply(pd.to_numeric(errors='coerce'))
            #pd.to_numeric(all_data['Fare'],errors='coerce')
            all_data['Fare']=all_data.Fare.str.replace(",","").astype(int)
        #tha above and below lines are for formatting the string to int
        #print(all_data.dtypes)#this is to used to know the type of dataframe columns
        mine=all_data[key].min()
        maxe=all_data[key].max()
        value_range=maxe-mine
        if(key=='Age'):
            bins=np.arange(0,maxe+10,10)
        else:
            bins=np.arange(0,maxe+100,100)
        non_survival=all_data[all_data['Survived']==0][key].reset_index(drop=True)
        survival=all_data[all_data['Survived']==1][key].reset_index(drop=True)
        plt.hist(non_survival,bins=bins,alpha=0.6,color='red',label='not_survived')
        plt.hist(survival,bins=bins,alpha=0.6,color='green',label='survived')
        plt.xlim(0,bins.max())
        plt.legend(framealpha=0.8)
        
    else:
        values=[]
        #if(key=='Class/Dept'):
        value=all_data[key].unique()
        for v in value:
            values.append(v)
        #values=values[170:174]    
        #print(len(values))
        frame=pd.DataFrame(index=(np.arange(len(values))),columns=[key,'survived','Nsurvived'])
        for i,value in enumerate(values):
           # print(len(all_data[(all_data['Survived']==1)&(all_data[key]==value)]))
            frame.loc[i]=[value,len(all_data[(all_data['Survived']==1)&(all_data[key]==value)]),len(all_data[(all_data['Survived']==0) &(all_data[key]==value)])]
        bar_width=0.4
        plt.figure(figsize=(10,10))
        for i in np.arange(len(frame)):
            no=plt.bar(i-bar_width,frame.loc[i]['Nsurvived'],width=bar_width,color='r')
            yes=plt.bar(i,frame.loc[i]['survived'],width=bar_width,color='g')
            plt.xticks(np.arange(len(frame)),values)
            plt.legend((no[0],yes[0]),('not survive','survived'),framealpha=0.8)
    plt.xlabel(key)
    plt.ylabel('no.of.passengers')
    plt.show()        
            
            