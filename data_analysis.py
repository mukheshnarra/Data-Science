# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:40:11 2018

@author: MUKHESH
"""

import pandas as pd
iris_data=pd.read_csv('C:/Users/MUKHESH/Documents/PYTHON/sample_data_set.csv',na_values='NA')
#%matplotlib inline
import matplotlib as plt
import seaborn as sb
iris_data.loc[iris_data['class']=='versicolor','class']='Iris-versicolor'
iris_data.loc[iris_data['class']=='Iris-setossa','class']='Iris-setosa'
sb.pairplot(iris_data.dropna(),hue='class')
#print(iris_data['class'].unique())
iris_data=iris_data.loc[(iris_data['class']!='Iris-setosa')|(iris_data['sepal_width_cm']>=2.5)]
#iris_data.loc[iris_data['class']=='Iris-setosa','sepal_width_cm'].hist()
iris_data.loc[(iris_data['class']=='Iris-versicolor')&(iris_data['sepal_length_cm']<1.0),'sepal_length_cm']*=100
#iris_data.loc[iris_data['class']=='Iris-versicolor','sepal_length_cm'].hist()
#print(iris_data.loc[(iris_data['sepal_length_cm'].isnull())|(iris_data['sepal_width_cm'].isnull())|(iris_data['petal_length_cm'].isnull())|(iris_data['petal_width_cm'].isnull())])
#iris_data.loc[iris_data['class']=='Iris-setosa','petal_width_cm'].hist()
average_width=iris_data.loc[iris_data['class']=='Iris-setosa','petal_width_cm'].mean()
#mean mutation for putting values in place of missing values by the average value 
iris_data.loc[(iris_data['class']=='Iris-setosa')&(iris_data['petal_width_cm'].isnull()),'petal_width_cm']=average_width
#print(iris_data.loc[(iris_data['class']=='Iris-setosa')&(iris_data['petal_width_cm'].isnull()),'petal_width_cm'])
#iris_data.dropna(inplace=TRUE)
iris_data.to_csv('C:/Users/MUKHESH/Documents/PYTHON/iris_data_set.csv',index=False)
