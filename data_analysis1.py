# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:07:44 2018

@author: MUKHESH
"""
import pandas as pd
import seaborn as sd
import matplotlib.pyplot as plt
iris_data_clean=pd.read_csv('C:/Users/MUKHESH/Documents/PYTHON/iris_data_set.csv')
#sd.pairplot(iris_data_clean,hue='class')
#testing
#assert len(iris_data_clean['class'].unique())==3
#assert iris_data_clean.loc[(iris_data_clean['class']=='Iris-versicolor'),'sepal_width_cm'].min()>=2.5
#assert len(iris_data_clean.loc[iris_data_clean['sepal_width_cm'].isnull()|iris_data_clean['sepal_length_cm'].isnull()|iris_data_clean['petal_width_cm'].isnull()|iris_data_clean['petal_length_cm'].isnull()])==0
#exploratory analysis is such that which will go deeper into data
#sd.pairplot(iris_data_clean)
#above one says that there is no classification based on species
#sd.pairplot(iris_data_clean,hue='class')
#above one says iris-setosaof petal measurements differ so we can find different species
#box plots are the plots to represent the quartiles
#range of data=largevaluedata-smallvaluedata then median=range/2 then box 1st limit=(smallvaluedata+median)/2 and 2nd limit=(largevaluedata+median)/2
#violin plot same as box plot where it plots as density of data
#plt.figure(figsize=(10,10))
#for column_index,column in enumerate(iris_data_clean):
#    if column=='class':
#        continue
#    plt.subplot(2,2,column_index+1)
#    sd.violinplot(x='class',y=column,data=iris_data_clean)
#making training set and testing set for model chossing descisin classifier model for easy
all_values=iris_data_clean[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']]
all_classes=iris_data_clean['class'].values
#print(len(all_values))
#print(all_classes[5])
#print(all_values[:5])
from sklearn.cross_validation import train_test_split
(training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_values,all_classes,train_size=0.75,random_state=1)
from sklearn.tree import DecisionTreeClassifier

#creating classifier
# Create the classifier
#model_accuracies=[]
#for rep in range(1000):
#    (training_inputs,testing_inputs,training_classes,testing_classes)=train_test_split(all_values,all_classes,train_size=0.75)#,random_state=1)
#    decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
#   decision_tree_classifier.fit(training_inputs, training_classes)

# Validate the classifier on the testing set using classification accuracy
#   de=decision_tree_classifier.score(testing_inputs, testing_classes)
#print(de)
#   model_accuracies.append(de)
#sd.distplot(model_accuracies)
#where above making the overfit model the data
#to avoid overfitting where we perform k-fold cross validation
#best one is 10 fold cross_validation
#StratifiedKFold is a iterator which will  create the object for 10 folds
from sklearn.cross_validation import StratifiedKFold
import numpy as np
#def pvplot(cv,n_sample):
#    #print(cv)
#    makes=[]
#    for train,test in cv:
#        #print(train)
#        #print(test)
#        make=np.zeros(n_sample,dtype=bool)
#        make[test]=1
#        makes.append(make)
#    plt.figure(figsize=(10,10))
#    plt.imshow(makes,interpolation='none')
#    plt.ylabel('fold')
#    plt.xlabel('row#')
#    #mask=[]
#    #for l in cv:
#     #   mask=np.zero
#pvplot(StratifiedKFold(all_classes,n_folds=10),len(all_classes))  
#where parameters to tune the performance of decisin tree classifier
classifier=DecisionTreeClassifier()
#where cross_val_score will evaluate as the strtified k-fold where it makes the 
from sklearn.cross_validation import cross_val_score
#de=cross_val_score(classifier,all_values,all_classes,cv=10)
#kde is to avoid the curve
#sd.distplot(de,kde=False)
#plt.title('average:{}'.format(np.mean(de)))
#for performance tuning wee use the gridsearchcv which collect the range of parameters and apply the combination and give the result
from sklearn.grid_search import GridSearchCV
parameters={'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':[1,2,3,4,5],'max_features':[1,2,3,4]}
cv1=StratifiedKFold(all_classes,n_folds=10)
grid=GridSearchCV(classifier,param_grid=parameters,cv=cv1)
grid.fit(all_values,all_classes)
#print('best_score:{}'.format(grid.best_score_))
#print('best_parameter:{}'.format(grid.best_params_))
#we can visulatilize which features are workable
#grid_virtualization=[]
#for grid_search in grid.grid_scores_:
    #print(grid_search)
    #grid_virtualization.append(grid_search.mean_validation_score)
#grid_virtualization=np.array(grid_virtualization)
#grid_virtualization.shape=(5,4)
#sd.heatmap(grid_virtualization,cmap='Blues')
#plt.xticks(np.arrange(4)+0.5,grid.param_grid['max_features'])
#plt.yticks(np.arrange(5)+0.5,grid.param_grid['max_depth'][::-1])
#here features corresponding to measurments and depth coresponding to all colums
#now we choose th bets estimator
classifier=grid.best_estimator_
#print(classifier)    
#we can do the decision tree classifier working by the graphviz 
import sklearn.tree as tree
from sklearn.externals.six import StringIO
with open('C:/Users/MUKHESH/Documents/PYTHON/iris-decision-tree-model.dot','w') as out_file:
    out_file=tree.export_graphviz(classifier,out_file=out_file)
final=cross_val_score(classifier,all_values,all_classes,cv=10) 
#sd.boxplot(final)
#sd.stripplot(final,jitter=True,color='white')
#where decision tree classifiers are prone to the overfitting
#so random_forest_classsifiers are used is same as decision tree but bunch of it
from sklearn.ensemble import RandomForestClassifier
random_classifier=RandomForestClassifier()
parameters={'n_estimators':[5,10,15,20],'max_features':[1,2,3,4],'criterion':['gini','entropy'],'warm_start':[True,False]}
cross=StratifiedKFold(all_classes,n_folds=10)
grid=GridSearchCV(random_classifier,param_grid=parameters,cv=cross)
grid.fit(all_values,all_classes)
print(grid.best_estimator_)
random_classifier=grid.best_estimator_
final1=cross_val_score(random_classifier,all_values,all_classes,cv=10)
#now we are going to compare the random forest and decision tree
#by creating a data frameby pandas
df=pd.DataFrame({'accuracy':final,'classifier':['decision']*10})
rf=pd.DataFrame({'accuracy':final1,'classifier':['random']*10})
both=rf.append(df)
sd.boxplot(x='classifier',y='accuracy',data=both)
sd.stripplot(x='classifier',y='accuracy',data=both,jitter=True,color='white')
#where to get the python script for this follow below steps
#%install_ext https://raw.githubusercontent.com/rasbt/watermark/master/watermark.py
#%load_ext watermark
#%watermark -a 'Randal S. Olson' -nmv --packages numpy,pandas,scikit-learn,matplotlib,Seaborn
#for showing the prediction
#print(testing_inputs[:10])
random_classifier.fit(training_inputs,training_classes)
for inputs,prediction,actual in zip(testing_inputs[:10],random_classifier.predict(testing_inputs),testing_classes[:10]):
    print('{}\t-->{}\t-->{}'.format(inputs,prediction,actual))