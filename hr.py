# -*- coding: utf-8 -*-
"""
Created on Mon May 15 03:58:53 2017

@author: sony vaio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\sony vaio\\Documents\\Kaggle\\HR_comma_sep.csv")

data.columns
data.head(5)

data['left'].unique()
data.describe()
data.groupby('left').size()
data.shape #14999,10

data.plot(kind='box',subplots=True,layout=(8,8),sharex=False,sharey=False)
plt.show()

data.hist()
plt.show()

from pandas.tools.plotting import scatter_matrix
scatter_matrix(data,figsize=(6,6))
plt.show()
data['sales']=data['sales'].map({'sales':1,'accounting':2,'hr':3,'technical':4,'support':5,'management':6,'IT':7,'product_mng':8,'marketing':9,'RandD':10}).astype(int)
data['salary']=data['salary'].map({'high':1,'medium':2,'low':3}).astype(int)
data.apply(pd.to_numeric,errors='ignore')

from sklearn.cross_validation import train_test_split
X=data
y=data['left']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)

from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier()
clf.fit(Xtrain,ytrain)
clf.score(Xtest,ytest)





