# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:46:51 2017

@author: sony vaio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df=pd.read_csv("C:\\Users\\sony vaio\\Documents\\Kaggle\\HR_comma_sep.csv")
columns_names=df.columns.tolist()
columns_names

df.shape
df.head()
df.corr()
# corr calculates the correlation between two columns
# positive value means that as the value of one column increases so does the valus of other column
# bigger the value of corr, stronger the relation between two columns

# Visualising correlation using seaborn library
correlation=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between columns')

#Doing some visualisation before moving onto PCA
df['sales'].unique()
sales=df.groupby('sales').sum()
sales

groupby_sales=df.groupby('sales').mean()
groupby_sales

# Principal Component Analysis(PCA)
df.head()

df_drop=df.drop(['sales','salary'],axis=1)

#now we need to bring 'left' to the front as it a label and not a feature
cols=df_drop.columns.tolist()
#bring left to front. first convert to list then cols.insert method
cols.insert(0,cols.pop(cols.index('left')))
df_drop=df_drop.reindex(columns=cols)

y=df_drop['left']
X=df_drop.drop(['left'],axis=1)

# Data Standardisation
# Standardization is shifting the distribution of each attribute to mean of zero and
# standard deviation of one
from sklearn.preprocessing import StandardScaler
X_std=StandardScaler().fit_transform(X)

# Compute Eigenvectors and Eigenvalues
# first we need a covariance matrix
mean_vec=np.mean(X_std,axis=0)
cov_mat=(X_std-mean_vec).T.dot((X_std-mean_vec))/(X_std.shape[0]-1)
cov_mat

cov_mat=np.cov(X_std.T) # we can also use np.cov to get the covariance matrix
#plot the covariance matrix
plt.figure(figsize=(8,8))
sns.heatmap(np.cov(X_std.T),vmax=1,square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between features')

# Eigen decomposition of the covariance matrix
eig_vals,eig_vecs=np.linalg.eig(cov_mat)

# Selecting Principal Components
# make a list of eigen vectors with their eigen values
import scipy

eig_pairs=list([(np.abs(eig_vals[i],eig_vecs[:,i]) for i in range(len(eig_vals)))])

#sort the eigen pairs in descending order
eig_pairs.sort(key=lambda x: x[0],reverse=True)
eig_pairs=list(eig_pairs)
for i in eig_pairs:
    print(i[0])

# Explained Variance
# how many principal components are we going to choose
tot=sum(eig_vals)
var_exp=[(i/tot)*100 for i in sorted(eig_vals,reverse=True)]
with plt.style.context('dark_background'):
    plt.figure(figsize=(6,4))
    plt.bar(range(7),var_exp,alpha=0.5,align='center',label='Individual Explained Variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.legend(loc='best')
    plt.tight_layout()
# the plot shows that we can only drop the 7th component because it has less than 10% of var

## Projection matrix
# this is used to transform the human resources analytics data onto new feautre subspace
# suppose only 1st and 2nd principal components share the maximum amount of data. 
# we drop other components
matrix_w= np.hstack((eig_pairs[0][1].reshape(7,1),eig_pairs[1][1].reshape(7,1)))
matrix_w

# projection onto new feature space
# Y=X x W
Y=X_std.dot(matrix_w)
Y

### IF WE DID THE SAME IN SCIKITLEARN
from sklearn.decomposition import PCA
pca=PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.ylabel('Cumulative Explained Variance')
plt.xlabel('Number of Components')

# plto shows 90% variance by th 6 components, drop the 7th
from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(X_std)