## Importing important libraries

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

## Reading data 'Symptoms_with_QoL.csv" from local

LE_data = pd.read_csv("path_to_file/Symptoms_with_QoL.csv",sep=",",header=0)
print(LE_data.shape)
LE_data.head()

## Setting up KMeans object 'km' with 3 clusters
km = KMeans(n_clusters=3, init='k-means++',n_init=10)
km.fit(LE_data)


## Fitting the KMeans to predict the clusters
LE_pred = km.fit_predict(LE_data)
LE_pred

## Storing the Cluster Labels in a new column 'Cluster_QoL'
LE_data["Cluster_QoL"] = LE_pred
LE_data.head

## Sorting values by Cluster Labels and new data LE_data_cluster is created
LE_data_cluster = LE_data.sort_values(by=['Cluster_QoL'])
LE_data_cluster.head()

## PCA/ SVD Decomposition from 'n' dimensional to 2 dimensional
## Plotting Clusters in 2 dimensional space

import pylab as pl
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(LE_data)
LE_clustered_2d = pca.transform(LE_data)
pl.figure('QoL Combined Cluster Plot')
pl.scatter(LE_clustered_2d[:,0],LE_clustered_2d[:,1],c=km.labels_)
pl.show
LE_data_cluster.head()


## Save Clustered Data in CSV to a local path
LE_data_cluster.to_csv('save_to_path\\Clustered_QoL_Output.csv',sep=',')