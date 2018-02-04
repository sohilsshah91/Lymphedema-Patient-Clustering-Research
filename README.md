# Lymphedema_Prediction_Clustering

This project is a research focused project highlighting application of unsupervised Machine learning techniques in predicting Lymphedema disease without having knowledge about the symptoms or fields.

Libraries Used: sklearn, pandas, numpy, matplotlib, sklearn. cluster KMeans and sklearn
decomposition PCA.

We have used the K-Means Clustering Technique to make clusters on the cleaned data. This is
an unsupervised learning technique. Other technique that we read about is Hierarchical
Clustering and Spectral Clustering. We need more data to make it learn the cluster as it does
not take any input k and automatically forms clusters on its own.

We followed the following steps while modeling for each of the files. The details are entailed in the code.

1. Reading the clean input file for all symptoms 
2. Used K-Means with 3 clusters and kmeans++ as type of k-means algorithm
3. Created a Clusters columns for getting the clustered labels in the actual data
4. Perform Principal Component Analysis (PCA) to convert n dimensions into 2 dimensions
5. Plotted the scatter plot on 2d using pyplot
6. Output into a csv file with the cluster labels in the code
