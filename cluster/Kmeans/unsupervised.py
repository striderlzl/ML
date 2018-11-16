import numpy as np
import pandas as pd

'''
Compute the centers of the two clusters and find the closest 30 data points to each center. Read the true
labels of those 30 data points and take a majority poll within them. The majority poll becomes the label
predicted by k-means for the members of each cluster.

Use Breast Cancer Wisconsin (Diagnostic) Data Set as an example.

Parameters
----------
X_train, y_train: pandas.DataFrame
    come from the dataset

Kmeans: sklearn.cluster.KMeans  
    trained Kmeans
    

Return
----------
clust_labels:pandas.Series
    Read the true labels and take a majority poll which becomes the label
    predicted by k-means for the members of each cluster

'''


def keamns_label(X_train, kmeans, y_train):
    dist_dict = {'a': [], 'b': []}
    # dist = np.linalg.norm(a-b)
    for index, row in X_train.iterrows():
        dist_dict['a'].append(np.linalg.norm(kmeans.cluster_centers_[0] - row))
        dist_dict['b'].append(np.linalg.norm(kmeans.cluster_centers_[1] - row))
    clust_labels = kmeans.predict(X_train)
    # a
    dist_a = pd.Series(dist_dict['a'])
    dist_a = pd.DataFrame(dist_dict['a'], columns=['a'])
    dist_a = dist_a.set_index(X_train.index)

    # b
    dist_b = pd.Series(dist_dict['b'])
    dist_b = pd.DataFrame(dist_dict['b'], columns=['b'])
    dist_b = dist_b.set_index(X_train.index)

    dist_a = dist_a.sort_values(by=['a'])
    ###  {'M':0, 'B':1}  ###
    k = 0
    B = 0
    M = 0
    for index in dist_a.iterrows():
        k = k + 1
        if k == 30:
            break
        if y_train[index[0]] == 1:
            B = B + 1
        else:
            M = M + 1
    clust_labels = pd.Series(clust_labels)
    if M < B:
        clust_labels = clust_labels.replace(to_replace=[0, 1], value=[1, 0])

    return (clust_labels)