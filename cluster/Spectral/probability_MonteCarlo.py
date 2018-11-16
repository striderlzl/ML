from sklearn.cluster import SpectralClustering
from functools import reduce
from ML.tools.train_test import train_test
import pandas as pd

'''
one way to calculate the probability of the output from Spectral cluster. Run many times Spectral cluster and label the datapoint
use their major result. Every runs, initialize the algoritm randomly.

Parameters
----------
rnd: int   
    random seeds

data: pandas.DataFrame
    dataset

M: int(default:100)
    run times

Return
----------
y_pred_prob: list  
    probability of prediction

'''

def Spectral_Monte(rnd,data,M=100):
    dict1 = {}
    y_pred_prob = []
    for i in range(M):
        print(i)
        X_train2, y_train2, X_test2, y_test2 = train_test(rnd,data)
        spectral = SpectralClustering(n_clusters=2,
                                      assign_labels="kmeans", gamma=50, random_state=i).fit(X_train2)

        # y_pred = Spectral_label(X_train2,spectral,y_train2)

        # train part
        y_pred1 = spectral.fit_predict(X_train2)
        y_pred1 = pd.Series(y_pred1)
        y_pred1 = y_pred1.replace(to_replace=[0, 1], value=[1, 0])
        if i == 0:
            for i in range(y_pred1.shape[0]):
                dict1[str(i)] = []
        for i in range(len(list(y_pred1))):
            dict1[str(i)].append(list(y_pred1)[i])

    for i in range(len(dict1)):
        y_pred_prob.append(reduce(lambda x, y: x + y, dict1[str(i)]) / M)
    return(y_pred_prob)