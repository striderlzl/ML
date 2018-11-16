from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

'''
Train a SVM with a pool of m randomly selected data points from the training set using linear kernel and L1 penalty. 
Select the parameters of the SVM with 10-fold cross validation. Choose the m closest data points in the training set 
to the hyperplane of the SVM and add them to the pool. Train a new SVM using the pool. Repeat the process until all 
training data is used.

Parameters
----------
X_rest, X_test, y_rest, y_test: pandas.DataFrame   
    come from dataset

m: int
    in every runs the number random selected data points from training set

c_range: ndarray or list
    svm parameters

Return
----------
dict1: dict 
    store every runs' classifier's score.

clf:
    the final classifier

'''

def active_learning(X_rest, X_test, y_rest, y_test, m, c_range = 10. ** np.arange(-5, 4)):
    dict1 = {}

    n = (X_rest.shape[0] // m) + 1

    for i in range(n):
        dict1[str(i)] = []

    ### initial with 10 randomly points ###

    X_rest, X_pool, y_rest, y_pool = train_test_split(
        X_rest, y_rest, test_size=m / X_rest.shape[0], random_state=1)

    for i in range(n):
        C_range = 10. ** np.arange(-5, 4)
        param_grid = dict(C=c_range)

        ### append new data into pool ###

        if i != 0:
            # print(i)
            X_pool = X_pool_old.append(X_pool)
            y_pool = y_pool_old.append(y_pool)
        grid = GridSearchCV(LinearSVC(penalty='l1', dual=False, max_iter=5000)
                            , param_grid=param_grid, cv=KFold(10))
        grid.fit(X_pool, y_pool)
        clf = grid.best_estimator_
        clf.fit(X_pool, y_pool)
        # score.append(clf.score(X_test,y_test))

        X_pool_old = X_pool
        y_pool_old = y_pool
        if i != n-1:

            ### calculate distance and m closet ###

            w = clf.coef_
            b = clf.intercept_
            ww = w * w.T
            distance = []
            for row in X_rest.iloc[:, 0:3].iterrows():
                x = row[1]
                fx = np.dot(w, x) + b
                distance.append((abs(fx) / ww)[0][0])
            d = pd.DataFrame(distance, columns=['D'])
            d = d.set_index(X_rest.index)

            X_rest = pd.concat([X_rest.iloc[:, 0:3], d], axis=1)
            X_pool = X_rest.nsmallest(10, 'D').iloc[:, 0:3]
            y_pool = y_rest.loc[X_pool.index]

            ### remove 10 data from X_rest ###

            for index in X_pool.index:
                X_rest = X_rest.drop([index])
        dict1[str(i)].append(clf.score(X_test, y_test))
    return (dict1,clf)
