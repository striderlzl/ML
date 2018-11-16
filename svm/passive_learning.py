from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

'''
Train a SVM with a pool of m randomly selected data points from the trainingset using linear kernel and L1 penalty. 
Select the penalty parameter using 10-fold cross validation. Repeat this process by adding m other randomly selected 
data points to the pool, until use all the data points.

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


def passive_learning(X_rest, X_test, y_rest, y_test, m, c_range = 10. ** np.arange(-5, 4)):

    n = (X_rest.shape[0]//m)+1

    dict1 = {}
    for i in range(n):
        dict1[str(i)] = []

    for i in range(n):
        if i == n-1:
            X_pool = X_rest
            y_pool = y_rest
        else:
            X_rest, X_pool, y_rest, y_pool = train_test_split(
                X_rest, y_rest, test_size=m / X_rest.shape[0], random_state=1)
        # print(X_pool.shape[0])
        param_grid = dict(C=c_range)
        if i != 0:
            # print(i)
            X_pool = X_pool_old.append(X_pool)
            y_pool = y_pool_old.append(y_pool)

        ### find best param ###
        grid = GridSearchCV(LinearSVC(penalty='l1', dual=False, max_iter=5000)
                            , param_grid=param_grid, cv=5)
        grid.fit(X_pool, y_pool)

        ### fit model ###
        clf = grid.best_estimator_
        clf.fit(X_pool, y_pool)

        dict1[str(i)].append(clf.score(X_test, y_test))
        X_pool_old = X_pool
        y_pool_old = y_pool
    return (dict1,clf)