from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
'''
Train an L1-penalized SVM to do the semi-supervised learning.
Find the unlabeled data point that is the farthest to the decision boundaryof the SVM.
Let the SVM label it (ignore its true label), and add it to the labeled data, and retrain the SVM. 
Continue this process until all unlabeled data are used.

Parameters
----------
X_label,y_label,X_unlabel,y_unlabel: pandas.DataFrame
    come from the dataset

C: float
    Penalty parameter C of the error term. Could be cross-validate and find the best C

Return
----------
clf: sklearn.svm.classes.SVC
    trained semi-supervised classifier 

new_sample: pandas.DataFrame
    X_label and X_unlabel
    
new_target: pandas.Dataframe
    y_label and y_unlabel(now labeled by semi-supervised classifer)
'''

def semi_supervised(X_label, y_label, X_unlabel, y_unlabel, C):
    clf = LinearSVC(penalty='l1', dual=False, C=C)
    clf.fit(X_label, y_label)

    ###  calculate distance  ###
    w = clf.coef_
    b = clf.intercept_
    ww = w * w.T
    distance = []
    for row in X_unlabel.iterrows():
        x = row[1]
        fx = np.dot(w, x) + b
        distance.append((abs(fx) / ww)[0][0])
    d = pd.DataFrame(distance, columns=['D'])
    d = d.set_index(X_unlabel.index)
    ###  self learning  ###
    for i in range(len(X_unlabel)):
        maxid = d[d['D'] == d.max()[0]].index[0]
        # if maxid in y_unlabel.index:
        d = d.drop([maxid])
        new_sample = X_label.append(X_unlabel.loc[maxid, :])
        new_target = y_label.append(pd.Series(y_unlabel[maxid]))
        X_unlabel = X_unlabel.drop([maxid])
        clf.fit(new_sample, new_target)
        w = clf.coef_
        b = clf.intercept_
        ww = w * w.T
        distance = []
        for row in X_unlabel.iterrows():
            x = row[1]
            fx = np.dot(w, x) + b
            distance.append((abs(fx) / ww)[0][0])
        d = pd.DataFrame(distance, columns=['D'])
        d = d.set_index(X_unlabel.index)
    return (clf, new_sample, new_target)