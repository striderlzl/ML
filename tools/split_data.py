from sklearn.model_selection import train_test_split
import pandas as pd
'''
split dataset into label part and unlabel part(just for study, for example semi-supervised simulation.

select 50% of the positive class along with 50% of the negative class in the training set as labeled data and the 
rest as unlabelled data.Select them randomly

Use Breast Cancer Wisconsin (Diagnostic) Data Set as an example.

Parameters
----------
rnd: int   
    random seeds

X_train, y_train: pandas.DataFrame
    dataset
    
feature:list

unlabel_size: float
    test part proportion

Return
----------
X_train, y_train, X_unlabel, y_unlabel: pandas.DataFrame

'''


def split_data(X_train, y_train, rnd, feature, unlabel_size=0.5):
    # {'M':0, 'B':1}
    data = pd.concat([X_train, y_train], axis=1)
    data_m = data[data['diagnosis'] == 0]
    data_b = data[data['diagnosis'] == 1]
    data_mx = data_m[feature]
    data_my = data_m['diagnosis']
    data_bx = data_b[feature]
    data_by = data_b['diagnosis']
    # malignant
    X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(
        data_mx, data_my, test_size=unlabel_size, random_state=rnd)
    # benign
    X_btrain, X_btest, y_btrain, y_btest = train_test_split(
        data_bx, data_by, test_size=unlabel_size, random_state=rnd)
    # aggregate
    X_label = X_mtrain.append(X_btrain)
    y_label = y_mtrain.append(y_btrain)
    X_unlabel = X_mtest.append(X_btest)
    y_unlabel = y_mtest.append(y_btest)
    return (X_label, y_label, X_unlabel, y_unlabel)