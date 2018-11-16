from sklearn.model_selection import train_test_split
'''
split dataset into training part and testing part

Use Breast Cancer Wisconsin (Diagnostic) Data Set as an example.

Parameters
----------
rnd: int   
    random seeds
    
data: pandas.DataFrame
    dataset
    
testsize: float
    test part proportion

Return
----------
X_train, y_train, X_test, y_test: pandas.DataFrame
    
'''


def train_test(rnd, data,testsize=0.2):
    # {'M':0, 'B':1}
    data_m = data[data['diagnosis'] == 0]
    data_b = data[data['diagnosis'] == 1]
    data_mx = data_m[self.feature]
    data_my = data_m['diagnosis']
    data_bx = data_b[self.feature]
    data_by = data_b['diagnosis']
    # malignant
    X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(
        data_mx, data_my, test_size=testsize, random_state=rnd)
    # benign
    X_btrain, X_btest, y_btrain, y_btest = train_test_split(
        data_bx, data_by, test_size=testsize, random_state=rnd)
    # aggregate
    X_train = X_mtrain.append(X_btrain)
    y_train = y_mtrain.append(y_btrain)
    X_test = X_mtest.append(X_btest)
    y_test = y_mtest.append(y_btest)

    return (X_train, y_train, X_test, y_test)