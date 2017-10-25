#%%
import numpy as np
import sys
sys.path.insert(0, '../utils/')
from sklearn import svm
import datasets
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import normalize
#%%
xtr, label_train, xts, label_test = datasets.load_cifar10_dataset('../cifar-10-batches-py/', mode='supervised')
parameters = { 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': 
              [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}
del xtr, xts
label_test = np.array(label_test)
#%% DAE
#def dae_svm():
dae_train = np.load('data/dae/dae_svm-dae_svm_train.npy')
dae_test = np.load('data/dae/dae_svm-dae_svm_test.npy')

svr=svm.SVC()
svm_dae = GridSearchCV(svr, parameters)
svr.fit(dae_train, label_train)
predicted_dae = svm_dae.predict(dae_test)
daeacc = acc(label_test, predicted_dae)


#%% CNN
def del_cols():
    tr = np.load('data/cnn/train_cnn.npy')
    ts = np.load('data/cnn/test_cnn.npy')
    arr = np.concatenate((tr, ts))    
    df = pd.DataFrame(arr)
    non_zero_df = df.iloc[:, df.columns[(df != 0).any()]]
    return non_zero_df.values[:50000,:], non_zero_df.values[50000:,:]

#def cnn_svm():
cnn_train, cnn_test = del_cols()
cnn_train = normalize(cnn_train)
cnn_test = normalize(cnn_test)
svr=svm.LinearSVC()


svm_cnn = GridSearchCV(svr, parameters)
svm_cnn.fit(cnn_train, label_train)
predicted_cnn = svm_cnn.predict(cnn_test)
cnnacc = acc(label_test, predicted_cnn)

#%%
