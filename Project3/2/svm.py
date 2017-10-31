#%%
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
import pickle
PATHS = '/home/mkhan31/ECE599_HW/Project3/2/'
#%%
label_train = np.load('trainlab.npy')
label_test = np.load('testlab.npy')
parameters = { 'C':[1,2,5], 'gamma': 
              [0.01,0.02,0.05], 'kernel':['linear', 'rbf'] }
#%% DAE
#def dae_svm():
dae_train = np.load('data/train_dae.npy')
dae_test = np.load('data/test_dae.npy')

svm_dae = svm.SVC(C=2.0, gamma = 0.05)
#svm_dae = GridSearchCV(svr, parameters)
svm_dae.fit(dae_train, label_train)
predicted_dae = svm_dae.predict(dae_test)
daeacc = acc(label_test, predicted_dae)
#model_params = str(svm_dae.best_estimator_)

print 'DAE accuracy - ' +str(daeacc)

with open(PATHS + 'dae_svm', 'wb') as f:
    pickle.dump(daeacc, f)

#%% CNN
cnn_train = np.load('data/train_cnn.npy')
cnn_test = np.load('data/test_cnn.npy')
   

#def cnn_svm():
cnn_train = normalize(cnn_train)
cnn_test = normalize(cnn_test)
svm_cnn=svm.SVC(C=2.0)

#svm_cnn = GridSearchCV(svr, parameters)
svm_cnn.fit(cnn_train, label_train)
predicted_cnn = svm_cnn.predict(cnn_test)
cnnacc = acc(label_test, predicted_cnn)
#model_params = str(svm_cnn.best_estimator_)

print 'CNN accuracy - ' +str(cnnacc)

with open(PATHS + 'cnn_svm', 'wb') as f:
    pickle.dump(cnnacc, f)
#%%
