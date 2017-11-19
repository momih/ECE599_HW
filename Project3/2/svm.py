#%%
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle
PATHS = '/home/mkhan31/ECE599_HW/Project3/2/'
#%%
label_train = np.load('trainlab.npy')
label_test = np.load('testlab.npy')
parameters = {'C':[0.5, 1, 2, 3, 4, 5]}
#%% DAE
def dae_svm():
    dae_train = np.load('data/train_dae.npy')[:10000]
    dae_test = np.load('data/test_dae.npy')[:5000]
    
    svm_dae = svm.SVC(C=2.0, gamma = 0.05, cache_size=2000)
    #svm_dae = GridSearchCV(svr, parameters)
    svm_dae.fit(dae_train, label_train)
    predicted_dae = svm_dae.predict(dae_test)
    daeacc = acc(label_test, predicted_dae)
    #model_params = str(svm_dae.best_estimator_)
    
    print 'DAE accuracy - ' + str(daeacc)
    
    with open(PATHS + 'dae_svm', 'wb') as f:
        pickle.dump(daeacc, f)

#%% CNN
cnn_train = np.load('data/train_cnn.npy')
cnn_test = np.load('data/test_cnn.npy')
   

#def cnn_svm():
cnn_train = normalize(cnn_train, axis=0)
cnn_test = normalize(cnn_test, axis=0)
svm_cnn=svm.SVC(C=2.0, gamma=0.05, cache_size=2000)
# %%
w = GridSearchCV(svm_cnn, parameters)

svm_cnn.fit(cnn_train, label_train)
predicted_cnn = svm_cnn.predict(cnn_test)
cnnacc = acc(label_test, predicted_cnn)
#model_params = str(svm_cnn.best_estimator_)

print 'CNN accuracy - ' +str(cnnacc)

with open(PATHS + 'cnn_svm', 'wb') as f:
    pickle.dump(cnnacc, f)
#%%    
import time
n_estimators = 3
clf = OneVsRestClassifier(BaggingClassifier(svm_cnn, bootstrap=False, n_jobs=3, max_samples=1.0 / n_estimators, n_estimators=n_estimators))
start = time.time()
clf.fit(cnn_train, label_train)
end = time.time() -start
predicted_cnn = clf.predict(cnn_test)
cnnacc = acc(label_test, predicted_cnn)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(cnn_train, label_train)
pred_rf = rf.predict(cnn_test)
acc(label_test, pred_rf)
