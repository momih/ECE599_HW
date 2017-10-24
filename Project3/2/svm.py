import numpy as np
import sys
sys.path.insert(0, '../utils/')
from sklearn import svm
import datasets
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV

trX, trY, teX, teY = datasets.load_cifar10_dataset('../cifar-10-batches-py/', mode='supervised')
train_y = trY[:50]
test_y = trY[50:100]
del trX, trY, teX, teY
dae_train = np.load('dae_svm-dae_svm.npy')[:50]
dae_test = np.load('dae_svm-dae_svm.npy')[50:100]

cnn_train = np.load('cnn.npy')[:50]
cnn_test = np.load('cnn.npy')[50:]

parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma': 
              [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5]}

svr=svm.SVC()

svm_cnn = GridSearchCV(svr, parameters)
svm_cnn.fit(cnn_train, train_y)
predicted_cnn = svm_cnn.predict(cnn_test)
cnnacc = acc(test_y, predicted_cnn)

svm_dae = GridSearchCV(svr, parameters)
svm_dae.fit(dae_train, train_y)
predicted_dae = svm_dae.predict(dae_test)
daeacc = acc(test_y, predicted_dae)
