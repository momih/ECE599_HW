import tensorflow as tf
import sys
sys.path.insert(0, '/home/momi/Documents/599/Project3/utils/')

import getdata
from LeNet5 import LeNet
import pickle

cifar_train = getdata.get_train()
cifar_test = getdata.get_test()
acc_list = []
if sys.argv[1]=='train':
    if sys.argv[2]=='rms':
        cnn = LeNet(lr=0.001, epochs=101, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='rmsprop', seed=124, 
                     model_name='lenet/rmsprop', init_dev=0.01, drop=0.2)
        cnn.train(acc_list)
        with open('rmsprop', 'wb') as f:
            pickle.dump(acc_list,f)
        cnn.reset()
        
    if sys.argv[2]=='adam':
        cnn = LeNet(lr=0.001, epochs=101, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='adam', seed=124, 
                     model_name='lenet/adam', init_dev=0.1, drop=0.5)
        cnn.train(acc_list)
        with open('adam', 'wb') as f:
            pickle.dump(acc_list,f)
        cnn.reset()


    if sys.argv[2]=='rmsx':
        cnn = LeNet(lr=0.0001, epochs=101, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='rmsprop', seed=124, 
                     model_name='lenet/rmspropp', init_dev=0.01, drop=0.2)
        cnn.train(acc_list)
        with open('rmspropp', 'wb') as f:
            pickle.dump(acc_list,f)
        cnn.reset()

elif sys.argv[1]=='testing':
    t=[]
    if sys.argv[2]=='rms':
        cnn = LeNet(lr=0.001, epochs=10000, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='rmsprop', seed=124, 
                     model_name='lenet/rmsprop', init_dev=0.01, drop=0.2)
        cnn.restore('models/lenet/rmsprop')
        cnn.test_eval('bs', t)
        with open('rmsprop', 'wb') as f:
            pickle.dump(t,f)
        cnn.reset()
        
    if sys.argv[2]=='adam':
        cnn = LeNet(lr=0.001, epochs=10000, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='adam', seed=124, 
                     model_name='lenet/adam', init_dev=0.1, drop=0.5)
        cnn.restore('models/lenet/adam')
        cnn.test_eval('bs', t)
        with open('adam', 'wb') as f:
            pickle.dump(t,f)
        cnn.reset()
        
    if sys.argv[2]=='rmsx':
        cnn = LeNet(lr=0.0001, epochs=10000, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='rmsprop', seed=124, 
                     model_name='lenet/rmspropp', init_dev=0.01, drop=0.2)
        cnn.restore('models/lenet/rmspropp')
        cnn.test_eval('bs', t)
        with open('rmspropp', 'wb') as f:
            pickle.dump(t,f)
        cnn.reset()
        

#test_rate=[]
#w = cnn.transform(cifar_train.images[:100])
###
#with open('w_tranform', 'wb') as f:
#    pickle.dump(w,f)
