import pickle
import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')

import getdata
from LeNet5 import LeNet

from autoencoder import DenoisingAutoencoder as DAE
import datasets

dae = DAE(model_name='dae_svm', pickle_name='svm', test_name='svm',
             n_components=256, main_dir='dae/', 
             enc_act_func='sigmoid', dec_act_func='none', 
             loss_func='mean_squared', num_epochs=11, batch_size=20, 
             dataset='cifar10', xavier_init=1, opt='momentum', 
             learning_rate=0.001, momentum=0.5, corr_type='gaussian',
             corr_frac=0.3, verbose=1, seed=1)    

trX, trY, teX, teY = datasets.load_cifar10_dataset('../cifar-10-batches-py/', mode='supervised')
val_dict = {}
dae.fit(trX, val_dict, teX, restore_previous_model=True) 

#dae.load_model(256, 'models/dae/dae_svm')
dae_svm_data = dae.transform(trX, name='dae_svm', save=True)


cifar_train = getdata.get_train()
cifar_test = getdata.get_test()
acc_list = []
cnn = LeNet(lr=0.001, epochs=101, batch_size=256, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=False,
                     decay_w=False, optimizer='rmsprop', seed=124, 
                     model_name='lenet/tests', init_dev=0.01, drop=0.2)
cnn.train(acc_list)
cnn.transform(cifar_train.images[:100], name='cnn', save=True)
