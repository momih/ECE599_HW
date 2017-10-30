import sys
import os
sys.path.insert(0, '../utils/')
import numpy as np

import getdata
from LeNet5 import LeNet

from autoencoder import DenoisingAutoencoder as DAE

if sys.argv[1] =='dae':
    dae = DAE(model_name='dae_svm', pickle_name='svm', test_name='svm',
                 n_components=256, main_dir='dae/', 
                 enc_act_func='sigmoid', dec_act_func='none', 
                 loss_func='mean_squared', num_epochs=50, batch_size=20, 
                 dataset='cifar10', xavier_init=1, opt='adam', 
                 learning_rate=0.0001, momentum=0.5, corr_type='gaussian',
                 corr_frac=0.5, verbose=1, seed=1)    
    
    trX, trY, teX, teY = getdata.load_cifar10_dataset('../cifar-10-batches-py/', mode='supervised')
    val_dict = {}
    dae.fit(trX, val_dict, teX, restore_previous_model=True) 
    
    #dae.load_model(256, 'models/dae/dae_svm')
    dae_svm_train = dae.transform(trX, name='dae_svm_train_na', save=True)
    dae_svm_test = dae.transform(teX, name='dae_svm_test_na', save=True)

    
elif sys.argv[1]=='cnn':
    cifar_train = getdata.get_train()
    cifar_test = getdata.get_test()
    acc_list = []
    cnn = LeNet(lr=0.001, epochs=101, batch_size=256, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=False, optimizer='rmsprop', seed=124, 
                         model_name='lenet/rmsprop', init_dev=0.01, drop=0.2)
    cnn.restore('models/lenet/rmsprop')
    train_list = np.split(cifar_train.images, 20)
    test_list = np.split(cifar_test.images, 4)
    for i in range(len(train_list)):
        print "\n Batch" + str(i)
        cnn.transform(train_list[i], name='cnn_train' + str(i), save=True)
    for i in range(len(test_list)):
        print "\n Batch" + str(i)
        cnn.transform(test_list[i], name='cnn_test' + str(i), save=True)


def combine():
    os.chdir('/home/momi/Documents/599/Project3/2/data/cnn/')
    files = os.listdir(os.getcwd())
    files.sort()
    train = []
    test = []
    
    for i in [x for x in files if 'test' in x]:
        test.append(np.load(i))
    np.save('test_cnn', np.vstack(test))   
    del test
    
    for i in [x for x in files if 'train' in x]:
        train.append(np.load(i)[:5])
    np.save('train_cnn', np.vstack(train))
    del train
       
