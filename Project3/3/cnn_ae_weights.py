import pickle
import sys
sys.path.insert(0, '../utils/')
import time 
import getdata
from LeNet_AE import LeNetAE

cifar_train = getdata.get_train()
cifar_test = getdata.get_test()

with open('conv1', 'rb') as f:
    c_1 = pickle.load(f)

if sys.argv[1]=='init':
    cnn = LeNetAE(lr=0.001, epochs=10, batch_size=256, train_data=cifar_train,
                             test_data=cifar_test, wd=0.004, decay_lr=False,
                             decay_w=False, optimizer='rmsprop', seed=124, 
                             model_name='lenet/init', init_dev=0.01, drop=0.2)
    
    cnn.W_conv1.load(c_1, cnn.sess)    
    train_err = []
    test_rate = []
    start = time.time()
    cnn.train(train_err)
    cnn.test_eval('init', test_rate)
    run_time=time.time()
    total = run_time - start
    with open('time_init', 'wb') as f:
       f.write('\n Time taken to run total for training with init - ' + str(total) +' s')
    cnn.reset()
    with open('init_errs', 'wb') as f:
        pickle.dump([train_err, test_rate], f)


elif sys.argv[1]=='none':
    cnn = LeNetAE(lr=0.001, epochs=10, batch_size=256, train_data=cifar_train,
                             test_data=cifar_test, wd=0.004, decay_lr=False,
                             decay_w=False, optimizer='rmsprop', seed=124, 
                             model_name='lenet/p', init_dev=0.01, drop=0.2)
    
    start = time.time()

    train_err = []
    test_rate = []
    cnn.train(train_err)
    cnn.test_eval('none', test_rate)
    run_time=time.time()
    total = run_time - start
    cnn.reset()
    with open('none_errs', 'wb') as f:
        pickle.dump([train_err, test_rate], f)
    with open('time_none', 'wb') as f:
        f.write('\n Time taken to run total for training with none - ' + str(total) +' s')