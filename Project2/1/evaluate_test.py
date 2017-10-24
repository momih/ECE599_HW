import get_cifardata
from CIFAR_LeNet5 import CifarLeNet
import pickle
import logging
import time

# ==============================================================================
# Evaluating test errors
# ==============================================================================
cifar_train = get_cifardata.get_train()
cifar_test = get_cifardata.get_test()
   
def lr_test():
    lr_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    # test_errors = {}
    train_errors = {}
    test_rate = []
    for rate in lr_rates:
        train_err = []
        cnn = CifarLeNet(lr=rate, epochs=10000, batch_size=128, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=True, optimizer='adam', seed=100)
        cnn.train(train_err)
        train_errors[str(rate)] = train_err
        cnn.test_eval(rate, test_rate)
        cnn.reset()

    with open('pickled_data/test_acc_lrs', 'wb') as f:
        pickle.dump([train_errors, test_rate], f)

    return test_rate


def batch_sizes():
    sizes = [10, 25, 50, 64, 100, 200, 300, 400, 500, 750, 1000, 2000]
    # test_errors = {}
    train_errors = {}
    test_rate = []
    for bs in sizes:
        train_err = []
        cnn = CifarLeNet(lr=0.001, epochs=10000, batch_size=bs, test_data=cifar_test,
                         train_data=cifar_train, wd=0.004, decay_lr=False,
                         decay_w=True, optimizer='adam', seed=100)
        cnn.train(train_err)
        train_errors[str(bs)] = train_err
        cnn.test_eval(bs, test_rate)
        cnn.reset()

    with open('pickled_data/test_acc_bs', 'wb') as f:
        pickle.dump([train_errors, test_rate], f)

    return test_rate


def epochs_test():
    epoch_list = [x for x in range(1000, 15001) if x % 1000 == 0]
    # test_errors = {}
    train_errors = {}
    test_rate = []
    for epochs in epoch_list:
        train_err = []
        cnn = CifarLeNet(lr=0.001, epochs=epochs, batch_size=128, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=False, optimizer='adam', seed=100)
        cnn.train(train_err)
        train_errors[str(epochs)] = train_err
        cnn.test_eval(epochs, test_rate)
        cnn.reset()

    with open('pickled_data/test_acc_epochs', 'wb') as f:
        pickle.dump([train_errors, test_rate], f)

    return test_rate


def weight_decay_test():
    epoch_list = [x for x in range(1000, 15001) if x % 1000 == 0]
    # test_errors = {}
    train_errors = {}
    test_rate = []
    for epochs in epoch_list:
        train_err = []
        cnn = CifarLeNet(lr=0.001, epochs=epochs, batch_size=128, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=True, optimizer='adam', seed=100)
        cnn.train(train_err)
        train_errors[str(epochs)] = train_err
        cnn.test_eval(epochs, test_rate)
        cnn.reset()

    with open('pickled_data/test_acc_epochs_wd', 'wb') as f:
        pickle.dump([train_errors, test_rate], f)

    return test_rate


def main():
    start = time.time() 
    lr_test()
    run_time = time.time()
    total =  run_time - start
    with open('time_test', 'a') as f:
        f.write('\n Time taken to run total for lr_test ' + str(total) +' s')
    
    start = time.time()
    epochs_test()
    run_time = time.time()
    total =  run_time - start
    with open('time_test', 'a') as f:
        f.write('\n Time taken to run total for epoch_test ' + str(total) +' s')
    
    start = time.time()
    weight_decay_test()
    run_time = time.time()
    total =  run_time - start
    with open('time_test', 'a') as f:
        f.write('\n Time taken to run total for decay test ' + str(total) +' s')
    
    start = time.time()
    batch_sizes()
    run_time = time.time()
    total =  run_time - start
    with open('time_test', 'a') as f:
        f.write('\n Time taken to run total for batches test ' + str(total) +' s')
    
logging.basicConfig(level=logging.DEBUG, filename='myapp.log')

#try:
#    main()
#except:
#    logging.exception("Oops:") 