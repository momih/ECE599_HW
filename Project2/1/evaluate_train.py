import sys
import get_cifardata
from CIFAR_LeNet5 import CifarLeNet
import pickle
import time

start = time.time()
cifar_train = get_cifardata.get_train()
cifar_test = get_cifardata.get_test()

# ==============================================================================
# Evaluating convergence
# ==============================================================================


def algos(optimizers, filename, m=False, m_val=0.5):
    if not m:
        train_errors = {}
        for algo in optimizers:
            acc_list = []
            cnn = CifarLeNet(lr=0.0001, epochs=5000, batch_size=500, train_data=cifar_train,
                             test_data=cifar_test, wd=0.004, decay_lr=False,
                             decay_w=False, optimizer=algo, seed=100)
            cnn.train(acc_list)
            cnn.reset()
            train_errors[algo] = acc_list
    
        with open('pickled_data/' + filename, 'wb') as f:
            pickle.dump(train_errors, f)
    else:
        train_errors = {}
        for value in m_val:
            acc_list = []
            cnn = CifarLeNet(lr=0.0001, epochs=5000, batch_size=500, train_data=cifar_train,
                             test_data=cifar_test, wd=0.004, decay_lr=False,
                             decay_w=False, optimizer='momentum', seed=100, momentum=value)
            cnn.train(acc_list)
            cnn.reset()
            train_errors['momentum' + str(value)] = acc_list
    
        with open('pickled_data/' + filename, 'wb') as f:
            pickle.dump(train_errors, f)


def lrs(rates, filename):
    train_errors = {}
    for lr_rate in rates:
        acc_list = []
        cnn = CifarLeNet(lr=lr_rate, epochs=5000, batch_size=500, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=False, optimizer='adam', seed=100)
        cnn.train(acc_list)
        cnn.reset()
        train_errors[str(lr_rate)] = acc_list
    with open('pickled_data/' + filename, 'wb') as f:
        pickle.dump(train_errors, f)

def decay_lr(filename, start_rate=0.01):
    train_errors = {}
    print('Decay')
    acc_list = []
    cnn = CifarLeNet(lr=start_rate, epochs=10000, batch_size=128, train_data=cifar_train,
                     test_data=cifar_test, wd=0.004, decay_lr=True,
                     decay_w=False, optimizer='adam', seed=100)
    cnn.train(acc_list)
    cnn.reset()
    train_errors['decay'] = acc_list
    with open('pickled_data/' + filename, 'wb') as f:
        pickle.dump(train_errors, f)
    return train_errors

if len(sys.argv) == 1:
    print "Please enter an argument: \n [optimizer] [momentum] - Check convergence" \
    " for type of optimizer \n [lrs] - Check convergence for learning rates" \
    "\n [decay] [starting_rate (optional, default = 0.01)] - Check convergence for learning rate decay"
else:
    if sys.argv[1] == 'optimizer':
        if len(sys.argv) == 3:
            m_list = [0.5, 0.6, 0.7, 0.8, 0.9]
            algos(None, 'convergence_momentum', True, m_list)
        else:
            optimizer_list = ['adam', 'adagrad', 'adadelta', 'momentum', 'rmsprop']
            algos(optimizer_list, 'convergence_optimizers')
    elif sys.argv[1] == 'lrs':
        rates_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
        lrs(rates_list, 'convergence_lrs')
    elif sys.argv[1] == 'decay':
        if len(sys.argv) == 2:
            decay_lr('convergence_decay_1k')
        else:
            decay_lr('convergence_decay_1k', float(sys.argv[2]))

run_time = time.time()
total =  run_time - start
with open('time', 'a') as f:
    f.write('\n Time taken to run total for ' + " ".join(sys.argv) + ' ' + str(total) +'s')
#
# Test acc -
# batch size with lr= 0.001, epochs - 20000
# lr with epochs = 20000, batch size = 5000
# epochs with lr = 0.001, batch size = 5000, weight decay
#
#
# train error
# type of optimizer: convergence vs epochs 20000, 0.001
# learning rate with decayed rate: convergence vs epochs 20000, 0.001
#