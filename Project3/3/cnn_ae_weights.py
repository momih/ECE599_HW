import pickle
import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import numpy as np
import cv2

import getdata
from LeNet_AE import LeNetAE
cifar_train = getdata.get_train()
cifar_test = getdata.get_test()
fc1 = rdill('/home/momi/Documents/599/Project3/3/fc1')
cnn = LeNetAE(lr=0.001, epochs=101, batch_size=256, train_data=cifar_train,
                         test_data=cifar_test, wd=0.004, decay_lr=False,
                         decay_w=False, optimizer='rmsprop', seed=124, 
                         model_name='lenet/rmsprop', init_dev=0.01, drop=0.2)