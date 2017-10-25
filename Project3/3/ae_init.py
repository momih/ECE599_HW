import pickle
import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import numpy as np
import cv2

import getdata
from LeNet5 import LeNet

from autoencoder import DenoisingAutoencoder as DAE
import datasets



#%%
# =============================================================================
#       Preprocessing
# =============================================================================
images_train, label_train, images_test, label_test = datasets.load_cifar10_dataset('../cifar-10-batches-py/', normalize=False, mode='supervised')


def resize(arr, n):
    arr = arr.reshape(n,3,32,32)
    arr = arr.transpose([0, 2, 3, 1])
    e =[]
    for i in range(n):
        x=cv2.resize(arr[i], (36,36)).reshape(3888)
        x = np.pad(x, (0,208), 'constant')
        e.append(x)
    return np.vstack(e)

resized_train = resize(images_train, 50000)       
resized_test = resize(images_test, 10000)
#%%
# =============================================================================
#       FC_1
# =============================================================================

dae = DAE(model_name='fc1', pickle_name='fc1', test_name='fc1',
         n_components=4096, main_dir='fc1/', 
         enc_act_func='sigmoid', dec_act_func='none', 
         loss_func='mean_squared', num_epochs=6, batch_size=20, 
         dataset='cifar10', xavier_init=1, opt='adam', 
         learning_rate=0.0001, momentum=0.5, corr_type='gaussian',
         corr_frac=0.5, verbose=1, seed=1)
val_dict = {}
dae.fit(resized_train, val_dict, resized_test, restore_previous_model=False) 
    
