from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_size', action="store", dest="prop", default=0.3)
parser.add_argument('--seed', action="store", dest="seed", default=100)
args = parser.parse_args()
prop = args.prop
seed = args.seed

home_dir = os.environ['HOME']
if 'momi' in home_dir:
    PATH = '/home/momi/Documents/599/Final/Code/densenet/'
elif 'mkhan31' in home_dir:
    PATH = '/home/mkhan31/ECE599_HW/Final/Code/densenet/' 

X_train, X_test, Y_train, Y_test = train_test_split(np.load(PATH + 'x.npy'),
                                                      np.load(PATH + 'y.npy'),
                                                      test_size=prop,
                                                      random_state=seed)

X_valid, Y_valid = X_train[10700:,:,:, None], Y_train[10700:,]
X_train, Y_train = X_train[:10700,:,:, None], Y_train[:10700,]
X_test = X_test[:,:,:, None]

np.savez(PATH + 'data', X_valid=X_valid, Y_valid=Y_valid, X_train=X_train, Y_train=Y_train,
         X_test=X_test, Y_test=Y_test)

print('Data saved at ' + PATH)

