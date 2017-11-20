from cv2 import imread, resize
from tqdm import tqdm
import numpy as np
import random
import os
  
os.chdir('../data/xrays/')

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def getxrays(filepath, label):
    files =  [x for x in os.listdir(filepath) if '.png' in x]
    images = []
    for i in tqdm(files):
        xray = resize(imread(filepath+i, 0), (256,256))
        images.append(xray)
    x = np.array(images)
    y = np.full((len(images)), label)
    return x, y

healthy_x, healthy_y = getxrays('healthy/', 0)
shadow_x, shadow_y = getxrays('shadow/', 1)

x = np.vstack((healthy_x, shadow_x))
y = np.concatenate((healthy_y, shadow_y))

y = to_categorical(y, 2)

np.save('x.npy', x)
np.save('y.npy', y)
