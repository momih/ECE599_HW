# =============================================================================
# Testing h5py etc
# =============================================================================

from __future__ import print_function
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from densenet121 import densenet
import tensorflow as tf
import numpy as np
import os
import time

tf.logging.set_verbosity(tf.logging.DEBUG)

home_dir = os.environ['HOME']
if 'momi' in home_dir:
    PATH = '/home/momi/Documents/599/Final/Code/densenet/'
elif 'mkhan31' in home_dir:
    PATH = '/home/mkhan31/ECE599_HW/Final/Code/densenet/'

models_dir = PATH + 'models/'
log_dir = PATH + 'logs/'
  
# =============================================================================
# Data loading and defining generator
# =============================================================================
print('Data loading and defining image generator')
with np.load('data.npz') as data:
    # Training data
    X_train = data['X_train'][:10]
    Y_train = data['Y_train'][:10]

    # Validation data
    X_valid = data['X_valid'][:10]
    Y_valid = data['Y_valid'][:10]

Y_train = np.argmax(Y_train, axis=1)
Y_valid = np.argmax(Y_valid, axis=1)

train_gen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                               zoom_range=0.2)
validation_gen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True)
train_gen.fit(X_train)
validation_gen.fit(X_valid)

# =============================================================================
# Define callbacks
# =============================================================================
print('Define callbacks')
ckpt = ModelCheckpoint(models_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       verbose=1, period=1)
early_stop = EarlyStopping(min_delta=0, patience=10, verbose=1)
csvlog = CSVLogger(PATH + 'stats.csv', append=True)
reducelr = ReduceLROnPlateau(verbose=3)

print('\nLoad model')
# =============================================================================
# Load and train model
# =============================================================================

model = densenet(img_rows=224, img_cols=224, color_type=1,
                 num_classes=2, bn_type='bn', opt='adam')
print('\n Train')    
start = time.time()
batch_size = 2
model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=2,
                    verbose=1,
                    validation_data=validation_gen.flow(X_valid, Y_valid, batch_size=batch_size),
                    validation_steps=len(X_valid) / batch_size,
                    callbacks=[reducelr, early_stop, csvlog, ckpt])

print(time.time() - start)
