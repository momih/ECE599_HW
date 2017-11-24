from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import accuracy_score, log_loss
import keras.backend as K
from densenet121 import densenet
import numpy as np
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', action="store", dest="epochs", default=20)
parser.add_argument('--bsize', action="store", dest="bs", default=8)
parser.add_argument('--load', action="store", dest="load", default='weights_iter_1')
parser.add_argument('--save', action="store", dest="save", default='weights_iter_2')
parser.add_argument('--n', action='store', dest='n', default=10700)

args = parser.parse_args()
epochs = int(args.epochs)
batch_size = int(args.bs)
load_weights_name = args.load
save_weights_name = args.save
n = int(args.n)

home_dir = os.environ['HOME']
if 'momi' in home_dir:
    PATH = '/home/momi/Documents/599/Final/Code/densenet/'
elif 'mkhan31' in home_dir:
    PATH = '/home/mkhan31/ECE599_HW/Final/Code/densenet/'

models_dir = PATH + 'models/'
log_dir = PATH + 'logs/'
try:
    os.mkdir(models_dir)
    os.mkdir(log_dir)
except OSError:
    print "Directories already created"
  
# =============================================================================
# Data loading and defining generator
# =============================================================================
with np.load('data.npz') as data:
    # Training data
    X_train = data['X_train'][:n]
    Y_train = data['Y_train'][:n]

    # Validation data
    X_valid = data['X_valid']
    Y_valid = data['Y_valid']

Y_train = np.argmax(Y_train, axis=1)
Y_valid = np.argmax(Y_valid, axis=1)

train_gen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             horizontal_flip=True,
                             zoom_range=0.2)
validation_gen = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True)
train_gen.fit(X_train)
validation_gen.fit(X_valid)

# =============================================================================
# Define callbacks
# =============================================================================
early_stop = EarlyStopping(min_delta=0, patience=8, verbose=1)
csvlog = CSVLogger(models_dir + save_weights_name +'_stats.csv', append=True)
reducelr = ReduceLROnPlateau(verbose=1, patience=2)
reducelr_loss = ReduceLROnPlateau(verbose=1, patience=1, monitor='loss', epsilon=0.001)

# =============================================================================
# Load and train model
# =============================================================================

model = densenet(img_rows=224, img_cols=224, color_type=1,
                 num_classes=2, bn_type='bn', lr=0.01, opt='adam')

pretrained_files = [x for x in os.listdir('models/') if 'weights_iter' in x]
if len(pretrained_files) > 0:
    with open(models_dir + load_weights_name, 'rb') as f:
        loaded_weights = pickle.load(f)
    model.set_weights(loaded_weights)
    print('Loaded weights from ' + models_dir + load_weights_name)

model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_gen.flow(X_valid, Y_valid, batch_size=batch_size),
                    validation_steps=len(X_valid) / batch_size,
                    callbacks=[reducelr, early_stop, csvlog, reducelr_loss])


weights = model.get_weights()
with open(models_dir + save_weights_name, 'wb') as f:
    pickle.dump(weights, f)

print('Saved weights to ' + models_dir + save_weights_name)