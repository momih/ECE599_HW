from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from densenet121 import densenet
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', action="store", dest="epochs", default=50)
parser.add_argument('--bsize', action="store", dest="bs", default=16)
args = parser.parse_args()
epochs = args.epochs
batch_size = args.bs

env = 1
PATH = '/home/momi/Documents/599/Final/Code/densenet/' if env == 1 else '/home/mkhan31/ECE599_HW/Final/Code/densenet/' 
models_dir = PATH + 'models/'
log_dir = PATH +'logs/'  
try:
    os.mkdir(models_dir)
    os.mkdir(log_dir)
except OSError:
    print "Directories already created"
    

   

ckpt = ModelCheckpoint(models_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       verbose=1, period=5)
early_stop = EarlyStopping(min_delta=0, patience=2)
csvlog = CSVLogger(PATH + 'stats.csv', append=True)
reducelr = ReduceLROnPlateau()

# Load our model
model = densenet(img_rows=224, img_cols=224, color_type=1, 
                 num_classes=2, bn_type='brn')


X_train, X_test, Y_train, Y_test = train_test_split(np.load('data/xrays/x.npy'),
                                                      np.load('data/xrays/y.npy'),
                                                      test_size=0.3,
                                                      random_state=10)

# Start Fine-tuning
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_split=0.2,
          callbacks=[reducelr, ckpt, early_stop, csvlog])

# Make predictions
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

# Cross-entropy loss score
score = log_loss(Y_valid, predictions_valid)
