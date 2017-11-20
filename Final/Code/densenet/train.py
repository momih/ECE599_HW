from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, log_loss
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
early_stop = EarlyStopping(min_delta=0, patience=15, verbose=1)
csvlog = CSVLogger(PATH + 'stats.csv', append=True)
reducelr = ReduceLROnPlateau(verbose=1)

# Load our model
model = densenet(img_rows=224, img_cols=224, color_type=1, 
                 num_classes=2, bn_type='bn', opt='adam')


X_train, X_test, Y_train, Y_test = train_test_split(np.load('x.npy'),
                                                      np.load('y.npy'),
                                                      test_size=0.3,
                                                      random_state=10)
X_valid, Y_valid = X_train[800:1000,:,:, None], Y_train[800:1000,]
X_train, Y_train = X_train[:800,:,:, None], Y_train[:800,]
X_test = X_test[:,:,:, None]

datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             horizontal_flip=True,
                             zoom_range=0.2)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=2),
                    steps_per_epoch=len(X_train) / 2,  
                    epochs=1,
                    verbose=1,
                    validation_data=(X_valid, Y_valid),
                    callbacks=[reducelr, ckpt, early_stop, csvlog])

## Make predictions
#datagen_y = ImageDataGenerator(featurewise_center=True,
#                             featurewise_std_normalization=True)
#datagen_y.fit(X_test)
#predictions_valid = model.predict_generator(datagen_y.flow(X_test[:800], batch_size=batch_size),
#                                            steps =len(X_test[:800])/batch_size,
#                                            verbose=1)
#r=X_test[:800]
#r =  (r - r.mean(axis=0))/r.std(axis=0)
#
#w = model.predict(r, batch_size=16, verbose=1)
## Cross-entropy loss score
#print log_loss(Y_test[:800], predictions_valid)
#


