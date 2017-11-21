from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import accuracy_score, log_loss
from densenet121 import densenet
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', action="store", dest="epochs", default=50)
parser.add_argument('--bsize', action="store", dest="bs", default=16)
args = parser.parse_args()
epochs = int(args.epochs)
batch_size = int(args.bs)

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
    X_train = data['X_train']
    Y_train = data['Y_train']

    # Validation data
    X_valid = data['X_valid']
    Y_valid = data['Y_valid']

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
ckpt = ModelCheckpoint(models_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                       verbose=1, period=5)
early_stop = EarlyStopping(min_delta=0, patience=10, verbose=1)
csvlog = CSVLogger(PATH + 'stats.csv', append=True)
reducelr = ReduceLROnPlateau(verbose=1)

# =============================================================================
# Load and train model
# =============================================================================

model = densenet(img_rows=224, img_cols=224, color_type=1,
                 num_classes=2, bn_type='brn', opt='adam')

model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_gen.flow(X_valid, Y_valid, batch_size=batch_size),
                    validation_steps=len(X_valid) / batch_size,
                    callbacks=[reducelr, ckpt, early_stop, csvlog])

# Make predictions
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