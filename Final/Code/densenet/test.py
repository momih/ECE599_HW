from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from densenet121 import densenet
import numpy as np
import pickle
batch_size = 2

  
# =============================================================================
# Data loading and defining generator
# =============================================================================
with np.load('data.npz') as data:
    # Training data
    X_test = data['X_test']
    Y_test = data['Y_test']

test_gen = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True)
test_gen.fit(X_test)

model = densenet(img_rows=224, img_cols=224, color_type=1,
                 num_classes=2, bn_type='brn', opt='adam')

with open('models/weights_iter_3', 'rb') as f:
    model.set_weights(pickle.load(f))

print('Loaded weights')

predictions_valid = model.predict_generator(test_gen.flow(X_test, batch_size=batch_size),
                                            steps =len(X_test)/batch_size,
                                            verbose=1)
prediction = np.argmax(predictions_valid, 1)
Yarg= np.argmax(Y_test, 1)
accuracy_score(Yarg, prediction)
accuracy = np.mean(np.equal(prediction, Yarg))

print accuracy
print accuracy_score

