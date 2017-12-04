from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import cPickle
import numpy


class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    self._num_examples = images.shape[0]

    if dtype == dtypes.float32:
    # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def one_hot(x):
    # type: (object) -> object
    encoded = numpy.zeros((len(x), 10))
    encoded[numpy.arange(len(x)), x] = 1
    return encoded


def get_train():
    # Reading training images
    for i in range(1, 6):
        with open('cifar-10-batches-py/data_batch_' + str(i), 'rb') as f:
            cifar_dict = cPickle.load(f)
        if i == 1:
            images = cifar_dict['data'].reshape(10000, 3, 1024)
            labels = one_hot(cifar_dict['labels'])
        else:
            data = cifar_dict['data'].reshape(10000, 3, 1024)
            y = one_hot(cifar_dict['labels'])
            images = numpy.concatenate((images, data), axis=0)
            labels = numpy.concatenate((labels, y), axis=0)
    return DataSet(images, labels])


def get_test():

    with open('cifar-10-batches-py/test_batch', 'rb') as f:
            cifar_dict = cPickle.load(f)
    images = cifar_dict['data'].reshape(10000, 3, 1024)
    labels = one_hot(cifar_dict['labels'])
    return DataSet(images, labels)
