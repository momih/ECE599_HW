import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0, '../utils/')
import getdata
import pickle

lr = 0.001
batch_size = 256
epochs = 4
# Input and target placeholders
x = tf.placeholder(tf.float32, (None, 3, 1024), name="input")
targets_ = tf.placeholder(tf.float32, (None, 3, 1024), name="target")

x_image = tf.reshape(x, [-1, 32, 32, 3])

xtr = tf.placeholder(tf.float32, (None, 3, 1024), name="input")

### Encoder
conv = tf.layers.conv2d(inputs=x_image, 
                         filters=32, 
                         kernel_size=(5,5), 
                         padding='same', 
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                         name='conv')

deconv = tf.layers.conv2d_transpose(inputs=conv,
                                    filters=3,
                                    kernel_size=(5,5),
                                    padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

cost = tf.sqrt(tf.reduce_mean(tf.square(x_image - deconv)))
train_step = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cifar_train = getdata.get_train()
cifar_test = getdata.get_test()

for i in range(epochs):
    batch = cifar_train.next_batch(batch_size)
    test = cifar_test.next_batch(1000)
    if i % 10 == 0:
        cost_v = sess.run(cost,feed_dict={x: batch[0], xtr:batch[0]})
        cost_t = sess.run(cost,feed_dict={x: test[0]})

        print('Step %d, Training loss: %g, val loss: %g ' % (i, cost_v, cost_t))
        
    sess.run([train_step], feed_dict={x: batch[0]})

weights = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv/kernel')[0])
#
#with open('conv1', 'wb') as f:
#    pickle.dump(weights, f)
