# My adapatation of  Liu's code for CIFAR10
# from - http://web.eecs.utk.edu/~qi/deeplearning/code/cnnMNIST.py
# =============================================================================

import tensorflow as tf


class CifarLeNet(object):
    def __init__(self, lr, epochs, batch_size, train_data, test_data, wd=0.004,
                 decay_lr=False, decay_w=False, optimizer='adam', seed=None, momentum=0.5):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.decay_lr = decay_lr
        self.decay_w = decay_w
        self.weight_decay = tf.constant(wd, dtype=tf.float32)
        self.seed = seed
        self.momentum = momentum
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 3, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # define conv-layer variables
        W_conv1 = self.weight_variable([5, 5, 3, 32])
        # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        x_image = tf.reshape(self.x, [-1, 32, 32, 3])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([8*8 * 64, 1024], name='W_fc1',
                                     decay=self.decay_w)

        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([1024, 10], name='W_fc2',
                                     decay=self.decay_w)

        b_fc2 = self.bias_variable([10])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        # Select optimizer
        if self.decay_lr:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step,
                                                       200, 0.96, 
                                                       staircase=True)
            # Passing global_step to minimize() will increment it at each step.
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                                      .minimize(self.cross_entropy,
                                                global_step=global_step)

        else:
            if self.optimizer == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.lr)\
                                          .minimize(self.cross_entropy)

            elif self.optimizer == 'adagrad':
                self.train_step = tf.train.AdagradOptimizer(self.lr)\
                                          .minimize(self.cross_entropy)

            elif self.optimizer == 'adadelta':
                self.train_step = tf.train.AdadeltaOptimizer(self.lr)\
                                          .minimize(self.cross_entropy)

            elif self.optimizer == 'momentum':
                # momentum = 0.5
                self.train_step = tf.train.MomentumOptimizer(self.lr,
                                                             self.momentum)\
                                          .minimize(self.cross_entropy)

            elif self.optimizer == 'rmsprop':
                self.train_step = tf.train.RMSPropOptimizer(self.lr)\
                                          .minimize(self.cross_entropy)

    def train(self, accuracies):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        for i in range(self.epochs):
            batch = self.train_data.next_batch(self.batch_size)
            # print (i)
            if i % 100 == 0:
                train_acc, cost_v = self.sess.run([self.accuracy,
                                                   self.cross_entropy],
                                                  feed_dict={self.x: batch[0],
                                                             self.y_: batch[1],
                                                             self.keep_prob: 1.0})
                print(self.optimizer + ' Step %d, Training accuracy: %g, Loss: %g' % (i, train_acc, cost_v))
                accuracies.append([i, train_acc, cost_v])
            self.sess.run([self.train_step], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
#        return accuracies

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1),
                                      tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self, param, test_list):
        self.eval()
        test_acc = self.sess.run(self.accuracy,
                                 feed_dict={self.x: self.test_data.images,
                                            self.y_: self.test_data.labels,
                                            self.keep_prob: 1.0})
        print('test accuracy %g' % test_acc)
        test_list.append([param, test_acc])
        return test_list

    def weight_variable(self, shape, name=None, decay=False):
        with tf.device('/cpu:0'):
            if decay:
                initial = tf.get_variable(name, shape,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, seed=self.seed),
                                          regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            else:
                initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1, seed=self.seed))

        return initial

    def bias_variable(self, shape):
#        initial = tf.constant(0.1, shape=shape)
#        return tf.Variable(initial)
        with tf.device('/cpu:0'):
            initial = tf.Variable(tf.constant(0.1, shape=shape))
        return initial

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')

    def reset(self):
        self.sess.close()
        tf.reset_default_graph()
