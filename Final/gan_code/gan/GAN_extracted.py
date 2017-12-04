from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.contrib.layers import layer_norm

from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import save_images, read_data, save_stats


class InfectGAN(object):
    def __init__(self, sess,
                 image_size=256,
                 batch_size=1,
                 sample_size=1,
                 output_size=256,
                 gf_dim=64,
                 df_dim=64,
                 l1_lambda=100,
                 defect_lambda=10,
                 wgan_lambda=10.0,
                 input_c_dim=1,
                 output_c_dim=1,
                 latent_bbox_size=32,
                 dataset_name="xray8",
                 checkpoint_dir=None,
                 loss="wgan",
                 paired=False,
                 contrast=False,
                 bbox_channels=3):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
            :type defect_lambda: object
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.bbox_channels = bbox_channels

        self.L1_lambda = l1_lambda
        self.defect_lamba = defect_lambda
        self.wgan_lambda = wgan_lambda
        
        self.latent_bbox_size = latent_bbox_size
        
        self.loss = loss
        self.paired = paired
        self.contrast = contrast
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.normal_xrays = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                           name='normal_xrays')
        self.real_abnormal = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                            name='abnormal_xrays')
        
        self.input_bbox = tf.placeholder(tf.float32,
                                         [self.batch_size, self.latent_bbox_size,
                                          self.latent_bbox_size, self.bbox_channels],
                                         name='bounding_defect')
        
        self.fake_abnormal, self.reconstructed_bbox = self.generator(image=self.normal_xrays, defect=self.input_bbox)

        if self.loss == "adversarial":
            if self.paired:
                # Pair normal images with real and generated abnormal images
                self.real_abnormal_to_discrm = tf.concat([self.normal_xrays, self.real_abnormal], 3)
                self.fake_abnormal_to_discrm = tf.concat([self.normal_xrays, self.fake_abnormal], 3)
            else:
                self.real_abnormal_to_discrm = self.real_abnormal
                self.fake_abnormal_to_discrm = self.fake_abnormal

            # Get discriminator logits for real images             
            self.D, self.real_logits = self.discriminator(self.real_abnormal_to_discrm, reuse=False)
            
            # Get discriminator logits for fake images
            self.D_, self.fake_logits = self.discriminator(self.fake_abnormal_to_discrm, reuse=True)
            
            # Discriminator loss
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                                                      labels=tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                                      labels=tf.zeros_like(self.D_)))
            
            self.d_loss = self.d_loss_real + self.d_loss_fake

            # Do we do add contrast loss?
            if self.contrast:
                self.abnormal_contrast = tf.concat([self.real_abnormal, self.fake_abnormal], 3)
                self.normal_contrast = tf.concat([self.normal_xrays, self.fake_abnormal], 3)
               
                # Get discriminator logits for abnormal contrast             
                self.D_c, self.ac_logits = self.discriminator(self.abnormal_contrast, reuse=False)
                
                # Get discriminator logits for normal contrast
                self.D_c_, self.nc_logits = self.discriminator(self.normal_contrast, reuse=True)    
                
                d_loss_ac = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.ac_logits,
                                                                                   labels=tf.ones_like(self.D_c)))
                d_loss_nc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.nc_logits,
                                                                                   labels=tf.zeros_like(self.D_c_)))

                self.d_loss = self.d_loss + d_loss_ac + d_loss_nc
        
            # Generator loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                                 labels=tf.ones_like(self.D_))) \
                          + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_abnormal - self.fake_abnormal)) \
                          + self.defect_lamba * tf.reduce_mean(tf.abs(self.input_bbox - self.reconstructed_bbox))
    
        elif self.loss == "wgan":
        
            # GP function
            def gradient_penalty(real, fake):
                def interpolate(a, b):
                    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                    inter = a + alpha * (b - a)
                    inter.set_shape(a.get_shape().as_list())
                    return inter
        
                x = interpolate(real, fake)
                unused, pred = self.discriminator(x, reuse=True)
                gradients = tf.gradients(pred, [x])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
                gp = tf.reduce_mean((slopes - 1.)**2)
                return gp          
            
            # Get discriminator logits for real images     
            self.D, self.real_logits = self.discriminator(self.real_abnormal, reuse=False)
            
            # Get discriminator logits for fake images  
            self.D_, self.fake_logits = self.discriminator(self.fake_abnormal, reuse=True)
        
            # Discriminator loss
            
            self.d_loss_real = tf.reduce_mean(self.real_logits)
            self.d_loss_fake = tf.reduce_mean(self.fake_logits)
            self.grad_p = gradient_penalty(self.real_abnormal, self.fake_abnormal)
            
            self.d_loss = self.d_loss_fake - self.d_loss_real + (self.grad_p * self.wgan_lambda)

            # Generator loss
            self.g_loss = -tf.reduce_mean(self.fake_logits) + self.defect_lamba * tf.reduce_mean(tf.abs(self.input_bbox - self.reconstructed_bbox))

        self.generated_sample = self.generator(self.normal_xrays, self.input_bbox, reuse=True)

        # Summary ops
        self.g_loss_summary = tf.summary.scalar("g_loss", self.g_loss)
        
        self.d_loss_real_summary = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        
        self.d_loss_summary = tf.summary.scalar("d_loss", self.d_loss)

        self.d_summary = tf.summary.histogram("real_logits", self.D)
        self.d__summary = tf.summary.histogram("fake_logits", self.D_)
        self.fake_abnormal_summary = tf.summary.image("fake_abnormal", self.fake_abnormal)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name or 't_' in var.name]

        self.saver = tf.train.Saver()
        self.tf_merged_summaries = tf.summary.merge_all()
        self.tf_summary_writer = tf.summary.FileWriter('logs/', self.sess.graph)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(layer_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(layer_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(layer_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, defect, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            # =============================================================================
            #     Target network        
            # =============================================================================
            def target_batch_norm(x, name, use=True):
                with tf.variable_scope(name):
                    epsilon = 1e-5
                    momentum = 0.9
                    naam = name
                if use:
                    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None,
                                                        epsilon=epsilon, scale=True, scope=naam)
                else:
                    return x
            
            # input defect is 32x32x1
            t_e1 = conv2d(defect, 8, name='t_e1_conv')
            # t_e1 is 16x16x8
            
            t_e2 = target_batch_norm(conv2d(lrelu(t_e1), 16, name="t_e2_conv"), name="t_e2_bn")
            # t_e2 is 8x8x16
            
            t_e3 = target_batch_norm(conv2d(lrelu(t_e2), 32, name="t_e3_conv"), name="t_e3_bn")
            # t_e3 is 4 x 4 x 32
            
            t_e4 = target_batch_norm(conv2d(lrelu(t_e3), 32, name="t_e4_conv"), name="t_e4_bn")
            # t_e4 is 2 x 2 x 32
                    
            latent_bbox = target_batch_norm(conv2d(lrelu(t_e4), self.latent_bbox_size,
                                                   name="latent_bbox_conv"), name="latent_bbox_bn")
            # latent_bbox is 1 x 1 x 32
            
            self.t_d4, self.t_d4_w, t_d4_b = deconv2d(tf.nn.relu(latent_bbox),
                                                      [self.batch_size, 2, 2, 32],
                                                      name='t_d4_conv',
                                                      with_w=True)
            t_d4 = target_batch_norm(self.t_d4, name="t_d4_bn")
            
            self.t_d3, self.t_d3_w, t_d3_b = deconv2d(tf.nn.relu(t_d4),
                                                      [self.batch_size, 4, 4, 32],
                                                      name='t_d3_conv',
                                                      with_w=True)
            t_d3 = target_batch_norm(self.t_d3, name="t_d3_bn")
            
            self.t_d2, self.t_d2_w, t_d2_b = deconv2d(tf.nn.relu(t_d3),
                                                      [self.batch_size, 8, 8, 16],
                                                      name='t_d2_conv',
                                                      with_w=True)
            t_d2 = target_batch_norm(self.t_d2, name="t_d2_bn")
            
            self.t_d1, self.t_d1_w, t_d1_b = deconv2d(tf.nn.relu(t_d2),
                                                      [self.batch_size, 16, 16, 8],
                                                      name='t_d1_conv',
                                                      with_w=True)
            t_d1 = target_batch_norm(self.t_d1, name="t_d1_bn")
            
            defect_output = lrelu(deconv2d(tf.nn.relu(t_d1), [self.batch_size, 32, 32, 1],
                                           name='reconstructed', with_w=False))

            # =============================================================================
            #     Decoder network        
            # =============================================================================
            
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), [self.batch_size, s128, s128, self.gf_dim*8],
                                                     name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7, t_d4], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6, t_d3], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5, t_d2], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4, t_d1], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            if not reuse:
                return tf.nn.tanh(self.d8), defect_output
            else:
                return tf.nn.tanh(self.d8)

    def sample_model(self, sample_dir, epoch, idx, data_type='val'):
        n_val = read_data('normal')
        d_val = read_data('defect')

        # Get batches
        n_batch = n_val.next_batch(self.batch_size, which=data_type, labels=True)
        d_batch = d_val.next_batch(self.batch_size, which=data_type, labels=True)

        # Run generator
        samples, d_loss, g_loss = self.sess.run([self.generated_sample, self.d_loss, self.g_loss],
                                                feed_dict={self.normal_xrays: n_batch[0],
                                                           self.input_bbox: d_batch[0]})

        # Save the generated images
        save_images(samples, [self.batch_size, 2], "./{}/train_{:02d}_{:04d}.png".format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args, sample_step=200, save_step=200):
        """
        Train GAN
        Data to generator -
        * Normal Xrays - 256x256, 1 channel - 10k - read in as normal
        * Bounding box defect - 32x32, 3 channels - 10k - read in as defect

        Data to discriminator -
        * Abnormal Xrays - 256x256, 1 channel - 5k - read in as abnormal
        """
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        print('Init')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_summary = tf.summary.merge([self.d__summary, self.fake_abnormal_summary,
                                           self.d_loss_fake_summary, self.g_loss_summary])
        self.d_summary = tf.summary.merge([self.d_summary, self.d_loss_real_summary, self.d_loss_summary])

        counter = 1
        start_time = time.time()

        # reading in generator data
        # read_data returns a DataSet object with next_batch method
        normal = read_data('normal/')
        defect = read_data('defect/')

        # reading in discriminator data
        abnormal = read_data('abnormal/')

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            batch_idxs = defect.num_examples // self.batch_size

            for idx in xrange(0, batch_idxs):
                print("\nStarting step {:4d} of epoch {:2d}".format(idx, epoch))
                # batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch = [load_data(batch_file) for batch_file in batch_files]
                # batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

                print('\nGetting batches')
                normal_batch_togen = normal.next_batch(self.batch_size)
                defect_batch_togen = defect.next_batch(self.batch_size)
                abnormal_batch_todisc = abnormal.next_batch(self.batch_size)

                input_feed_dict = {self.normal_xrays: normal_batch_togen,
                                   self.input_bbox: defect_batch_togen,
                                   self.fake_abnormal: abnormal_batch_todisc}
                print('\nUpdating G network')
                _, summary_str = self.sess.run([g_optim, self.g_summary],
                                               feed_dict=input_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Run critic five times to make sure that d_loss does not go to zero (different from paper)
                print('Updating D network 5 times')
                for i in range(5):
                    _, summary_str = self.sess.run([d_optim, self.d_summary],
                                                   feed_dict=input_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(feed_dict=input_feed_dict)
                errD_real = self.d_loss_real.eval(feed_dict=input_feed_dict)
                errD = self.d_loss.eval(feed_dict=input_feed_dict)

                errG = self.g_loss.eval(feed_dict=input_feed_dict)

                counter += 1
                print("\nEpoch: [%2d] [%4d/%4d] time: %4.4f, Disc loss: %.8f + %.8f = %.8f, Gen loss: %.8f"
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_real, errD_fake, errD, errG))

                # See sample images every sample_step steps
                if np.mod(counter, sample_step) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                # Save model after save_step steps
                if np.mod(counter, save_step) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "GAN_model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print("Succesfully restored from {}".format(ckpt_name))
            return True
        else:
            return False

    def test(self, args, stats_file='model_stats'):
        """Test GAN"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # read_data returns a DataSet object with next_batch method
        normal = read_data('normal')
        defect = read_data('defect')

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed. Continuing")

        batch_idxs = defect.num_examples // self.batch_size
        batch_end_time = time.time()
        for i in batch_idxs:
            print('Generating samples for batch %2d, time: %4.4f' % (i, batch_end_time - time.time()))

            normal_test_batch = normal.next_batch(self.batch_size)
            defect_test_batch = defect.next_batch(self.batch_size)
            file_combinations = zip(normal_test_batch[1], defect_test_batch[1])

            samples = self.sess.run(self.generated_sample,
                                    feed_dict={self.normal_xrays: normal_test_batch[0],
                                               self.input_bbox: defect_test_batch[0]})
            image_filename = './{}/test_{:04d}.png'.format(args.test_dir, i)
            save_images(images=samples, size=[self.batch_size, 2], image_path=image_filename)
            save_stats(filename=stats_file, image_name=image_filename, labels=file_combinations)
            batch_end_time = time.time()
