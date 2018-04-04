# -*- coding: utf-8 -*-
"""
Condition deconvolution GAN model.
"""

import tensorflow as tf
import dcgan_ops as ops
import numpy as np

class CGAN(object):

    def __init__(self, data_ob, sample_dir, learn_rate, batch_size, z_dim, log_dir, model_path):
        self.data_ob = data_ob
        self.sample_dir = sample_dir
        self.output_size = self.data_ob.dims
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = self.data_ob.classes
        self.log_dir = log_dir
        self.model_path = model_path
        self.channel = self.data_ob.shape[2]
        self.sample = tf.placeholder(tf.float32, [None, self.output_size, 1, self.channel])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim])

    def gern_net(self, z, y):
        #########################################################################
        # 没写完
        with tf.variable_scope('generator') as scope:
            batch = y[0]
            y = tf.reshape(y, shape=[-1, self.y_dim])
            yb = tf.reshape(y, shape=[-1, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)
            c1, c2 = self.output_size / 4, self.output_size / 2

            # 10 stand for the num of labels
            d1 = tf.nn.relu(ops.batch_normal(ops.fully_connect(z, output_size=1024, scope='gen_fully'), scope='gen_bn1'))

            d1 = tf.concat([d1, y], 1)

            d2 = tf.nn.relu(ops.batch_normal(ops.fully_connect(d1, output_size=7*7*2*64, scope='gen_fully2'), scope='gen_bn2'))

            d2 = tf.reshape(d2, [batch, 1, c1, 64 * 2])
            d2 = ops.conv_cond_concat(d2, yb)

            d3 = tf.nn.relu(ops.batch_normal(ops.de_conv(d2, output_shape=[batch, 1, c2, 128], name='gen_deconv1'), scope='gen_bn3'))

            d3 = ops.conv_cond_concat(d3, yb)

            d4 = ops.de_conv(d3, output_shape=[batch, 1, self.output_size, self.channel], name='gen_deconv2')

            return tf.nn.sigmoid(d4)

    def build_model(self):
        self.fake = self.gern_net(self.z, self.y)