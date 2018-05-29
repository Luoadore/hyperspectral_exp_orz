# coding: utf-8
"""
Validation. Generated samples of class a feed into discriminator of class b.
"""

import tensorflow as tf
import wgan_tf as w
import scipy.io as sio
import numpy as np

# load data
file_dir_1 = '/media/luo/result/hsi-wgan/test/exp_0/data0.mat'
file_dir_2 = '/media/luo/result/hsi-wgan/test/exp_1/data1.mat'
data = sio.loadmat(file_dir_1)
g_sample_1 = data['g_sample']
data = sio.loadmat(file_dir_2)
g_sample_2 = data['g_sample']
print(np.shape(g_sample_1))
print(np.shape(g_sample_2))

g = tf.placeholder(tf.float32, shape=[None, 176])
D_fake = w.discriminator(g)
_, acc_real, _ = w.acc(D_fake)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, '/media/luo/result/hsi-wgan/test/exp_1')
fake_1, realAcc_1 = sess.run([D_fake, acc_real], feed_dict={g: g_sample_1})
fake_2, realAcc_2 = sess.run([D_fake, acc_real], feed_dict={g: g_sample_2})
print('fake1: ', fake_1)
print('fake_2: ', fake_2)
print('acc1 & acc2:', realAcc_1, realAcc_2)
sess.close()
