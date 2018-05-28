# coding: utf-8

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import scipy.io as sio
import numpy as np
import sys
sys.path.append('/media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/CNN')
import original_cnn as oc
import train_original as to

data = sio.loadmat('/media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/model/wgan.model/data0.mat')
hsi_data = data['g_sample']
data = sio.loadmat('/media/luo/result/hsi_gan_result/KSC/hsi_data0.mat')
hsi_label = data['label']
hsi_label = hsi_label[0 : len(hsi_data), :]
print(np.shape(hsi_label))
print(len(hsi_data))

with tf.Graph().as_default():
    data_placeholder, label_placeholder = to.placeholder_inputs(100)
    softmax = oc.inference(data_placeholder, 20, 19, 1, 100)
    pred, correct = oc.acc(softmax, label_placeholder)
    saver = tf.train.Saver()

    # restore model
    sess = tf.Session()
    saver.restore(sess, '/media/luo/result/hsi_gan_result/KSC/model/checkpoint-19999')

    acc, prediction = to.do_eval(sess, correct, data_placeholder, label_placeholder, hsi_data, hsi_label,
                                          softmax)
    sio.savemat('/media/luo/result/hsi_gan_result/KSC/predic_data0.mat', {'acc': acc, 'prediction': prediction})
    sess.close()