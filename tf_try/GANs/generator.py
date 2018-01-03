# coding: utf-8
"""Generator fixed."""

import tensorflow as tf
import scipy.io as sio
import numpy as np

def generator(noise, params):
    """
    Generate data sample from a noise.

    Arg:
        noise: Input noise.
        params: Parameters of generator, from train_gain.
    Return:
        gen_data: Generate output data. Size as data_dim.
    """
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(noise, weights['gen_hidden1']), biases['gen_hidden1']))
    gen_data = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, weights['gen_out']), biases['gen_out']))
    print('Generate data done.')
    return gen_data

if __name__ == '__main__':
    gan_data = sio.loadmat('F:\hsi_result\gan\gan_data.mat')
    gen_param = gan_data['gen_params']
    hsi_data = gan_data['data_set']

    gen_samples = []
    for i in range(len(hsi_data)):
        gen_samples.append([])

    for classes, eachclass in enumerate(hsi_data):
        noise = np.random.uniform(-1., 1., size=[len(eachclass), gc.noise_dim])