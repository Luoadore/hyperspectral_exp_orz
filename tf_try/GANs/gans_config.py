# coding: utf-8
"""
Using generative adversarial networks (GAN) to generate hyperspectral data from a noise distribution.
"""

import tensorflow as tf


# training parameters
iterations = 100000
batch_size = None
learning_rate = 0.0002

# Network parameters
data_dim = 176 # single-pixel
generator_hidden_uints = 256
discriminator_hidden_uints = 256
noise_dim = 100 # noise data points

# A custom initialization (Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape = shape, stddev = 1. / tf.sqrt(shape[0] / 2.))

# Store layers weights and bias
weights = {
    'gen_hidden1' : tf.Variable(glorot_init([noise_dim, generator_hidden_uints])),
    'gen_out': tf.Variable(glorot_init([generator_hidden_uints, data_dim])),
    'disc_hidden': tf.Variable(glorot_init([data_dim, discriminator_hidden_uints])),
    'disc_out': tf.Variable([discriminator_hidden_uints, 1]),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([generator_hidden_uints])),
    'gen_out': tf.Variable(tf.zeros([data_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([discriminator_hidden_uints])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# Generator
def generator(noise):
    """
    Generate data sample from a noise.

    Arg:
        noise: Input noise.
    Return:
        gen_data: Generate output data. Size as data_dim.
    """
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(noise, weights['gen_hidden1']), biases['gen_hidden1']))
    gen_data = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, weights['gen_out']), biases['gen_out']))
    print('Generate data done.')
    return gen_data

# Discriminator
def discriminator(gen_data):
    """
    Discriminator of real data and fake data.

    Arg:
        gen_data: Data generated from generator().
    Return:
        single_output: Represents the data is real and fake. 0 means fake and 1 means real.
    """
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(gen_data, weights['disc_hidden1']), biases['disc_hidden1']))
    single_output = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, weights['disc_out']), biases['disc_out']))
    print('Discriminator.')
    return single_output