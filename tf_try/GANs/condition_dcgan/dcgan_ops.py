# -*- coding: utf-8 -*-
"""
Basic operations of DCGAN graph.
"""

import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import batch_norm

def lrelu(x, alpha = 0.2, name = 'LeakyReLU'):
    """
    LeakyReLU.

    Args:
        x: Input data.
        alpha: Coefficient of x when x < 0.
    Return:
        lre: Output after activation.
    """
    lre = tf.maximum(x, alpha * x)
    return lre

def conv(data, input_units, output_units, kernels, stride, name):
    """
    Convolution layer.

    Args:
        data: Input data.
        input_units, output_units: Neuron numbers.
        kernels: Kernel height.
        stride: Convolution stride.
        name: Layer's name.
    Return:
        con: Result of conv.
        w: Weights.
    """
    with tf.name_scope(name):
        w = tf.Variable(
            tf.truncated_normal([1, kernels, input_units, output_units],
                                stddev=1.0 / math.sqrt(float(input_units + output_units) / 2.0)),
            name='weights')
        b = tf.Variable(tf.zeros(output_units),
                             name='biases')
        con = b + tf.nn.conv2d(data, w,
                                strides=[1, stride, stride, 1],
                                padding='VALID')
    return con, w

def de_conv(data, input_uints, output_uints, kernels, stride, name):
    """
    Deconvolution layer.
    Args:
        data:
        input_uints:
        output_uints:
        kernels:
        stride:
        name:
    Return:

    """

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse = reuse, updates_collections=None)