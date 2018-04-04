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

def conv(data, output_units, kernels, stride, name):
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
            tf.truncated_normal([1, kernels, data.get_shape()[-1], output_units],
                                stddev=1.0 / math.sqrt(float(data.get_shape()[-1] + output_units) / 2.0)),
            name='weights')
        b = tf.Variable(tf.zeros(output_units),
                             name='biases')
        con = b + tf.nn.conv2d(data, w,
                                strides=[1, stride, stride, 1],
                                padding='SAME')
    return con, w

def de_conv(data, output_shape, kernels, stride, name, with_w=False):
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
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.Variable(tf.truncated_normal([1, kernels, output_shape[-1], data.get_shape()[-1]],
                            stddev=1.0 / math.sqrt(float(data.get_shape()[-1] + output_shape[-1]) / 2.0)),
                        name = 'w')
        b = tf.Variable(tf.zeros(output_shape[-1]),
                             name='biases')
        deconv = b + tf.nn.conv2d_transpose(data, w, output_shape=output_shape,
                                            strides = [1, stride, stride, 1])

        if with_w:
            return deconv, w, b
        else:
            return deconv

def fully_connect(data, output_size, scope=None, with_w=False):
    shape = data.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.Variable(tf.truncated_normal([shape[1], output_size],
                                stddev=1.0 / math.sqrt(float(shape[1]))),
                 name = "Matrix")
        b = tf.Variable(tf.zeros([output_size]),
                             name='biases')
        if with_w:
              return tf.matmul(data, matrix) + b, matrix, b
        else:
              return tf.matmul(data, matrix) + b

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse = reuse, updates_collections=None)