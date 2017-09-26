# coding: utf-8
"""
this net contains five layers for hyperspectral image classification use CNN:
    Input 1 @ n1 * 1
    conv1 20 @ n2 * 1
    maxPool 20 @ n3 * 1
    fc n4
    output n5
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import math

#labels
NUM_CLASSES = 10
#hyperspectral bands
BANDS_SIZE = 176 

def inference(dataset, conv1_uints, conv1_kernel, conv1_stride, mpool_uints, fc_uints):
    """Build the model up to where it may be used for inference.
    
    Args:
        dataset: Data placeholder, from inputs()
        conv1_units: Size of the convolution layer
        conv1_kernel: kernel size, which sharing same weights
        conv1_stride: stride size
        mpool_units: Size of the max pooling layer
        fc_units:Size of the fully connection layer
    
    Returns:
        softmax_linear: Output tensor with the computed logits
    """
    #conv1
    with tf.name_scope('conv1'):
        weights = tf.Varibale(
            tf.truncated_normal([1, conv1_kernel, 1, conv1_units],
                                stddev = 1.0 / math.sqrt(float(conv1_uints))),
            name = 'weights')
        biases = tf.Varibale(tf.zeros(conv1_uints),
                             name = 'biases')
        x_data = tf.reshape(dataset, [-1, conv1_kernel, 1, 1])
        conv1 = tf.nn.relu(tf.nn.conv2d(x_data, weights, 
                                          strides = [1, conv1_stride, conv1_stride, 1], 
                                          padding = 'VALID')) 
    
    #mpool
    with tf.name_scope('mpool'):
        mpool = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1],
                               strides = [1, 2, 2, 1], 
                               padding = 'VALID')
    
    #fc
    with tf.name_scope('fc'):
        weights = 