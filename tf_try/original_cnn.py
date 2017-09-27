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
        softmax_re: Output tensor with the computed logits
    """
    #conv1
    with tf.name_scope('conv1'):
        weights = tf.Varibale(
            tf.truncated_normal([1, conv1_kernel, 1, conv1_uints],
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
        input_uints = (BANDS_SIZE - conv1_kernel + 1) / 2
        weights = tf.Varibale(
            tf.turncated_normal([input_uints, fc_uints],
                                stddev = 1.0 / math.sqrt(float(input_uints))),
            name = 'weights')
        biases = tf.Varibale(tf.zeros([fc_uints]),
                             name = 'biases')
        mpool_flat = tf.reshape(mpool, [-1, input_uints])
        fc = tf.nn.relu(tf.matmul(mpool_flat, weights) + biases)
    
    #softmax regression
    with tf.name_scope('softmax_re'):
        weights = tf.Varibale(
            tf.truncated_normal([fc_uints, NUM_CLASSES],
                                stddev = 1.0 / math.sqrt(float(fc_uints))),
            name = 'weights')
        biases = tf.Varibale(tf.zeros([NUM_CLASSES]),
                             name = 'biases')
        softmax_re = tf.nn.softmax(tf.matmul(fc, weights) + biases)
        
    return softmax_re
    
def loss_acc(softmax_re, labels):
    """Calculates the loss.
    
    Args:
        softmax_re: net forward output tensor, float - [batch_size, NUM_CLASSES]
        labels: labels tensor, 
        
    Return:
        entroy: Loss tensor of type float
    """
    
    #loss
    with tf.name_scope('loss'):
        entroy = tf.reduce_mean(
            -tf.reduce_sum(labels * tf.log(softmax_re),
            reduction_indices = [1]))
        
    return entroy
    
def acc(softmax_re, labels):
     """Calculates the accuracy.
    
    Args:
        softmax_re: net forward output tensor, float - [batch_size, NUM_CLASSES]
        labels: labels tensor, 
        
    Return:
        accuracy: classification accuracy
    """
    
    #accuarcy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(softmax_re, 1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    return accuracy
    
def training(loss, learning_rate):
    """Sets up the training Ops.
    
    Args:
        loss: Loss tensor, from loss_acc()
        learning_rate: The learning rate to use for gradient descent.
        
    Returns:
        train_op: The Op for training
    """
    
    #Add a scalar summary for the snapshot loss. Creates a summarizer to track the loss over time in TensorBoard.
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = optimizer.minimize(loss, global_step = global_step)
    
    return train_op