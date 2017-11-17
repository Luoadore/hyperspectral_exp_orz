# coding: utf-8
"""
This net extend from original net for hyperspectral image classification use CNN;
The computational formula of convolutional layer's kernel numbers is defined as followed:
    kernel_num = (band_size - kernel_size) / neighbors + 1
It contains 7 layers:
    Input 1 @ n1 * 1
    conv1 kernel_num  @ n2 * 1
    reshape 1 @ n2 * n2(equal to kernel_num)
    conv2 64 @ n3 * n3
    maxPool1 64 @ n4 * n4
    conv3 128 @ n5 * n5
    maxpool2 128 @ n6 * n6
    fc1 1024
    fc2 100
    output num_class
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import math

# labels
NUM_CLASSES = 13
# hyperspectral bands
BANDS_SIZE = 176


def inference(dataset, conv1_uints, conv1_kernel, conv1_stride, conv2_uints, conv3_uints, conv4_uints, fc1_uints, fc2_uints):
    """Build the model up to where it may be used for inference.

    Args:
        dataset: Data placeholder, from inputs()
        conv1_units, conv2_uints, conv3_uints, conv4_uints: Size of the convolution layer
        conv1_kernel: kernel size, which sharing same weights
        conv1_stride: stride size
        fc1_units, fc2_uints: Size of the fully connection layer

    Returns:
        softmax_re: Output tensor with the computed logits
    """
    # conv1
    with tf.name_scope('conv1'):
        weights = tf.Variable(
            tf.truncated_normal([1, conv1_kernel, 1, conv1_uints],
                                stddev=1.0 / math.sqrt(float(conv1_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros(conv1_uints),
                             name='biases')
        x_data = tf.reshape(dataset, [-1, 1, BANDS_SIZE * conv1_stride, 1])  #这里注意之后邻域变化需要修改band size的值
        print(x_data.get_shape())
        #conv1 = tf.nn.relu(biases + tf.nn.conv2d(x_data, weights,
                                        #strides=[1, conv1_stride, conv1_stride, 1],
                                        #padding='VALID'))
        conv1 = tf.sigmoid(biases + tf.nn.conv2d(x_data, weights,
                                        strides=[1, conv1_stride, conv1_stride, 1],
                                        padding='VALID'))
        print(conv1.get_shape())

    # reshape
    with tf.name_scope('reshape'):
        reshape = tf.reshape(conv1, [-1, conv1_uints, conv1_uints, 1])
        print(reshape.get_shape())

    # conv2
    with tf.name_scope('conv2'):
        weights = tf.Variable(
            tf.truncated_normal([3, 3, 1, conv2_uints],
                                stddev=1.0 / math.sqrt(float(conv2_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros(conv2_uints),
                             name='biases')
        conv2 = tf.sigmoid(biases + tf.nn.conv2d(reshape, weights,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID'))

    # mpool
    with tf.name_scope('mpool1'):
        mpool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # conv3
    with tf.name_scope('conv3'):
        weights = tf.Variable(
            tf.truncated_normal([3, 3, conv2_uints, conv3_uints],
                                stddev=1.0 / math.sqrt(float(conv3_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros(conv3_uints),
                             name='biases')
        conv3 = tf.sigmoid(biases + tf.nn.conv2d(mpool1, weights,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID'))

    # mpool2
    with tf.name_scope('mpool2'):
        mpool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # conv4
    with tf.name_scope('conv4'):
        weights = tf.Variable(
            tf.truncated_normal([3, 3, conv3_uints, conv4_uints],
                                stddev=1.0 / math.sqrt(float(conv4_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros(conv4_uints),
                             name='biases')
        conv4 = tf.sigmoid(biases + tf.nn.conv2d(mpool2, weights,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID'))

    # mpool2
    with tf.name_scope('mpool3'):
        mpool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')

    # fc1
    with tf.name_scope('fc1'):
        #input_uints = int((BANDS_SIZE - conv1_kernel + 1) / 2)
        print(mpool2.get_shape())
        print(mpool3.get_shape())
        x = mpool3.get_shape()[2].value
        y = mpool3.get_shape()[3].value
        z = mpool3.get_shape()[1].value
        input_uints = x * y * z
        weights = tf.Variable(
            tf.truncated_normal([input_uints, fc1_uints],
                                stddev=1.0 / math.sqrt(float(input_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc1_uints]),
                             name='biases')
        mpool_flat = tf.reshape(mpool3, [-1, input_uints])
        mpool_flat_drop = tf.nn.dropout(mpool_flat, keep_prob = 0.5)
        #fc1 = tf.nn.relu(tf.matmul(mpool_flat_drop, weights) + biases)
        fc1 = tf.sigmoid(tf.matmul(mpool_flat_drop, weights) + biases)

    # fc2
    with tf.name_scope('fc2'):
        weights = tf.Variable(
            tf.truncated_normal([fc1_uints, fc2_uints],
                                stddev=1.0 / math.sqrt(float(fc1_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc2_uints]),
                             name='biases')
        fc1_drop = tf.nn.dropout(fc1, keep_prob = 0.5)
        #fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights) + biases)
        fc2 = tf.sigmoid(tf.matmul(fc1_drop, weights) + biases)

    # softmax regression
    with tf.name_scope('softmax_re'):
        weights = tf.Variable(
            tf.truncated_normal([fc2_uints, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc2_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        softmax_re = tf.nn.softmax(tf.matmul(fc2, weights) + biases)

    return softmax_re


def loss(softmax_re, labels):
    """Calculates the loss.

    Args:
        softmax_re: net forward output tensor, float - [batch_size, NUM_CLASSES]
        labels: labels tensor,

    Return:
        entroy: Loss tensor of type float
    """

    # loss
    with tf.name_scope('loss'):

        log_tf = tf.log(softmax_re, name = 'log_name')
        #log_tf = tf.log(softmax_re + 1e-10, name = 'log_name') 据说可以解决loss nan的问题
        entroy = tf.reduce_mean(
            -tf.reduce_sum(labels * log_tf,
                           reduction_indices=[1]))

    return entroy


def acc(softmax_re, labels):
    """Calculates the accuracy.

   Args:
       softmax_re: net forward output tensor, float - [batch_size, NUM_CLASSES]
       labels: labels tensor,

   Return:
       accuracy: classification accuracy
   """
    # accuracy
    with tf.name_scope('accuracy'):
        correct_predicition = tf.equal(tf.argmax(softmax_re, 1), tf.argmax(labels, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))
        accuracy = tf.reduce_sum(tf.cast(correct_predicition, tf.float32))

    return accuracy


def training(loss, learning_rate):
    """Sets up the training Ops.

    Args:
        loss: Loss tensor, from loss_acc()
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training
    """

    # Add a scalar summary for the snapshot loss. Creates a summarizer to track the loss over time in TensorBoard.
    with tf.name_scope('training'):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
