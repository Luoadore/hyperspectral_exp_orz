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


def inference(dataset, conv1_uints, conv1_kernel, conv1_stride, conv2_uints, conv3_uints, fc1_uints, fc2_uints):
    """Build the model up to where it may be used for inference.

    Args:
        dataset: Data placeholder, from inputs()
        conv1_units, conv2_uints, conv3_uints: Size of the convolution layer
        conv1_kernel: kernel size, which sharing same weights
        conv1_stride: stride size
        fc1_units, fc2_uints: Size of the fully connection layer

    Returns:
        softmax_re: Output tensor with the computed logits
    """
    # conv1
    with tf.name_scope('conv1'):
        conv1_weights = tf.Variable(
            tf.truncated_normal([1, conv1_kernel, 1, conv1_uints],
                                stddev=1.0 / math.sqrt(float(conv1_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros(conv1_uints),
                             name='biases')
        x_data = tf.reshape(dataset, [-1, 1, BANDS_SIZE, 1])  #这里注意之后邻域变化需要修改band size的值
        print(x_data.get_shape())
        #conv1 = tf.nn.relu(biases + tf.nn.conv2d(x_data, weights,
                                        #strides=[1, conv1_stride, conv1_stride, 1],
                                        #padding='VALID'))
        conv1 = tf.sigmoid(biases + tf.nn.conv2d(x_data, conv1_weights,
                                        strides=[1, conv1_stride, conv1_stride, 1],
                                        padding='VALID'))

        # mpool
    with tf.name_scope('mpool'):
        mpool = tf.nn.max_pool(conv1, ksize=[1, 1, 2, 1],
                               strides=[1, 1, 2, 1],
                               padding='VALID')

    # fc
    with tf.name_scope('fc'):
        #input_uints = int((BANDS_SIZE - conv1_kernel + 1) / 2)
        print(mpool.get_shape())
        x = mpool.get_shape()[2].value
        y = mpool.get_shape()[3].value
        input_uints = x * y
        fc_weights = tf.Variable(
            tf.truncated_normal([input_uints, fc_uints],
                                stddev=1.0 / math.sqrt(float(input_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc_uints]),
                             name='biases')
        mpool_flat = tf.reshape(mpool, [-1, input_uints])
        #fc = tf.nn.relu(tf.matmul(mpool_flat, weights) + biases)
        fc = tf.sigmoid(tf.matmul(mpool_flat, fc_weights) + biases)

    # softmax regression
    with tf.name_scope('softmax_re'):
        softmax_weights = tf.Variable(
            tf.truncated_normal([fc_uints, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc_uints))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        softmax_re = tf.nn.softmax(tf.matmul(fc, softmax_weights) + biases)

    return softmax_re, conv1_weights, fc_weights, softmax_weights, conv1, mpool, fc


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
        accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

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
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
