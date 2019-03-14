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

# labels
NUM_CLASSES = 13
# hyperspectral bands
BANDS_SIZE = 176


def inference(dataset, conv1_uints, conv1_kernel, conv1_stride, fc_uints, is_training):
    """Build the model up to where it may be used for inference.
    
    Args:
        dataset: Data placeholder, from inputs()
        conv1_units: Size of the convolution layer
        conv1_kernel: kernel size, which sharing same weights
        conv1_stride: stride size
        fc_units:Size of the fully connection layer
    
    Returns:
        softmax_re: Output tensor with the computed logits
    """
    # conv1
    with tf.name_scope('conv1'):
        conv1_weights = tf.Variable(
            tf.truncated_normal([1, conv1_kernel, 1, conv1_uints],
                                stddev=1.0 / math.sqrt(float(conv1_uints))),
            name='weights', trainable=is_training)
        conv1_biases = tf.Variable(tf.zeros(conv1_uints),
                             name='biases', trainable=is_training)
        x_data = tf.reshape(dataset, [-1, 1, BANDS_SIZE * conv1_stride, 1])  #这里注意之后邻域变化需要修改band size的值, * 1, 5, 9
        print(dataset.get_shape())
        print(x_data.get_shape())
        #conv1 = tf.nn.relu(biases + tf.nn.conv2d(x_data, weights,
                                        #strides=[1, conv1_stride, conv1_stride, 1],
                                        #padding='VALID'))
        conv1 = tf.sigmoid(conv1_biases + tf.nn.conv2d(x_data, conv1_weights,
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
            name='weights', trainable=is_training)
        fc_biases = tf.Variable(tf.zeros([fc_uints]),
                             name='biases', trainable=is_training)
        mpool_flat = tf.reshape(mpool, [-1, input_uints])
        #fc = tf.nn.relu(tf.matmul(mpool_flat, weights) + biases)
        fc = tf.sigmoid(tf.matmul(mpool_flat, fc_weights) + fc_biases)

    # softmax regression
    with tf.name_scope('softmax_re'):
        softmax_weights = tf.Variable(
            tf.truncated_normal([fc_uints, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc_uints))),
            name='weights')
        softmax_biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        # softmax_re = tf.matmul(fc, softmax_weights) + softmax_biases
        softmax_re = tf.nn.softmax(tf.matmul(fc, softmax_weights) + softmax_biases)
        print('softmax size: hhhhhhhhhhh')
        print(softmax_re.get_shape())
    return softmax_re, fc # , conv1_weights, conv1_biases, fc_weights, fc_biases, softmax_weights, softmax_biases #, conv1, mpool, fc


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
       accuracy_num: Classification correct numbers
   """
    # accuracy
    with tf.name_scope('accuracy'):
        correct_predicition = tf.equal(tf.argmax(softmax_re, 1), tf.argmax(labels, 1))
        accuracy_num = tf.reduce_sum(tf.cast(correct_predicition, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(correct_predicition, tf.float32))

    return accuracy, accuracy_num


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
