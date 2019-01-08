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

def ppr_block(inp, kernels, stride, bands_size):
    channels = len(kernels)
    for i in range(channels):
        with tf.name_scope('conv_' + str(i)):
            conv_weights = tf.Variable(
                tf.truncated_normal([1, kernels[i], 1, 1],
                                    stddev=1.0 / math.sqrt(float(1))),
                name='weights')
            conv_biases = tf.Variable(tf.zeros(1),
                                       name='biases')
            x_data = tf.reshape(inp, [-1, 1, bands_size * stride, 1])
            conv = tf.sigmoid(conv_biases + tf.nn.conv2d(x_data, conv_weights,
                                                           strides=[1, stride, stride, 1],
                                                           padding='VALID'))
            print('conv shape', conv.get_shape())
        with tf.name_scope('deconv_' + str(i)):
            deconv_weights = tf.Variable(
                tf.truncated_normal([1, kernels[i], 1, 1],
                                    stddev=1.0 / math.sqrt(float(1))),
                name='weights')
            deconv_biases = tf.Variable(tf.zeros(1),
                                      name='biases')
            deconv = deconv_biases + tf.nn.conv2d_transpose(conv, deconv_weights,
                                                             output_shape=tf.stack([tf.shape(x_data)[0], 1, bands_size, 1]),
                                                             strides=[1, stride, stride, 1], padding='VALID')

            print('deconv_shape', deconv.get_shape())
        if i == 0:
            out = deconv
        else:
            out = tf.concat([out, deconv], 3)

    print('concat_out_shape', out.get_shape())
    assert out.get_shape()[3] == channels
    return out


def inference(dataset, conv1_kernel, conv1_stride, fc_uints, num_class, bands_size):
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
    with tf.name_scope('ppr'):
        conv1 = ppr_block(dataset, conv1_kernel, conv1_stride, bands_size)

        # mpool
    with tf.name_scope('mpool'):
        mpool = tf.nn.max_pool(conv1, ksize=[1, 1, 2, 1],
                               strides=[1, 1, 2, 1],
                               padding='VALID')

    # fc
    with tf.name_scope('fc'):
        #input_uints = int((bands_size - conv1_kernel + 1) / 2)
        print('mpool_shape', mpool.get_shape())
        x = mpool.get_shape()[2].value
        y = mpool.get_shape()[3].value
        input_uints = x * y
        fc_weights = tf.Variable(
            tf.truncated_normal([input_uints, fc_uints],
                                stddev=1.0 / math.sqrt(float(input_uints))),
            name='weights')
        fc_biases = tf.Variable(tf.zeros([fc_uints]),
                             name='biases')
        mpool_flat = tf.reshape(mpool, [-1, input_uints])
        #fc = tf.nn.relu(tf.matmul(mpool_flat, weights) + biases)
        fc = tf.sigmoid(tf.matmul(mpool_flat, fc_weights) + fc_biases)

    # softmax regression
    with tf.name_scope('softmax_re'):
        softmax_weights = tf.Variable(
            tf.truncated_normal([fc_uints, num_class],
                                stddev=1.0 / math.sqrt(float(fc_uints))),
            name='weights')
        softmax_biases = tf.Variable(tf.zeros([num_class]),
                             name='biases')
        # softmax_re = tf.matmul(fc, softmax_weights) + softmax_biases
        softmax_re = tf.nn.softmax(tf.matmul(fc, softmax_weights) + softmax_biases)
        # print('softmax size: hhhhhhhhhhh')
        # print(softmax_re.get_shape())
    return softmax_re # , conv1_weights, conv1_biases, fc_weights, fc_biases, softmax_weights, softmax_biases #, conv1, mpool, fc


def loss(softmax_re, labels):
    """Calculates the loss.
    
    Args:
        softmax_re: net forward output tensor, float - [batch_size, num_class]
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
       softmax_re: net forward output tensor, float - [batch_size, num_class]
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
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
