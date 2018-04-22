# -*- coding: utf-8 -*-

import tensorflow as tf

from params import param


def inference(datas, labels):
    feature = param.hsi.feature_number
    vector = param.vector
    channel = param.channel
    length = param.length

    with tf.name_scope('Conv'):
        kernel1 = param.hsi.kernel1
        stride1 = param.hsi.stride1
        weights = tf.Variable(tf.truncated_normal([1, kernel1, 1, channel * vector], stddev=1/16), name='weights')
        biases = tf.Variable(tf.zeros(channel * vector), name='biases')
        x_data = tf.reshape(datas, [-1, 1, feature, 1])
        conv1 = tf.nn.conv2d(x_data, weights, strides=[1, 1, stride1, 1], padding='VALID') + biases
        width1 = int((feature - kernel1) // stride1 + 1)  # int
        conv1 = tf.reshape(conv1, [-1, 1, width1, channel * vector])
        conv1 = tf.tanh(conv1)

    with tf.name_scope('PrimaryCaps'):
        kernel2 = param.hsi.kernel2
        stride2 = param.hsi.stride2
        weights = tf.Variable(
            tf.truncated_normal([1, kernel2, channel * vector, channel * vector], stddev=1/16), name='weights')
        biases = tf.Variable(tf.zeros(channel * vector), name='biases')
        conv2 = tf.nn.conv2d(conv1, weights, strides=[1, 1, stride2, 1], padding='VALID') + biases
        width2 = int((width1 - kernel2) // stride2 + 1)  # int
        primary = tf.reshape(conv2, [-1, width2 * channel, vector, 1])
        primary = squash(primary)

    with tf.name_scope('DigitCaps'):
        primary = primary[:, :, tf.newaxis, :, :]
        digit = routing(primary, length)
        digit = tf.squeeze(digit, axis=1)

    with tf.name_scope('Masking'):
        v_length = tf.sqrt(tf.reduce_sum(tf.square(digit), axis=2, keep_dims=True) + 1e-9)
        v_length = tf.nn.softmax(v_length + 1e-9, dim=1)
        v_length = tf.squeeze(v_length)
        masked = tf.multiply(tf.squeeze(digit), tf.reshape(labels, [-1, param.hsi.class_number, 1]))
        masked = tf.reduce_sum(masked, axis=1)

    with tf.name_scope('Decoder'):

        with tf.name_scope('decoder-fc1'):
            weights = tf.Variable(tf.truncated_normal([length, 64], stddev=1/16), name='weights')
            biases = tf.Variable(tf.zeros(64), name='biases')
            fc1 = tf.matmul(masked, weights) + biases
            fc1 = tf.tanh(fc1)

        with tf.name_scope('decoder-fc2'):
            weights = tf.Variable(tf.truncated_normal([64, 128], stddev=1/16), name='weights')
            biases = tf.Variable(tf.zeros(128), name='biases')
            fc2 = tf.matmul(fc1, weights) + biases
            fc2 = tf.tanh(fc2)

        with tf.name_scope('decoder-fc3'):
            weights = tf.Variable(
                tf.truncated_normal([128, feature], stddev=0.01), name='weights')
            biases = tf.Variable(tf.zeros(feature), name='biases')
            fc3 = tf.matmul(fc2, weights) + biases
            decoded = tf.tanh(fc3)

    return v_length, decoded


def squash(vector):
    vec_squared = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar = vec_squared / (1 + vec_squared)
    vector_norm = scalar * vector / tf.square(vec_squared + 1e-9)
    return vector_norm


def routing(caps, length, iter_routing=param.routing):
    feature = caps.get_shape()[1].value
    b_ij = tf.constant(0, dtype=tf.float32, shape=[param.batch_size, feature, param.hsi.class_number, 1, 1])
    w = tf.get_variable(
        name='weight', dtype=tf.float32,
        shape=[1, feature, param.hsi.class_number, caps.shape[3], length],
        initializer=tf.random_normal_initializer(stddev=1/16))
    caps = tf.tile(caps, [1, 1, param.hsi.class_number, 1, 1])
    w_t = tf.tile(w, [param.batch_size, 1, 1, 1, 1])
    u_hat = tf.matmul(w_t, caps, transpose_a=True, name='u_hat')
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    for r_iter in range(iter_routing):
        with tf.name_scope('iter_' + str(r_iter)):
            c_ij = tf.nn.softmax(b_ij, dim=2)

            if r_iter < iter_routing - 1:
                s_j = tf.multiply(c_ij, u_hat_stopped)
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                v_j = squash(s_j)
                v_j_titled = tf.tile(v_j, [1, feature, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_j_titled, transpose_a=True)
                b_ij += u_produce_v

            else:
                s_j = tf.multiply(c_ij, u_hat)
                s_j = tf.reduce_sum(s_j, axis=1, keep_dims=True)
                v_j = squash(s_j)
                return v_j


def reconstruct_loss(decoded, datas):
    with tf.name_scope("reconstruct_loss"):
        squared = tf.square(decoded - datas)
        part_loss = tf.reduce_mean(squared)
    return part_loss


def margin_loss(v_length, labels, regular=None):
    import numpy as np
    with tf.name_scope("margin_loss"):
        max_l = tf.square(tf.maximum(0., param.m_plus - v_length))
        max_r = tf.square(tf.maximum(0., v_length - param.m_minus))
        c = labels * max_l + param.down_val * (1 - labels) * max_r
        if regular is None:
            pass
        else:
            p = np.sum(regular) / len(regular)
            regular = np.array(regular, ndmin=2)
            regular = p / regular
            c = c * regular
        part_loss = tf.reduce_mean(tf.reduce_sum(c, axis=1))
    return part_loss


def total_loss(v_length, decoded, labels, datas):
    margin = margin_loss(v_length, labels)
    reconstruct = reconstruct_loss(decoded, datas)
    loss = margin + param.regular_scale * reconstruct
    return loss


def acc(v_length, labels):
    with tf.name_scope('accuracy'):
        idx = tf.argmax(v_length, 1)
        target = tf.argmax(labels, 1)
        correct_prediction = tf.equal(idx, target)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def predict(v_length, labels):
    with tf.name_scope('predict'):
        idx = tf.argmax(v_length, 1)
        target = tf.argmax(labels, 1)
    return idx, target


def training(value, learning_rate, global_step=None):
    with tf.name_scope('training'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(value, global_step=global_step)
    return train_op


def get_graph(graph):
    with graph.as_default():
        data = tf.placeholder(tf.float32, [param.batch_size, param.hsi.feature_number], name='data')
        label = tf.placeholder(tf.float32, [param.batch_size, param.hsi.class_number], name='label')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        y, decoded = inference(data, label)

        margin = margin_loss(y, label)
        reconstruct = reconstruct_loss(decoded, data)
        total = total_loss(y, decoded, label, data)
        accuracy = acc(y, label)
        pred, target= predict(y, label)
        train_op = training(total, param.learning_rate, global_step)

        margin_summary = tf.summary.scalar('margin_loss', margin)
        reconstruct_summary = tf.summary.scalar('reconstruct_loss', reconstruct)
        total_summary = tf.summary.scalar('total_loss', total)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        summary_op = tf.summary.merge(
            [margin_summary, reconstruct_summary, total_summary, accuracy_summary])
        graph_dict = {
            'data': data, 'label': label, 'margin': margin, 'reconstruct': reconstruct, 'loss': total,
            'accuracy': accuracy, 'train': train_op, 'step': global_step, 'summary': summary_op,
            'prediction': [pred, target]
            }
        return graph_dict
