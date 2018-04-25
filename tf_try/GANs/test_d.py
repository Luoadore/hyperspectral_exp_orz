# -*- coding: utf-8 -*-
""" This is an implementation of conditional generative adversarial net using tensorflow (not tfgan)."""
import tensorflow as tf
import numpy as np
import os
import random
import scipy.io as sio
import math
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# param config
flags = tf.app.flags
flags.DEFINE_integer('iter', 2000, 'Iteration to train.')
flags.DEFINE_integer('batch_size', 32, 'The size of each batch.')
flags.DEFINE_string('model_path', './model/cgan.model', 'Save model path.')
flags.DEFINE_boolean('is_train', True, 'Train or test.')
flags.DEFINE_string('train_dir1', '/media/luo/result/hsi_gan_result/KSC/hsi_data0.mat', 'Train data path.')
flags.DEFINE_string('train_dir2', '/media/luo/result/hsi_gan_result/KSC/hsi_data1.mat', 'Train data path.')
FLAGS = flags.FLAGS

# load data
data1 = sio.loadmat(FLAGS.train_dir1)
spectral_data1 = data1['data']
spectral_labels1 = data1['label']
data2 = sio.loadmat(FLAGS.train_dir2)
spectral_data2 = data2['data']
spectral_labels2 = data2['label']
print(np.shape(spectral_data1))
print(np.shape(spectral_labels1))
print(np.shape(spectral_data2))
print(np.shape(spectral_labels2))

def shuffling(data_set, label_set):
    num = len(label_set)
    index = np.arange(num)
    shuffle(index)
    shuffled_data = []
    shuffled_label = []
    for i in range(num):
        shuffled_data.append(data_set[index[i]])
        shuffled_label.append(label_set[index[i]])
    # print('Shuffling done.')
    return shuffled_data, shuffled_label

def next_batch(batch_size, num_step, data_set, label_set):
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]
    batch_label = label_set[start_index : end_index]
    """if end_index - start_index != batch_size:
        index = batch_size - (end_index - start_index)
        batch_data = np.append(batch_data, data_set[: index])
        batch_label = np.append(batch_label, label_set[: index])"""
    return batch_data, batch_label

X_dim = spectral_data1.shape[1]
y_dim = spectral_labels1.shape[1]
h_dim = 128

def xaiver_init(size):
    in_dim = size[0]
    return tf.random_normal(shape = size, stddev = 1. / tf.sqrt(in_dim / 2.))

"""Discriminator"""
X1 = tf.placeholder(tf.float32, shape = [None, X_dim])
X2 = tf.placeholder(tf.float32, shape = [None, X_dim])
y1 = tf.placeholder(tf.float32, shape = [None, y_dim])
y2 = tf.placeholder(tf.float32, shape = [None, y_dim])

D_W1 = tf.Variable(xaiver_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

D_W2 = tf.Variable(xaiver_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape = [1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x, y):
    inputs = tf.concat(axis = 1, values = [x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    print('discriminating...')

    return D_prob, D_logit

def all_loss(D_logit_real, D_logit_fake):
    logit = tf.concat([D_logit_real, D_logit_fake], 0)
    label = tf.concat([tf.ones_like(D_logit_real), tf.zeros_like(D_logit_fake)], 0)
    a_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit, labels = label))
    return a_loss

def acc(prob_r, prob_f):
    """
    equal_r = [1 for x in prob_r if x > 0.5]
    equal_f = [1 for x in prob_f if x < 0.5]
    accuracy = float(sum(equal_r) + sum(equal_f)) / float(len(prob_r) + len(prob_f))"""
    r = tf.cast(prob_r > 0.5, tf.float32)
    f = tf.cast(prob_f <= 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.concat([r, f], 0))
    return accuracy

# Train set
D_real, D_logit_real = discriminator(X1, y1)
D_fake, D_logit_fake = discriminator(X2, y2)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
D_a_loss = all_loss(D_logit_real, D_logit_fake)
D_acc = acc(D_real, D_fake)

tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('D_a_loss', D_a_loss)
tf.summary.scalar('D_acc', D_acc)
summary_op = tf.summary.merge_all()

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
D_a_solver = tf.train.AdamOptimizer().minimize(D_a_loss, var_list = theta_D)

saver = tf.train.Saver()

sess = tf.Session()

if FLAGS.is_train:
    sess.run(tf.global_variables_initializer())
else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    #model_file=tf.train.latest_checkpoint(FLAGS.model_path)
    #saver.restore(sess, model_file)

summary_writer = tf.summary.FileWriter(FLAGS.model_path, sess.graph)

if os.path.exists(os.path.join(FLAGS.model_path + '.index')):
    saver.restore(sess, FLAGS.model_path)
    print('restore model...')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir('train/')
mkdir('test/')
mkdir('model/')

def test():
    pass

def main(_):
    if FLAGS.is_train:
        d_loss_value = []
        d_a_loss_value = []
        index = []
        d_acc = []
        for it in range(FLAGS.iter):
            X1_mb, y1_mb = next_batch(FLAGS.batch_size, it, spectral_data1, spectral_labels1)
            X2_mb, y2_mb = next_batch(FLAGS.batch_size, it, spectral_data2, spectral_labels2)
            X1_mb, y1_mb = shuffling(X1_mb, y1_mb)
            X2_mb, y2_mb = shuffling(X2_mb, y2_mb)
            """print(np.shape(X1_mb))
            print(np.shape(y1_mb))
            print(np.shape(X2_mb))
            print(np.shape(y2_mb))"""
            assert len(X2_mb) == len(y2_mb)
            _, _, D_loss_curr, D_a_loss_curr, d_r, d_f, D_acc_curr = sess.run([D_solver, D_a_loss, D_loss, D_a_loss, D_real, D_fake, D_acc],
                                                                  feed_dict = {X1: X1_mb, X2: X2_mb, y1: y1_mb, y2: y2_mb})
            if it % 10 == 0:
            # train_samples_gen = test()
            # sio.savemat('./train/' + str(i) + 'data.mat', {'data': train_samples_gen})
                d_loss_value.append(D_loss_curr)
                d_a_loss_value.append(D_a_loss_curr)
                index.append(it)
                print('Iter: {}'.format(it))
                print('D_loss: ' + str(D_loss_curr))
                print('D_a_loss:', D_acc_curr)
                print('r_acc:', D_acc_curr)
                d_acc.append(D_acc_curr)
                # print('f:',d_f)
                saver.save(sess, FLAGS.model_path)
                summary_str = sess.run(summary_op, feed_dict = {X1: X1_mb, X2: X1_mb, y1: y1_mb, y2: y2_mb})
                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()
        sio.savemat('./train/data_acc_loss.mat', {'D_loss': d_loss_value, 'D_a_loss': d_a_loss_value, 'd_acc': d_acc, 'iters': index})

    else:
        test_samples_gen = test()
        sio.savemat('./test/data' + '.mat', {'data': test_samples_gen})

    sess.close()

if __name__ == '__main__':
    tf.app.run()