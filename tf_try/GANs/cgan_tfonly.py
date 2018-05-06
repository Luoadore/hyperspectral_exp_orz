# -*- coding: utf-8 -*-
""" This is an implementation of conditional generative adversarial net using tensorflow (not tfgan)."""
import tensorflow as tf
import numpy as np
import os
import random
import scipy.io as sio
import math
import fid
from random import shuffle

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# param config
flags = tf.app.flags
flags.DEFINE_integer('iter', 1000000, 'Iteration to train.')
flags.DEFINE_integer('batch_size', 100, 'The size of each batch.')
flags.DEFINE_string('model_path', './model/cgan.model', 'Save model path.')
flags.DEFINE_boolean('is_train', True, 'Train or test.')
flags.DEFINE_integer('test_number', 0, 'The class that want to generate, if None, generate randomly.')
flags.DEFINE_string('train_dir', '/media/luo/result/hsi_gan_result/KSC/hsi_data0.mat', 'Train data path.')
FLAGS = flags.FLAGS

# load data
data = sio.loadmat(FLAGS.train_dir)
spectral_data = data['data']
spectral_labels = data['label']

def next_batch(batch_size, num_step, data_set, label_set):
    """Return the next 'batch_size' examples from the data set.

    Args:
        batch_size: The batch size
        num_step: The step of iteration
        data_set: The data set
        label_set: The correspoding label set

    Return:
        batch_data: Next batch size data
        batch_label: Next batch size correspoding label
    """
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]
    batch_label = label_set[start_index : end_index]
    return batch_data, batch_label

Z_dim = 100
X_dim = spectral_data.shape[1]
y_dim = spectral_labels.shape[1]
h_dim = 128  

def xaiver_init(size):
    in_dim = size[0]
    return tf.random_normal(shape = size, stddev = 1. / tf.sqrt(in_dim / 2.))

"""Discriminator"""
X = tf.placeholder(tf.float32, shape = [None, X_dim])
y = tf.placeholder(tf.float32, shape = [None, y_dim])

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

"""Generator"""
Z = tf.placeholder(tf.float32, shape = [None, Z_dim])

G_W1 = tf.Variable(xaiver_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

G_W2 = tf.Variable(xaiver_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape = [X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z, y):
    inputs = tf.concat(axis = 1, values = [z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    print('generating...')

    return G_prob

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size = [m, n])

def acc(prob_r, prob_f):
    r = tf.cast(prob_r > 0.5, tf.float32)
    f = tf.cast(prob_f <= 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.concat([r, f], 0))
    return accuracy

# Train set
G_sample = generator(Z, y)
D_real, D_logit_real = discriminator(X, y)
D_fake, D_logit_fake = discriminator(G_sample, y)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_real, labels = tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_logit_fake, labels = tf.ones_like(D_logit_fake)))
D_acc = acc(D_real, D_fake)
tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('D_acc', D_acc)
summary_op = tf.summary.merge_all()

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = theta_G)

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

    n_samples = 16
    X_sample = spectral_data[: 16]
    print(np.shape(X_sample))
    Z_sample = sample_Z(n_samples, Z_dim)
    y_sample = np.zeros(shape = [n_samples, y_dim])
    if FLAGS.test_number != None:
        y_sample[:, FLAGS.test_number] = 1
    else:
        for i in range(n_samples):
            y_sample[i][random.randint(0, y_dim - 1)] = 1

    print(y_sample)

    samples = sess.run(G_sample, feed_dict = {Z: Z_sample, y: y_sample})
    D_value = sess.run(D_real, feed_dict = {X: samples, y: y_sample})
    D_real_value = sess.run(D_real, feed_dict = {X: X_sample, y: y_sample})
    print('Generate and saved samples.')
    return samples, D_value, D_real_value

def caculate_fid(real_samples, Z_dim, y_mb, G_sample):
    """
    Args:
        real_samples: Samples of per batch.
        Z_dim: Dimension of noise.
        y_mb: Onehot label of per batch.
        G_sample: Tensorflow operation.
    Return:
        : [D_G fid, D_D fid]
    """
    n_fid = [[], []]
    for i in range(20):
        shuffle(real_samples)
        l = len(real_samples) // 2
        real_1 = real_samples[: l]
        real_2 = real_samples[l :]
        Z_sample = sample_Z(y_mb.shape[0], Z_dim)
        g_sample = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_mb})
        fake = g_sample[: l]
        mu_r1, sigma_r1 = fid.calculate_statistics(real_1)
        mu_r2, sigma_r2 = fid.calculate_statistics(real_2)
        mu_f, sigma_f = fid.calculate_statistics(fake)
        n_fid[0].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_f, sigma_f))
        n_fid[1].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_r2, sigma_r2))
    return np.mean(n_fid, axis=1)

def main(_):
    if FLAGS.is_train:
        g_loss_value = []
        d_loss_value = []
        fid_value = []
        for it in range(FLAGS.iter):
            for i in range(1):
                X_mb, y_mb = next_batch(FLAGS.batch_size, it * 10 + i, spectral_data, spectral_labels)

                Z_sample = sample_Z(y_mb.shape[0], Z_dim)
                _, D_loss_curr, D_acc_curr = sess.run([D_solver, D_loss, D_acc], feed_dict = {X: X_mb, Z: Z_sample, y: y_mb})
                if it % 1000 == 0:
                    d_loss_value.append(D_loss_curr)
            # Z_sample = sample_Z(y_mb.shape[0], Z_dim)
            _, G_loss_curr, g_sample = sess.run([G_solver, G_loss, G_sample], feed_dict = {Z: Z_sample, y: y_mb})

            if it % 1000 == 0:
            # train_samples_gen = test()
            # sio.savemat('./train/' + str(i) + 'data.mat', {'data': train_samples_gen})
                print('Iter: {}'.format(it))
                print('D_loss: ' + str(D_loss_curr))
                print('G_loss: ' + str(G_loss_curr))
                print('D_acc', D_acc_curr)
                g_loss_value.append(G_loss_curr)
                saver.save(sess, FLAGS.model_path)
                summary_str = sess.run(summary_op, feed_dict = {X: X_mb, Z: Z_sample, y: y_mb})
                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()

                f = caculate_fid(X_mb, Z_dim, y_mb, G_sample)
                print('FID: ', f)
                fid_value.append(f)
        sio.savemat('./train/data' + str(FLAGS.test_number) + '_loss.mat', {'D_loss': d_loss_value, 'G_loss': g_loss_value, 'fid': fid_value})

    else:
        test_samples_gen, d_value, d_real_value = test()
        sio.savemat('./test/data' + str(FLAGS.test_number) + '.mat', {'data': test_samples_gen, 'd_value': d_value, 'd_real_value': d_real_value})

    sess.close()

if __name__ == '__main__':
    tf.app.run()