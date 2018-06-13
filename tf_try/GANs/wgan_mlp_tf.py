import tensorflow as tf

import numpy as np
import os
import scipy.io as sio
import math
import fid
from random import shuffle
import copy
import data_scaling as ds
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# param config
flags = tf.app.flags
flags.DEFINE_integer('iter', 10000, 'Iteration to train.')
flags.DEFINE_integer('batch_size', 516, 'The size of each batch.')
flags.DEFINE_string('model_path', './model/exp_11', 'Save model path.')
flags.DEFINE_boolean('is_train', True, 'Train or test.')
flags.DEFINE_integer('class_number', 11, 'The class that want to generate, if None, generate randomly.')
flags.DEFINE_string('train_dir', '/media/luo/result/hsi_gan_result/KSC/hsi_data11.mat', 'Train data path.')
FLAGS = flags.FLAGS

# load data
data = sio.loadmat(FLAGS.train_dir)
spectral_data = data['data']
spectral_labels = data['label']
spectral_data = ds.delete(spectral_data)
spectral_data = ds.scaling(spectral_data)

X_dim = spectral_data.shape[1]
z_dim = 100
h_dim_1 = 128
h_dim_2 = 128

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
    if end_index - start_index < 10:
        shuffling(data_set)
        start_index, end_index = 0, batch_size
    batch_data = data_set[start_index : end_index]
    batch_label = label_set[start_index : end_index]
    return batch_data, batch_label

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim_2]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim_2]))

D_W2 = tf.Variable(xavier_init([h_dim_2, h_dim_1]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim_1]))

D_W3 = tf.Variable(xavier_init([h_dim_1, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim_1]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim_1]))

G_W2 = tf.Variable(xavier_init([h_dim_1, h_dim_2]))
G_b2 = tf.Variable(tf.zeros(shape=[h_dim_2]))

G_W3 = tf.Variable(xavier_init([h_dim_2, X_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    # G_prob = tf.nn.sigmoid(G_log_prob)
    G_prob = tf.nn.relu(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    # gradients
    [dw1, db1] = tf.gradients(D_h1, [D_W1, D_b1])
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    [dw2, db2] = tf.gradients(D_h2, [D_W2, D_b2])
    out = tf.matmul(D_h2, D_W3) + D_b3
    out = tf.nn.sigmoid(out)
    [dw3, db3] = tf.gradients(out, [D_W3, D_b3])
    grad = [[dw1, db1], [dw2, db2], [dw3, db3]]
    return out, grad

G_sample = generator(z)
D_real, grad_real = discriminator(X)
D_fake, grad_fake = discriminator(G_sample)

def acc(D_fake):
    # accuracy_real = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) > 0.5, tf.float32))
    # accuracy_fake = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) <= 0.5, tf.float32))
    accuracy_real = tf.reduce_mean(tf.cast(D_fake > 0.5, tf.float32))
    accuracy_fake = tf.reduce_mean(tf.cast(D_fake < 0.5, tf.float32))
    return accuracy_real / 2, accuracy_fake / 2, D_fake

def shuffling(data):
    num = len(data)
    index = np.arange(num)
    shuffle(index)
    s_data = []
    for i in range(num):
        s_data.append(data[index[i]])
    return np.array(s_data)

def cal_fid(real_samples, G_sample):
    """
    Args:
        real_samples: Samples of per batch.
        G_sample: Tensorflow operation.
    Return:
        : [D_G fid, D_D fid]
    """
    if real_samples.shape[0] < 4:
        return 0, 0
    n_fid = [[], []]
    l = len(real_samples) // 2
    # print(type(real_samples))
    for i in range(5):
        mu_f, sigma_f = fid.calculate_statistics(G_sample[: l])
        shuffling(G_sample)
        mu_r1, sigma_r1 = fid.calculate_statistics(real_samples[: l])
        mu_r2, sigma_r2 = fid.calculate_statistics(real_samples[l :])
        shuffling(real_samples)
        # print('The %d times samples:' % i)
        # print(real_samples)
        n_fid[0].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_f, sigma_f))
        n_fid[1].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_r2, sigma_r2))
    # print(n_fid[1])
    return np.mean(n_fid, axis=1)

D_loss_fake = tf.reduce_mean(D_fake)
D_loss_real = tf.reduce_mean(D_real)
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)
D_acc_1, D_acc, l = acc(D_fake)
D_real_acc, _, _ = acc(D_real)
# fid = cal_fid(X, G_sample)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-5)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-5)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# summary
tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', -D_loss)
# tf.summary.scalar('D_fake_acc', D_acc)
# tf.summary.scalar('D_real_acc', D_real_acc)

# tf.summary.scalar('fid', fid)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

sess = tf.Session()
if os.path.exists(os.path.join(FLAGS.model_path + '.index')):
    saver.restore(sess, FLAGS.model_path)
    print('restore model...')
else:
    sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(FLAGS.model_path, sess.graph)

"""
if FLAGS.is_train:
    sess.run(tf.global_variables_initializer())
else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
"""

def main(_):
    if FLAGS.is_train:
        fid = []
        d_a = []
        loss = []
        for it in range(FLAGS.iter):
            # print('spec:', spectral_data)
            # print('x_mb', X_mb)
            for i in range(10):
                data = sio.loadmat(FLAGS.train_dir)
                spectral_data = data['data']
                X_mb, _ = next_batch(FLAGS.batch_size, it + i, spectral_data, spectral_labels)
                z_sample = sample_z(X_mb.shape[0], z_dim)
                # print('Z_X_mb', X_mb)
                # print('Z_SPEC', spectral_data)
                _, D_loss_curr, _, D_l_f, D_l_r = sess.run(
                    [D_solver, D_loss, clip_D, D_loss_fake, D_loss_real],
                    feed_dict={X: X_mb, z: z_sample}
                )
                # print('D_x_mb', X_mb)
                # print('D_spec', spectral_data)
            D_real_acc_curr_1, D_fake_acc_curr_1, last_value, grad_real_curr, grad_fake_curr = sess.run([D_real_acc, D_acc, l, grad_real, grad_fake],
                                                                                                        feed_dict={X: X_mb, z: z_sample})
            # print('acc_x_mb', X_mb)
            # print('acc_spec', spectral_data)
            _, G_loss_curr, g_sample = sess.run(
                [G_solver, G_loss, G_sample],
                feed_dict={z: z_sample}
            )
            z_sample_1 = sample_z(X_mb.shape[0], z_dim)
            D_real_acc_curr_2, D_fake_acc_curr_2 = sess.run([D_real_acc, D_acc_1], feed_dict={X: X_mb, z: z_sample_1})
            # print('g_acc_x_mb', X_mb)
            # print('g_acc_spec', spectral_data)
            # print('**********************************************************************')
            if it == 0:
                saver.save(sess, FLAGS.model_path, global_step=it)
            if it % 100 == 0:
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                      .format(it, -D_loss_curr, G_loss_curr))
                # print('shuffle', X_mb)
                X_for_fid = copy.copy(X_mb)
                fid.append(cal_fid(X_for_fid, g_sample))
                # print('shuffle down', X_mb)
                d_a.append([D_real_acc_curr_1, D_fake_acc_curr_1, D_real_acc_curr_2, D_fake_acc_curr_2])
                loss.append([-D_loss_curr, G_loss_curr, D_l_f, D_l_r])
                print('FID:', cal_fid(X_for_fid, g_sample))
                print('D_acc:', D_real_acc_curr_1, D_fake_acc_curr_1)
                print('G_acc:', D_real_acc_curr_2, D_fake_acc_curr_2)
                if it > 0 and D_fake_acc_curr_1 == 0:
                    print(last_value)
                    saver.save(sess, FLAGS.model_path, global_step=it)
                    print('Acc becomes 0, training has no meaning...')
                    sio.savemat(FLAGS.model_path + '/data' + str(FLAGS.class_number) + 'break.mat',
                                {'fid': fid, 'd_acc': d_a, 'z_sample': z_sample, 'real': X_mb})
                    break

                    # print(X_mb)
            saver.save(sess, FLAGS.model_path)
            summary_str = sess.run(summary_op, feed_dict={X: X_mb, z: z_sample})
            summary_writer.add_summary(summary_str, it)
            summary_writer.flush()

            if it % 1000 == 0:
                samples = sess.run(G_sample, feed_dict={z: sample_z(100, z_dim)})
                # print(samples)
                sio.savemat(FLAGS.model_path + '/data' + str(it) + '.mat', {'g_sample': samples, 'grad_real': grad_real_curr, 'grad_fake': grad_fake_curr})

        sio.savemat(FLAGS.model_path + '/data' + str(FLAGS.class_number) + '.mat',
                    {'fid': fid, 'd_acc': d_a, 'loss': loss})
    else:
        pass

    sess.close()

if __name__ == '__main__':
    tf.app.run()
