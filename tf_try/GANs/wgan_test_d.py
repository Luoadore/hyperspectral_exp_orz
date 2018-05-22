import tensorflow as tf

import numpy as np
import os
import scipy.io as sio
import math
import fid
from random import shuffle
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# param config
flags = tf.app.flags
flags.DEFINE_integer('iter', 1000, 'Iteration to train.')
flags.DEFINE_integer('batch_size', 100, 'The size of each batch.')
flags.DEFINE_string('model_path', './model/wgan.model', 'Save model path.')
flags.DEFINE_boolean('is_train', True, 'Train or test.')
flags.DEFINE_integer('class_number', 0, 'The class that want to generate, if None, generate randomly.')
flags.DEFINE_string('train_dir', '/media/luo/result/hsi_gan_result/KSC/hsi_data0.mat', 'Train data path.')
FLAGS = flags.FLAGS

# load data
data = sio.loadmat(FLAGS.train_dir)
# spectral_data = data['data']
spectral_data = copy.copy(data['data'])
spectral_labels = data['label']
print('data:', data['data'])

X_dim = spectral_data.shape[1]
z_dim = 100
h_dim = 128

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
    print('hsi_data:', data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch

    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]
    batch_label = label_set[start_index : end_index]
    # print('start-end:', start_index, end_index)
    # print(batch_data)
    return batch_data, batch_label

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

def acc(D_fake):
    # accuracy_real = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) > 0.5, tf.float32))
    # accuracy_fake = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) < 0.5, tf.float32))
    accuracy_real = tf.reduce_mean(tf.cast(D_fake > 0.5, tf.float32))
    accuracy_fake = tf.reduce_mean(tf.cast(D_fake < 0.5, tf.float32))
    return accuracy_real / 2, accuracy_fake / 2

def is_duplicate(X):
    for i in range(len(X) - 1):
        index = (X[i] == X[i + 1])
        if False in index:
            continue
        else:
            return True
    return False

def cal_fid(real_samples, G_sample):
    """
    Args:
        real_samples: Samples of per batch.
        G_sample: Tensorflow operation.
    Return:
        : [D_G fid, D_D fid]
    """
    n_fid = [[], []]
    for i in range(5):
        l = len(real_samples) // 2
        mu_f, sigma_f = fid.calculate_statistics(G_sample[: l])
        shuffle(G_sample)
        mu_r1, sigma_r1 = fid.calculate_statistics(real_samples[: l])
        mu_r2, sigma_r2 = fid.calculate_statistics(real_samples[l :])
        shuffle(real_samples)
        n_fid[0].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_f, sigma_f))
        n_fid[1].append(fid.calculate_frechet_distance(mu_r1, sigma_r1, mu_r2, sigma_r2))
    # print(n_fid[1])
    return np.mean(n_fid, axis=1)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)
D_acc_1, D_acc = acc(D_fake)
D_real_acc, _ = acc(D_real)
# fid = cal_fid(X, G_sample)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

# summary
tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('D_loss', -D_loss)
tf.summary.scalar('D_fake_acc', D_acc)
tf.summary.scalar('D_real_acc', D_real_acc)
# tf.summary.scalar('fid', fid)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(FLAGS.model_path, sess.graph)

if FLAGS.is_train:
    sess.run(tf.global_variables_initializer())
else:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)

def main(_):
    if FLAGS.is_train:
        fid = []
        d_a = []
        for it in range(FLAGS.iter):
             #data = sio.loadmat(FLAGS.train_dir)
             # spectral_data = data['data']
            # global spectral_data
            X_mb, _ = next_batch(FLAGS.batch_size, it, spectral_data, spectral_labels)
            print(spectral_data)
            if is_duplicate(X_mb):
                print('iters, :', it)
                print(X_mb)
            for i in range(5):

                z_sample = sample_z(X_mb.shape[0], z_dim)
                _, D_loss_curr, _ = sess.run(
                    [D_solver, D_loss, clip_D],
                    feed_dict={X: X_mb, z: z_sample}
                )


            D_real_acc_curr_1, D_fake_acc_curr_1 = sess.run([D_real_acc, D_acc], feed_dict={X: X_mb, z: z_sample})

            if it > 100 and D_fake_acc_curr_1 < 0.1:
                print(X_mb)
                break
            _, G_loss_curr, g_sample = sess.run(
                [G_solver, G_loss, G_sample],
                feed_dict={z: z_sample}
            )
            z_sample_1 = sample_z(X_mb.shape[0], z_dim)
            D_real_acc_curr_2, D_fake_acc_curr_2 = sess.run([D_real_acc, D_acc_1], feed_dict={X: X_mb, z: z_sample_1})

            if it % 10 == 0:
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                      .format(it, - D_loss_curr, G_loss_curr))
                fid.append(cal_fid(X_mb, g_sample))
                d_a.append([D_real_acc_curr_1, D_fake_acc_curr_1, D_real_acc_curr_2, D_fake_acc_curr_2])
                print('FID:', cal_fid(X_mb, g_sample))
                print('D_acc:', D_real_acc_curr_1, D_fake_acc_curr_1)
                print('G_acc:', D_real_acc_curr_2, D_fake_acc_curr_2)
                saver.save(sess, FLAGS.model_path)
                summary_str = sess.run(summary_op, feed_dict={X: X_mb, z: z_sample})
                summary_writer.add_summary(summary_str, it)
                summary_writer.flush()
                samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

        #if break, only train D
        if it > 100 and D_fake_acc_curr_1 < 0.1:
            for i in range(5):
                _, D_loss_curr, _, g_sample = sess.run(
                    [D_solver, D_loss, clip_D, G_sample],
                    feed_dict={X: X_mb, z: z_sample}
                )

                D_real_acc_curr_1, D_fake_acc_curr_1 = sess.run([D_real_acc, D_acc], feed_dict={X: X_mb, z: z_sample})

                if i % 1 == 0:
                    print('Iter: {}; D loss: {:.4}'.format(i, -D_loss_curr))
                    print('D_acc:', D_real_acc_curr_1, D_fake_acc_curr_1)
                    d_a.append([D_real_acc_curr_1, D_fake_acc_curr_1, 0, 0])


        sio.savemat('./train/data' + str(FLAGS.class_number) + '.mat', {'fid': fid, 'd_acc': d_a, 'X': X_mb, 'z': z_sample, 'g_sample': g_sample})
    else:
        pass

    sess.close()

if __name__ == '__main__':
    tf.app.run()
