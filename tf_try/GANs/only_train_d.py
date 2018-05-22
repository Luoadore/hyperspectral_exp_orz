import tensorflow as tf
import scipy.io as sio
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = sio.loadmat('./train/data0.mat')
spectral_data = data['X']
z_sample = data['z']

X_dim = spectral_data.shape[1]
z_dim = 100
h_dim = 128

def next_batch(batch_size, num_step, data_set):
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch
    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    batch_data = data_set[start_index : end_index]

    return batch_data

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

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
    accuracy_real = tf.reduce_mean(tf.cast(D_fake > 0.5, tf.float32))
    accuracy_fake = tf.reduce_mean(tf.cast(D_fake < 0.5, tf.float32))
    return accuracy_real / 2, accuracy_fake / 2

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
D_acc_1, D_acc = acc(D_fake)
D_real_acc, _ = acc(D_real)
D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./model')
model_file = tf.train.latest_checkpoint('./model')
saver.restore(sess, model_file)

d_a = []
for it in range(1000):
    X_mb = next_batch(25, it, spectral_data)
    _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb, z: z_sample})
    D_real_acc_curr_1, D_fake_acc_curr_1 = sess.run([D_real_acc, D_acc], feed_dict={X: X_mb, z: z_sample})
    print('Iter: {}; D loss: {:.4}'.format(it, - D_loss_curr))
    print('D_acc:', D_real_acc_curr_1, D_fake_acc_curr_1)
    d_a.append([D_real_acc_curr_1, D_fake_acc_curr_1])

sio.savemat('./train/data_d.mat', {'d_a': d_a})

sess.close()