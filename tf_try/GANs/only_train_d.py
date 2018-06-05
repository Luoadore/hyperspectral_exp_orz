import tensorflow as tf
import scipy.io as sio
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = sio.loadmat('/media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/model/exp_11/data11break.mat')
spectral_data = data['real']
z_sample = data['z_sample']

X_dim = spectral_data.shape[1]
z_dim = 10
h_dim = 64

def shuffling(data):
    return data

def next_batch(batch_size, num_step, data_set):
    data_size = len(data_set)
    num_per_epoch = math.ceil(data_size / batch_size)
    remainder = num_step % num_per_epoch
    start_index = remainder * batch_size
    end_index = min(start_index + batch_size, data_size)
    if end_index - start_index < 10:
        shuffling(data_set)
        start_index, end_index = 0, batch_size
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
    return G_log_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out

G_sample = generator(z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

def acc(D_fake):
    accuracy_real = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) > 0.5, tf.float32))
    accuracy_fake = tf.reduce_mean(tf.cast(tf.nn.sigmoid(D_fake) <= 0.5, tf.float32))
    return accuracy_real / 2, accuracy_fake / 2, tf.nn.sigmoid(D_fake)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
D_acc_1, D_acc, l = acc(D_fake)
D_real_acc, _, _ = acc(D_real)
# D_solver = (tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(-D_loss, var_list = theta_D))
D_solver = (tf.train.RMSPropOptimizer(learning_rate=0.0001)
            .minimize(-D_loss, var_list=theta_D))
clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, '/media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/model/exp_11-0')
print('where is restore model???')

d_a = []
d_loss = []
for it in range(2000):
    X_mb = next_batch(50, it, spectral_data)
    _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: X_mb, z: z_sample})
    D_real_acc_curr_1, D_fake_acc_curr_1, last_value = sess.run([D_real_acc, D_acc, l], feed_dict={X: X_mb, z: z_sample})
    if it % 10 == 0:
        print('Iter: {}; D loss: {:.4}'.format(it, - D_loss_curr))
        print('D_acc:', D_real_acc_curr_1, D_fake_acc_curr_1)
        d_a.append([D_real_acc_curr_1, D_fake_acc_curr_1])
        d_loss.append(D_loss_curr)
print(last_value)

sio.savemat('./model/data_d.mat', {'d_a': d_a, 'd_l': d_loss})

sess.close()