# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.io as sio
import sys
sys.path.append('..')
import GANs.data_scaling as ds

'''
prepare for datasets.
'''
data = sio.loadmat('/media/luo/result/hsi_gan_result/KSC/hsi_data11.mat')
spectral_data = data['data']
spectral_data = ds.delete(spectral_data)
spectral_data = ds.scaling(spectral_data)

def next_batch(batch_size, data_set, batch_num):
    start_index = i * batch_size
    end_index = min(start_index + batch_size, len(data_set))
    batch_data = data_set[start_index: end_index]
    return batch_data


'''
hyper parameters.
'''
n_input = spectral_data.shape[1]
n_hidden_1 = 256
n_hidden_2 = 2
learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_epoch = 3
show_num = 10
x = tf.placeholder(dtype=tf.float32,shape=[None,n_input])
# 输入分布数据，用来生成模拟数据样本
zinput = tf.placeholder(dtype=tf.float32,shape=[None,n_hidden_2])

'''
net parameters.
'''
weights = {
 'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1],stddev = 0.001)),
 'mean_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev = 0.001)),
 'log_sigma_w1':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2],stddev = 0.001)),
 'w2':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_1],stddev = 0.001)),
 'w3':tf.Variable(tf.truncated_normal([n_hidden_1,n_input],stddev = 0.001))
 }
biases = {
 'b1':tf.Variable(tf.zeros([n_hidden_1])),
 'mean_b1':tf.Variable(tf.zeros([n_hidden_2])),
 'log_sigma_b1':tf.Variable(tf.zeros([n_hidden_2])),
 'b2':tf.Variable(tf.zeros([n_hidden_1])),
 'b3':tf.Variable(tf.zeros([n_input]))
 }

'''
net structure.
'''
# two encoders

# first layer input_dim->256
h1 = tf.nn.relu(tf.add(tf.matmul(x,weights['w1']),biases['b1']))
# second layer has two outputs
z_mean = tf.add(tf.matmul(h1,weights['mean_w1']),biases['mean_b1'])
z_log_sigma_sq = tf.add(tf.matmul(h1,weights['log_sigma_w1']),biases['log_sigma_b1'])

# gaussian
eps = tf.random_normal(tf.stack([tf.shape(h1)[0],n_hidden_2]),0,1,dtype=tf.float32)
z = tf.add(z_mean,tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)),eps))

# decoder
# first layer 2->256
h2 = tf.nn.relu(tf.matmul(z,weights['w2']) + biases['b2'])
# second layer 256->input_dim reconstruction
reconstruction = tf.matmul(h2,weights['w3']) + biases['b3']

'''
For generating data.
'''
h2out = tf.nn.relu(tf.matmul(zinput,weights['w2']) + biases['b2'])
reconstructionout = tf.matmul(h2out,weights['w3']) + biases['b3']

'''
loss
'''
# MSE TODO: add cross entropy loss
reconstr_loss = 0.5*tf.reduce_sum((reconstruction-x)**2)
print(reconstr_loss.shape)
# KL
latent_loss = -0.5*tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq),1)
print(latent_loss.shape) #(128,)
cost = tf.reduce_mean(reconstr_loss+latent_loss)
# optim
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
num_batch = int(np.ceil(len(spectral_data) / batch_size))

'''
train
'''
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
 
    print('start train')
    for epoch in range(training_epochs):
        total_cost = 0.0
        for i in range(num_batch):
            batch_x = next_batch(batch_size, spectral_data, i)
            _,loss = sess.run([optimizer,cost],feed_dict={x:batch_x})
            total_cost += loss

        if epoch % display_epoch == 0:
            print('Epoch {}/{} average cost {:.9f}'.format(epoch+1,training_epochs,total_cost/num_batch))
 
    print('train down')
 
    # test
    # print('Result:',cost.eval({x:mnist.test.images}))


 


'''
 labels = [np.argmax(y) for y in mnist.test.labels] 
 mean,log_sigma = sess.run([z_mean,z_log_sigma_sq],feed_dict={x:mnist.test.images})
 plt.scatter(mean[:,0],mean[:,1],c=labels)
 plt.colorbar()
 plt.show()

 plt.figure(figsize=(5,4))
 plt.scatter(log_sigma[:,0],log_sigma[:,1],c=labels)
 plt.colorbar()
 plt.show()
'''
 
'''
高斯分布取样，生成模拟数据

 n = 15
 digit_size = 28
 figure = np.zeros((digit_size * n, digit_size * n))
 grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
 grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) 
 for i, yi in enumerate(grid_x):
 for j, xi in enumerate(grid_y):
 z_sample = np.array([[xi, yi]])
 x_decoded = sess.run(reconstructionout,feed_dict={zinput:z_sample})
 
 digit = x_decoded[0].reshape(digit_size, digit_size)
 figure[i * digit_size: (i + 1) * digit_size,
 j * digit_size: (j + 1) * digit_size] = digit
'''
