# coding: utf-8
"""Visualizing Large-scale and High-dimensional data using t-SNE."""

#plot
"""import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as pyplot
plt.style.use('ggplot')

def sne_crowding():
    npoints = 1000 #抽去1000个m维球内均匀分布的点
    plt.figure(figsize = (20, 5))
    for i, m in enumerate((2, 3, 5, 8)):
        #这里模拟m维球中的均匀分布用到了拒绝采样
        #即先生成m维立方中的均匀分布，再剔除m维球外部的点
        accepts = []
        while len(accepts) < 1000:
            points = np.random.rand(500, m)
            accepts.extend([d for d in norm(points, axis = 1) if d <= 1.0]) #拒绝采样

        accepts = accepts[:npoints]"""

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
 
def cal_pairwise_dist(x):
    """
    计算pairwise距离，(a - b)^2 = a^w + b^2 - 2*a*b
        x: matrix
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist

def cal_perplexity(dist, idx = 0, beta = 1.0):
    """
    计算perplexity，这里的perp仅计算熵，方便计算
        D：距离向量
        idx： dist中自己与自己距离的位置
        beta： 高斯分布参数
    """
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob) + beta + np.sum(dist * prob) / sum_prob
    prob /= sum_prob
    return perp, prob

def search_prob(x, tol = 1e-5, perplexity = 30.0):
    """
    二分搜索寻找beta， 并计算pariwise的prob
    """

    #初始化参数
    print('Computing pairwise distances...')
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    #取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print('Computing pair_prob for point %s of %s ...' %(i, n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索，寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0 
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] / 2         #这里是本来是*，但我觉得是/
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                 betamax = beta[i].copy()
                 if betamin == np.inf or betamin == -np.inf:
                     beta[i] = beta[i] / 2
                 else:
                     beta[i] = (beta[i] + betamin) / 2

            #更新perb，prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1

        #记录prob值
        pair_prob[i,] = this_prob
    print('Mean value of sigma: ', np.mean(np.sqrt(1 / beta)))
    return pair_prob

def pca(x, no_dims = 50):
    """
    使用PCA算法进行欲降维
    """
    print('Proprecessing the data using PCA...')
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1)) # Construct an array by repeating A the number of times given by reps
    v, M = np.linalg.eig(np.dot(x.T, x)) # eig计算矩阵的特征向量
    y = np.dot(x, M[:, 0:no_dims])
    return y

def tsne(x, no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
    Run t-SNE on the dataset in the N * D array x to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity), where x is an N*D numpy array.
    """
    #check inputs
    if isinstance(no_dims, float):
        print('Error: array x should have type float.')
        return -1
    if round(no_dims) != no_dims:
        print('Error: number of dimensions should be an integer.')
        return -1

    # initial parameters and variables
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = search_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration(trick)
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum(P / 4 * np.log(P / 4 / Q))
            print('Iteration', (iter + 1), ': error is ', C)
        #Stop lying about P-values
        if iter == 100:
            P = P / 4
    print('Finished training!')
    return y

if __name__ == '__main__':
    # Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on dataset
    data = sio.loadmat('I:\\try\\result_data\KSCdata\data1.mat')
    X = data['train_data']
    labels = data['train_label']
    Y = tsne(X, 2, 50, 20.0)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.show()