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
		prep, this_prob = cal_perplexity(dist[i], i, beta[i])

		# 二分搜索，寻找最佳sigma下的prob
		perp_diff = perp - base_perp
		tries = 0 
		while np.abs(perp_diff) > tol and tries < 50:
			if perp_diff > 0:
				betamin = beta[i].copy()
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2         #这里是不是应该是/
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

def pca(x, no_dim = 50):
	"""
	使用PCA算法进行欲降维
	"""
	print('Proprecessing the data using PCA...')
	(n, d) = x.shape
	x = x - np.tile(np.mean(x, 0), (n, 1)) # Construct an array by repeating A the number of times given by reps
	1, M = np.linalg.eig(np.dot(x.T, x)) # 计算矩阵的特征向量