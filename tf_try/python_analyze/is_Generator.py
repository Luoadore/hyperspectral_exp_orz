# coding: utf-8
"""
This .py tries to estimate whether the GAN model is a fine generator after training.
"""
import numpy as np
import scipy.io as sio

def get_statistic(path):
    data = sio.loadmat(path)
    hsi_data = data['data']
    fea_mean = np.mean(hsi_data, axis = 0)
    fea_std = np.std(hsi_data, axis = 0)
    return fea_mean, fea_std

def mse_like(path, fea_mean):
    data = sio.loadmat(path)
    samples_data = data['data']
    num, dimen = samples_data.shape
    conf = np.zeros(num)
    for i in range(num):
        for j in range(dimen):
            conf[i] = conf[i] + np.square(fea_mean[j] - samples_data[i][j])
    return conf / dimen

if __name__ == '__main__':
    data_path = 'D:\hsi_gan_result\KSC\hsi_data0.mat'
    sample_path = 'F:\codestore\hyperspectral_exp_orz\\tf_try\GANs\\test\data0.mat'
    fea_mean, _ = get_statistic(data_path)

    conf = mse_like(sample_path, fea_mean)
    print(sum(conf) / len(conf))