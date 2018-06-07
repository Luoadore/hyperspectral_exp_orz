# coding: utf-8
from sklearn import preprocessing
import scipy.io as sio
import numpy as np

def scaling(data):
    """
    Min_max_scaling.
    :param data:Original data.
    :return: Scaling data.
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)
    return data_minmax

def remove(data):
    """
    Remove noise replaced by 0.
    """
    for eachdata in data:
        eachdata[eachdata >= 10] = 0
    return data

def delete(data):
    """
    Delete samples which has noise value.
    """
    index = []
    for i, each in enumerate(data):
        if True in (each >= 10):
            index.append(i)
            print('too big')
    print(index)
    data = np.delete(data, index, axis=0)
    return data

"""
data = sio.loadmat('/media/luo/result/hsi_gan_result/KSC/hsi_data11.mat')
spectral_data = data['data']
print(type(spectral_data))
print(np.max(spectral_data))
# hsi_data = remove(spectral_data)
hsi_data = delete(spectral_data)
print(np.max(hsi_data))
print(hsi_data.shape)
data_scaling = scaling(hsi_data)
print(np.max(data_scaling))
print(data_scaling)
"""