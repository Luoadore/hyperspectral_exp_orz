# coding: utf-8
"""
Due to the lack of training samples, there to generate Gaussian noise and extend the data set.
"""

import data_preprocess_pos as dp
import data_analysis as da
import read_data as rd
import numpy as np
import scipy.io as sio

def generate_gaussian_dataset(file, spectral):
    """
    Extend data set.

    Args:
       file: The file dir to get data_set, data needs to extend, one spectral add gaussian noise generates, and label_set, Corresponding label.
       spectral: A list, its index means the categories and the corresponding value is the spectral which needs add noise.

    Return:
        data: Extended data set.
        label: Extended label set.
    """
    data_set, label_set = rd.get_data(file)
    classes = np.max(label_set) + 1
    data_num = np.zeros(classes, 1)
    for i in label_set:
        data_num[i] += 1
    data_mean, data_std = da.get_statistic_by_class(data_set, label_set)
    noise_by_class = []
    for eachclass in range(classes):
        spectral_idx = spectral[eachclass]
        noise = np.random.normal(data_mean[eachclass][spectral_idx], data_std[eachclass][spectral_idx], data_num[classes])
        noise_by_class.append(noise)

    data = data_set
    label = label_set
    for eachdata, eachlabel in zip(data_set, label_set):
