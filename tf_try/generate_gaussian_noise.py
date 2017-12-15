# coding: utf-8
"""
Due to the lack of training samples, there to generate Gaussian noise and extend the data set.
"""

import data_preprocess_pos as dp
import python_analyze.data_analysis as da
import python_analyze.read_data as rd
import numpy as np
import math

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
    data_set, label_set, _, position = rd.get_data(file)
    classes = np.max(label_set) + 1
    data_num = np.zeros(classes, 1)
    for i in label_set:
        data_num[i] += 1
    data_mean, data_std = da.get_statistic_by_class(data_set, label_set)
    noise_by_class = []
    for eachclass in range(classes):
        if eachclass == 7 or eachclass == 12:
            pass
        else:
            spectral_idx = spectral[eachclass]
            noise = np.random.normal(data_mean[eachclass][spectral_idx], data_std[eachclass][spectral_idx], data_num[classes])
            noise_by_class.append(noise)
    print('Generate gaussian noise done.')

    data = data_set
    label = label_set
    count = np.zeros(classes, 1)
    for eachdata, eachlabel in zip(data_set, label_set):
        eachdata_noise = eachdata + noise_by_class[eachlabel][count[eachlabel]]
        data.append(eachdata_noise)
        count[eachlabel] += 1

    label.extend(label_set)
    print('Extend data set done.')

    data, label, position = dp.shuffling2(data, label, position)

    return data, label, position

def e_step(data, pi_k, miu, sigma):
    """
    The E step of EM, to calculate posterior probability.

    Args:
        data: Entire data which needs fitting to GMM.
        pi_k: The prior probability of two components.
        miu: The means of two components.
        sigma: The std of two components.

    Return:
        posterior_prob: Size of num * component.
    """


def m_step(data, pi_k, miu, sigma):
    """
    The M step of EM, to update three parameters include pi, mean, std using maximum likehood function.
    Args:
        data: Entire data which needs fitting to GMM.
        pi_k: The prior probability of two components.
        miu: The means of two components.
        sigma: The std of two components.

    Return:
        pi_k_new, miu_new, sigma_new: The updated parameters.
    """

def gmm(data, iter):
    """
    算法迭代iter次，或者三个参数均不再变化时，拟合混合高斯模型, two component.

    Args:
        data: Entire data which needs fitting to GMM.
        iter: Iterations for fitting.

    Return:
        data_generated: The original `1data which 

    """
    data_num = len(data)
    miu = np.random.random(2)
    sigma = np.random.random(2)
    pi_k = [0.5, 0.5]
    expectations = np.zeros((data_num, 2))
    print('The initial mean: ' + str(miu))
    print('The initial std: ' + str(sigma))
    print('The initial pi: ' + str(pi_k))

    for i in range(iter):
        posterior_prob = e_step(data, pi_k, miu, sigma)
        pi_k_new, miu_new, sigma_new = m_step(data, pi_k, miu, sigma)

        if pi_k_new != pi_k or miu_new != miu or sigma_new != sigma:
            pi_k = pi_k_new
            miu = miu_new
            sigma = sigma_new
        else:
            break

        if i % 100 == 0:
            print('The present params(mean, std, pi): ' + str(miu) + str(sigma) + str(pi_k))

        print('The final params: mean' + str(miu) + ', std ' + str(sigma) + ', pi ' + str(pi_k))

