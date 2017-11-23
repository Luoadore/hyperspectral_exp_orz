# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import read_data as rd

def get_statistic_by_class(fea, label):
    """
    Calculate each class's mean and std.

    Args:
        fea: Input feature.
        label: Label.

    Returns:
        fea_mean: Feature's mean.
        fea_std: Feature's standard deviation.

    """
    class_num = np.max(label) + 1
    fea_num = np.size(fea, 1)

    data_list = []
    for mark in range(class_num):
        data_list.append([])

    for idx in range(label.size):
        lab_idx = label[0, idx]
        fdata = fea[idx]
        data_list[lab_idx].append(fdata)

    fea_mean = np.zeros([class_num, fea_num])
    fea_std = np.zeros([class_num, fea_num])
    for idx in range(class_num):
        fdata = np.array(data_list[idx])
        fea_mean[idx, :] = np.mean(fdata, 0)
        fea_std[idx, :] = np.std(fdata, 0)

    return fea_mean, fea_std

def plot_mean_line(idata, data_name):
    """
    Plotting.

    Args:
        idata: Plotting data.
        data_name: Plotting figure name.

    """
    class_num = np.size(idata, 0)
    fea_num = np.size(idata,1)
    xidx = range(0, fea_num)

    for idx in range(class_num):
        data = idata[idx, :]
        plt.plot(xidx, data, label = 'class-' + str(idx))

    plt.xlabel('Feature No.')
    plt.ylabel('Mean feature value')
    plt.title('Feature value distribution (' + data_name + ')')
    plt.legend()
    plt.show()

def plot_std_line(idata, data_name):
    """
    Plotting.

    Args:
        idata: Plotting data.
        data_name: Plotting figure name.

    """
    class_num = np.size(idata, 0)
    fea_num = np.size(idata,1)
    xidx = range(0, fea_num)

    for idx in range(class_num):
        data = idata[idx, :]
        plt.plot(xidx, data, label = 'class-' + str(idx))

    plt.xlabel('Feature No.')
    plt.ylabel('Std of feature value')
    plt.title('Feature value distribution (' + data_name + ')')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    root = 'F:\hsi_result\original\KSC\data'
    train_data, train_label, train_pred, test_data, test_label, test_pred = rd.get_data(root + '\data1.mat')
    train_mean, train_std = get_statistic_by_class(train_data, train_label)
    plot_mean_line(train_mean, 'Data1')