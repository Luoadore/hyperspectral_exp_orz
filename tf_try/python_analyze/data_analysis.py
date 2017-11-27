# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import read_data as rd
#from sklearn import preprocessing as sp

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

def plot_misclassify_mean(data_mean, class_mis, data_mis, data_name):
    """
    Plotting the misclassfication mean line, which includes real and incorrect class's mean and its value.

    Args:
        data_mean: All the classes' mean array, from get_statistic_by_class()
        class_mis: Samples's real class and incorrect label.
        data_mis: Misclassified samples' bands value.
        data_name: Data name.

    """
    if np.size(class_mis) == 0:
        print('No misclassfication.')
    else:
        mis_num = np.size(data_mis, 0)
        fea_num = np.size(data_mis, 1)
        xidx = range(0, fea_num)

        for each, i in zip(class_mis, range(mis_num)):
            real_label = each[0]
            pred_label = each[1]
            data = data_mis[i]
            data_mean_r = data_mean[real_label, :]
            data_mean_m = data_mean[pred_label, :]
            plt.plot(xidx, data, label = 'class-' + str(real_label) + ' pred to ' + str(pred_label))
            plt.plot(xidx, data_mean_r, label = 'class-' + str(real_label))
            plt.plot(xidx, data_mean_m, label = 'class-' + str(pred_label))

            plt.xlabel('Feature No.')
            plt.ylabel('Mean feature value')
            plt.title('Feature value distribution (' + data_name + ')')
            plt.legend()
            #plt.show()
            plt.savefig('I:\\try\\result_data\\' + data_name + str(i))

            plt.clf()
            plt.cla()


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
    #root_split = 'F:\hsi_result\original\KSC\data'
    root_split = 'I:\\try\\result_data\KSCdata'
    root_original = 'D:\hsi\dataset\Kennedy Space Center (KSC)'

    data_o = sio.loadmat(root_original + '\KSCData.mat')
    data = data_o['DataSet']
    label_o = sio.loadmat(root_original + '\KSCGt.mat')
    label = label_o['ClsID']
    #scaler = sp.StandardScaler().fit(data)
    #data = scaler.transform(data)

    train_data, train_label, train_pred, train_pos, test_data, test_label, test_pred, test_pos = rd.get_data(root_split + '\data1.mat')
    train_mean, train_std = get_statistic_by_class(train_data, train_label)
    test_mean, test_std = get_statistic_by_class(test_data, test_label)
    # plot_mean_line(train_mean, 'Data1')
    train_conf_matrix = rd.get_confuse_matrix(train_label, train_pred)
    test_conf_matrix = rd.get_confuse_matrix(test_label, test_pred)
    print('train conf matrix')
    print(train_conf_matrix)
    print('test conf matrix')
    print(test_conf_matrix)
    tr_mis_class, tr_mis_position, tr_mis_bands, tr_mis_neighbors = rd.get_misClassify_neighbors_info(train_conf_matrix, label, train_data, train_label, train_pred, train_pos)
    te_mis_class, te_mis_position, te_mis_bands, te_mis_neighbors = rd.get_misClassify_neighbors_info(test_conf_matrix, label,test_data, test_label, test_pred, test_pos)

    print(np.size(tr_mis_class))

    plot_misclassify_mean(train_mean, tr_mis_class, tr_mis_bands, 'KSC_train')
    plot_misclassify_mean(test_mean, te_mis_class, te_mis_bands, 'KSC_test')

    sio.savemat(root_split + '\misData_info.mat', {'tr_mis_class': tr_mis_class, 'tr_mis_position': tr_mis_position, 'tr_mis_bands': tr_mis_bands, 'tr_mis_neighbors': tr_mis_neighbors, 'train_conf_matrix': train_conf_matrix, 
                                                   'te_mis_class': te_mis_class, 'te_mis_position': te_mis_position, 'te_mis_bands': te_mis_bands, 'te_mis_neighbors': te_mis_neighbors, 'test_conf_matrix': test_conf_matrix})
    print('Done.') 