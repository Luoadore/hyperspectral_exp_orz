"""
Normalize source and target with all dataset samples.
"""

import scipy.io as sio
import numpy as np
from random import shuffle, randint
import math
from sklearn import preprocessing as sp
from config import *
from select_domaindata import load_data

def mean_std(data_file, labels_file, neighbor):
    """
    N-neighbor and data select from rectangle area.(N=0, 4, 8)

    :param center_pos: [x1, y1]
    :param rectangle_size: area size, [length, width]
    :param data_file:
    :param labels_file:
    :param neighbor:
    :return:
    data
    pos
    """

    global data_position
    data = sio.loadmat(data_file)
    label = sio.loadmat(labels_file)
    data_o = data['DataSet']
    labels = label['ClsID']
    classes = np.max(labels)
    print('there are ' + str(classes) + ' class in the data set.')
    print(labels.shape)
    rows, cols = labels.shape
    bands = np.size(data_o, 2)
    print('The data has ' + str(bands) + ' bands.')

    data_set = []
    data_list = []
    for mark in range(classes):
        data_list.append([])

    data_pos = []
    for mark in range(classes):
        data_pos.append([])

    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label != 0:
                data_temp = data_o[i, j]
                data_position = [i, j]
                if neighbor > 0:
                    center_data = data_temp
                    ##################################
                    # The neighbors-structer:
                    #  data1      data2      data3
                    #  data4   center_data   data5
                    #  data6      data7      data8
                    ##################################

                    # judgment standard, how to choose neighbors' coordinates
                    lessThan = lambda x: x - 1 if x > 0 else 0
                    greaterThan_row = lambda x: x + 1 if x < rows - 1 else rows - 1
                    greaterThan_col = lambda x: x + 1 if x < cols - 1 else cols - 1


                    if neighbor == 8:
                        data_temp = []
                        for each in range(bands):
                            data1 = data_o[lessThan(i), lessThan(j)][each]
                            data2 = data_o[lessThan(i), j][each]
                            data3 = data_o[lessThan(i), greaterThan_col(j)][each]
                            data4 = data_o[i, lessThan(j)][each]
                            data5 = data_o[i, greaterThan_col(j)][each]
                            data6 = data_o[greaterThan_row(i), lessThan(j)][each]
                            data7 = data_o[greaterThan_row(i), j][each]
                            data8 = data_o[greaterThan_row(i), greaterThan_col(j)][each]
                            data_1 = np.append(data1, data2)
                            data_2 = np.append(data3, data4)
                            data_3 = np.append(data_1, data_2)
                            data_4 = np.append(data_3, center_data[each])
                            data_5 = np.append(data5, data6)
                            data_6 = np.append(data7, data8)
                            data_7 = np.append(data_4, data_5)
                            data_t = np.append(data_7, data_6)
                            data_temp = np.append(data_temp, data_t)
                        # print('8 neighbors extract done.')

                data_temp[data_temp >= 65534] = 0
                data_set.append(data_temp)
                data_list[label - 1].append(data_temp)
                data_pos[label - 1].append(data_position)


    scaler = sp.StandardScaler().fit(data_set)
    print('Extract data done.')
    return scaler, data_list, data_pos

def normalize_dataset(scaler, train_data, test_data):
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    print('Normalized.')
    return train_data, test_data

if __name__ == '__main__':

    # prepare
    data = [ip, pu, sa]
    # dataset = ksc
    neighbor = 8

    for dataset in data:
        print('dataset: ' + str(dataset))
        scaler, data_list, data_pos = mean_std(dataset['data_dir'], dataset['label_dir'], neighbor)


        # source and target normalize
        """
        data_set = sio.loadmat(dataset['train_dir'] + '/data.mat')
        # Get the sets of data
        print('Source')
        s_train_data, s_test_data = data_set['source_train_data'], data_set['source_test_data']
        s_train_data, s_test_data = normalize_dataset(scaler, s_train_data, s_test_data)

        # Get the sets of data
        print('Target')
        t_train_data, t_test_data = data_set['target_train_data'], data_set['target_test_data']
        t_train_data, t_test_data = normalize_dataset(scaler, t_train_data, t_test_data)

        sio.savemat(dataset['train_dir'] + '/data_normalize.mat', {
            'source_train_data': s_train_data,
            'source_test_data': s_test_data,
            'target_train_data': t_train_data,
            'target_test_data': t_test_data,
            })
        """

        # the whole dataset normalize
        train_data, train_label, train_pos, test_data, test_label, test_pos = load_data(data_list, data_pos, 80)
        train_data, test_data = normalize_dataset(scaler, train_data, test_data)
        sio.savemat(dataset['train_dir'] + '/data_set.mat', {
            'train_data': train_data, 'train_label': train_label, 'train_pos': train_pos,
            'test_data': test_data, 'test_label': test_label, 'test_pos': test_pos,
            })
