# coding: utf-8
"""Preprocessing original dataset:
    extract 8-neighbors sample data sets;
    divide original data set into trian-dataset and test-dataset.
"""

import scipy.io as sio
import numpy as np
from random import shuffle
import math
from sklearn import preprocessing as sp

def extract_data(data_file, labels_file):
    """The original data will be convert into n-neighbors label with all its bands information(value).

    Args:
        data_file: File of the original data
        labels_file: File of the original labels
        neighbor: Three use neighborhood information, 0 - single pixel, 4 - four neighbors, 8 - eight neighbors

    Return:
        data_list: Useful data set for build model and test, while the [index + 1] repersent its corresponging label
        data_pos: Data position in ClsID, include raw and col information
    """

    ############################################################
    #需不需要先做预处理，数据标准化和除去奇异值
    #
    ############################################################

    global data_position
    data = sio.loadmat(data_file)
    label = sio.loadmat(labels_file)
    data_o = data['indian_pines_corrected']
    labels = label['indian_pines_gt']
    classes = np.max(labels)
    print('there are ' + str(classes) + ' class in the data set.')
    print(labels.shape)
    rows, cols = labels.shape
    bands = np.size(data_o, 2)
    print('The data has ' + str(bands) + ' bands.')

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
                center_data = data_o[i, j]
                data_position = [i, j]

                ##################################
                #The neighbors-structer:
                #  data1      data2      data3
                #  data4   center_data   data5
                #  data6      data7      data8
                ##################################

                #judgment standard, how to choose neighbors' coordinates
                lessThan = lambda x: x - 1 if x > 0 else 0
                greaterThan_row = lambda x: x + 1 if x < rows - 1 else rows - 1
                greaterThan_col = lambda x: x + 1 if x < cols - 1 else cols - 1

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

                print('8 neighbors extract done.')
                data_list[label - 1].append(data_temp)
                data_pos[label - 1].append(data_position)

    print('Extract data done.')
    data_list, data_pos = shuffling1(data_list, data_pos)
    return data_list, data_pos

def shuffling1(data_set, data_pos):
    """Rewrite the shuffle function.

    Args:
         data_set: Original data set, numpy darray, len(data_set) repersents its label, len(data_set[i]) repersent the numbers of one class, from extract_data().
         data_pos: Original data position, numpy darray, corresponding to the data_set, from extract_data().

    Return:
         dataset: Shuffled data.
         datapos:Shuffled data correspond to data_set.
    """
    dataset = []
    datapos = []
    for eachclass, eachpos in zip(data_set, data_pos):
        num, _ = np.shape(eachclass)
        index = np.arange(num)
        shuffle(index)
        shuffled_data = []
        shuffled_pos = []
        for i in range(num):
            shuffled_data.append(eachclass[index[i]])
            shuffled_pos.append(eachpos[index[i]])
        dataset.append(shuffled_data)
        datapos.append(shuffled_pos)

    print('Shuffling1 done.')

    return dataset, datapos

