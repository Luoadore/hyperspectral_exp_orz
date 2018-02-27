# coding: utf-8
"""Preprocessing original dataset:
    extract 8-neighbors sample data sets;
    divide original data set into trian-dataset and test-dataset.
"""

import scipy.io as sio
import numpy as np
from random import shuffle
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
    global classes
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

    data_list = []
    for mark in range(classes):
        data_list.append([])

    data_pos = []
    for mark in range(classes):
        data_pos.append([])

    data_norm = []

    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label != 0:
                center_data = data_o[i, j]
                data_position = [i, j]
                data_list[label - 1].append(remove_singular(center_data))
                data_pos[label - 1].append(data_position)
                data_norm.append(remove_singular(center_data))

    print('Extract data done.')
    data_list, data_pos = shuffling1(data_list, data_pos)

    # normalization
    datalist = []
    scaler = sp.StandardScaler().fit(data_norm)
    for each in data_list:
        each = scaler.transform(each)
        print('Normalization done.')
        datalist.append(each)

    print(datalist[0][0])

    return datalist, data_pos

def remove_singular(value):
    """
    Remove singular if value is bigger than 65534.
    Arg:
        value: Single value of original data from extract_data()
    Return:
        value(correction): Value modification, always 0.
    """
    value[value >= 65534] = 0
    return value

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

def onehot_label(label, num_labels, num_class):
    """To transfer labels into one-hot values.

    Arg:
        label: Original label.
        num_labels: Total labels.
        num_class: Total classes of data.

    Return:
        onehot_labels: One-hot labels to fit in network.
    """
    onehot_labels = np.zeros((num_labels, num_class), float)
    for i in range(num_labels):
        onehot_labels[i][label] = 1
    print('One hot label transformed done.')

    return onehot_labels

if __name__ == '__main__':
    data_file = 'F:\hsi_data\KennedySpaceCenter(KSC)\KSCData.mat'
    label_file = 'F:\hsi_data\KennedySpaceCenter(KSC)\KSCGt'
    hsi_data = extract_data(data_file, label_file)
    print(classes)
    for i in range(classes):
        data = hsi_data[0][i]
        label = onehot_label(i, len(data), classes)
        sio.savemat('D:\hsi_gan_result\KSC\hsi_data' + str(i) + '.mat', {'data': data, 'label': label})
        print('Save ' + str(i) + '-class data done.')

    print('Yeah.')