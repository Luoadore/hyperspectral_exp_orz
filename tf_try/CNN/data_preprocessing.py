# coding: utf-8
"""Preprocessing original dataset:
    extract single-pixel or 4-neighbors or 8-neighbors sample data sets;
    divide original data set into trian-dataset and test-dataset.
"""

import scipy.io as sio
import numpy as np
from random import shuffle
import math
from sklearn import preprocessing as sp

def extract_data(data_file, labels_file, neighbor):
    """The original data will be convert into n-neighbors label with all its bands information(value).
    
    Args:
        data_file: File of the original data
        labels_file: File of the original labels
        neighbor: Three use neighborhood information, 0 - single pixel, 4 - four neighbors, 8 - eight neighbors
        
    Return:
        data_list: Useful data set for build model and test, while the [index + 1] repersent its corresponging label
    """
    
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
    
    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label != 0:
                data_temp = data_o[i, j]
                if neighbor > 0:
                    center_data = data_temp
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
                    
                    if neighbor == 4:
                        data_temp = []
                        for each in range(bands):
                            data2 = data_o[lessThan(i), j][each]
                            data4 = data_o[i, lessThan(j)][each]
                            data5 = data_o[i, greaterThan_col(j)][each]
                            data7 = data_o[greaterThan_row(i), j][each]
                            data_1 = np.append(data2, data4)
                            data_2 = np.append(data_1, center_data[each])
                            data_3 = np.append(data5, data7)
                            data_t = np.append(data_2, data_3)
                            data_temp = np.append(data_temp, data_t)
                        print(data_temp.shape)
                        print('4 neighbors extract done.')

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
                        print('8 neighbors extract done.')
                
                data_list[label - 1].append(data_temp)
    
    print('Extract data done.')
    return data_list

def onehot_label(labels, num_class):
    """To transfer labels into one-hot values.

    Arg:
        labels: Original label set.
        num_class: Classes of data.

    Return:
        onehot_labels: One-hot labels to fit in network.
    """
    #data_size = tf.size(labels)
    #labels = tf.expand_dims(labels, 1)
    #indices = tf.expand_dims(tf.range(0, data_size), 1)
    #concated = tf.concat([indices, labels], 1)
    #onehot_labels = tf.sparse_to_dense(concated, tf.stack([data_size, oc.NUM_CLASSES]), 1.0, 0.0)
    #enc = sp.OneHotEncoder(n_values = [num_class])
    #enc.fit([[num_class - 1], [random.randint(0, num_class - 1)]])
    #onehot_labels = []
    #print(enc.feature_indices_)
    #for i in range(len(labels)):
    #    onehot_labels.append(enc.transform([[labels[i]]]).toarray().reshape(num_class))
    onehot_labels = np.zeros((len(labels), num_class), float)
    for i in range(len(labels)):
        onehot_labels[i][labels[i]] = 1
    print('One hot label transformed done.')

    return onehot_labels

def shuffling1(data_set):
    """Rewrite the shuffle function.
    
    Args:
         data_set: Original data set, numpy darray, len(data_set) repersents its label, len(data_set[i]) repersent the numbers of one class, from extract_data().
   
    Return:
         data_set: Shuffled data.
    """
    for eachclass in data_set:
        shuffle(eachclass)
    print('Shuffling1 done.')

    return data_set
    
def shuffling2(data_set, label_set):
    """Rewrite the shuffle function. The data is ordered according to the category, which need to transfrom into out-of-order data set for train net model.
    
    Args:
        data_set: Data numpy darray, from load_data()
        label_set: Label set, from load_data()
        
    Return:
        shuffled_data: Out-of-order class data
        shuffled_label: Corresponding label.
    """
    num = len(label_set)
    index = np.arange(num)
    shuffle(index)
    shuffled_data = []
    shuffled_label = []
    for i in range(num):
        shuffled_data.append(data_set[index[i]])
        shuffled_label.append(label_set[index[i]])

    print('Shuffling2 done.')

    return shuffled_data, shuffled_label
    
def load_data(dataset, ratio):
    """Load percific train data set and test data set according to ratio.
    
    Args：
        dataset: Including data and label, from extract_data()
        ratio: Hundred times of Train data's ratio in the whole data set, real_ratio = ratio / 100
    
    Return:
        train_data: Numpy darray, train data set value
        train_label: Numpy darray, train label
        test_data: Numpy darray, train data set value
        test_label: Numpy darray, test label
    """
    data_num = 0
    for eachclass in dataset:
        data_num += len(eachclass)
    print('There are ' + str(data_num) + ' examples in data set.')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    shuffle_data = shuffling1(dataset)
    for classes, eachclass in enumerate(shuffle_data):
        trainingNumber = int(math.ceil(len(eachclass) * int(ratio)) / 100.0)
        testingNumber = int(len(eachclass) - trainingNumber)
        #print('the ' + str(classes) +' class has ' + str(trainingNumber) + ' training examples and ' + str(testingNumber) + ' testing examples.')
        for i in range(trainingNumber):
            train_data.append(eachclass[i])
            train_label.append(classes)
        for i in range(testingNumber):
            test_data.append(eachclass[trainingNumber + i])
            test_label.append(classes)

    print('load train: ' + str(len(train_data)) + ', ' + str(len(train_label)))
    print('load test: ' + str(len(test_data)) + ', ' + str(len(test_label)))
    #shuffle all the data set
    train_data, train_label = shuffling2(train_data, train_label)
    test_data, test_label = shuffling2(test_data, test_label)
    scaler = sp.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    #print('train data normalize:')
    #print(train_data[0])
    print('Load data.')

    return train_data, train_label, test_data, test_label