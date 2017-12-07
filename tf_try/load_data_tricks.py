# coding: utf-8
"""Load original dataset:
    extract single-pixel or 4-neighbors or 8-neighbors sample data sets;
    divide original data set into trian-dataset and test-dataset.
"""

import scipy.io as sio
import numpy as np
import data_preprocess_pos as dp

def extract_data_cen_bor(data_file, labels_file, neighbor):
    """The original data will be convert into n-neighbors label with all its bands information(value).
       Divide original classes into two class: center and border, train and test respectively.
       So the category of the data set is extend to two times by original.

    Args:
        data_file: File of the original data
        labels_file: File of the original labels
        neighbor: Three use neighborhood information, 0 - single pixel, 4 - four neighbors, 8 - eight neighbors

    Return:
        data_list: Useful data set for build model and test, while the [index + 1] repersent its corresponging label
        data_pos: Data position in ClsID, include raw and col information
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

    data_list = []
    classes_extend = int(classes) * 2
    print(classes_extend)

    for mark in range(classes_extend):
        data_list.append([])

    data_pos = []
    for mark in range(classes_extend):
        data_pos.append([])

    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label != 0:
                data_temp = data_o[i, j]
                data_position = [i, j]

                #judgment standard, how to choose neighbors' coordinates
                lessThan = lambda x: x - 1 if x > 0 else 0
                greaterThan_row = lambda x: x + 1 if x < rows - 1 else rows - 1
                greaterThan_col = lambda x: x + 1 if x < cols - 1 else cols - 1

                data_label = []
                data_label.extend([labels[lessThan(i), lessThan(j)], labels[lessThan(i), j], labels[lessThan(i), greaterThan_col(j)],
                                  labels[i, lessThan(j)], labels[i, greaterThan_col(j)],
                                  labels[greaterThan_row(i), lessThan(j)], labels[greaterThan_row(i), j], labels[greaterThan_row(i), greaterThan_col(j)]])

                data_flag = 0    # center
                for idx in range(len(data_label)):
                    if data_label[idx] != label:
                        data_flag = 1    # border
                    else:
                        continue

                if neighbor > 0:
                    center_data = data_temp
                    ##################################
                    #The neighbors-structer:
                    #  data1      data2      data3
                    #  data4   center_data   data5
                    #  data6      data7      data8
                    ##################################

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
                        # print(data_temp.shape)
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

                if data_flag == 0:
                    data_list[label - 1].append(data_temp)
                    data_pos[label - 1].append(data_position)
                else:
                    data_list[label - 1 + classes].append(data_temp)
                    data_pos[label - 1 + classes].append(data_position)

    print(data_list)
    print(len(data_pos))
    print('Extract data done.')
    return data_list, data_pos

def load_data_equal_by_class(dataset, datapos):
    """Load percific train data set and test data set according to ratio.

    Args：
        dataset: Including data and label, from extract_data()
        datapos: Data location information, from extract_data()

    Return:
        train_data: Numpy darray, train data set value
        train_label: Numpy darray, train label
        test_data: Numpy darray, train data set value
        test_label: Numpy darray, test label
        train_pos: Numpy darray, train data position
        test_pos: Numpy darray, test data position
    """
    data_num = 0
    class_num = []
    for eachclass in dataset:
        class_num.append(len(eachclass))
        data_num += len(eachclass)
    print('There are ' + str(data_num) + ' examples in data set.')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    train_pos = []
    test_pos = []
    shuffle_data, shuffle_pos = dp.shuffling1(dataset, datapos)
    for classes, eachclass in enumerate(shuffle_data):
        trainingNumber = min(class_num)
        testingNumber = len(eachclass) - trainingNumber
        #print('the ' + str(classes) +' class has ' + str(trainingNumber) + ' training examples and ' + str(testingNumber) + ' testing examples.')
        for i in range(trainingNumber):
            train_data.append(eachclass[i])
            train_label.append(classes)
            train_pos.append(datapos[classes][i])
        for i in range(testingNumber):
            test_data.append(eachclass[trainingNumber + i])
            test_label.append(classes)
            test_pos.append(datapos[classes][trainingNumber + i])

    print('load train: ' + str(len(train_data)) + ', ' + str(len(train_label)))
    print('load test: ' + str(len(test_data)) + ', ' + str(len(test_label)))
    #shuffle all the data set
    train_data, train_label, train_pos = dp.shuffling2(train_data, train_label, train_pos)
    test_data, test_label, test_pos = dp.shuffling2(test_data, test_label, test_pos)
    scaler = sp.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    #print('train data normalize:')
    #print(train_data[0])
    print('Load data.')

    return train_data, train_label, train_pos, test_data, test_label, test_pos