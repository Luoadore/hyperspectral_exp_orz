# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio

from params import param

data_file = os.path.join(param.pre, param.hsi.data_file)
class_number = param.hsi.class_number


class DataSet:
    def __init__(self, data, label, class_number):
        self.data = data
        self.label = label
        self.length = data.shape[0]
        self.data_length = data.shape[1]
        self.label_length = class_number

    def next_one(self, shuffle=True):
        indexes = np.arange(self.length)
        while True:
            if shuffle:
                np.random.shuffle(indexes)
            for i in indexes:
                yield self.data[i, :], self.label[i, 0]

    def next_batch(self, batch_size, shuffle=False):
        produce = self.next_one(shuffle)
        datas = np.zeros(shape=[batch_size, self.data_length], dtype=np.float32)
        labels = np.zeros(shape=[batch_size, self.label_length], dtype=np.float32)
        index = 0
        for data, label in produce:
            datas[index, :] = data[:]
            labels[index, label] = 1.0
            index = (index + 1) % batch_size
            if index == 0:
                yield datas, labels
                datas = np.zeros(shape=[batch_size, self.data_length], dtype=np.float32)
                labels = np.zeros(shape=[batch_size, self.label_length], dtype=np.float32)


class DataManager:
    def __init__(self, file=data_file, class_number=class_number):
        data = sio.loadmat(file)
        train_data = data['train_data']
        train_label = np.transpose(data['train_label'])
        test_data = data['test_data']
        test_label = np.transpose(data['test_label'])
        self.train = DataSet(train_data, train_label, class_number)
        self.test = DataSet(test_data, test_label, class_number)
