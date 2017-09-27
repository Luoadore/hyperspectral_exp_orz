# coding: utf-8
"""Preprocessing original dataset:
    extract single-pixel or 4-neighbors or 8-neighbors sample data sets;
    divide original data set into trian-dataset and test-dataset.
"""

import scipy.io as sio
import numpy as np
import random

def extract_data(data_file, labels_file, neighbor):
    """The original data will be convert into n-neighbors label with all its bands information(value).
    
    Args:
        data_file: File of the original data
        labels_file: File of the original labels
        neighbor: Three use neighborhood information, 0 - single pixel, 4 - four neighbors, 8 - eight neighbors
        
    Return:
        data_set: Useful data set for build model and test
        label_set: Useful label set for build model and canculate accuarcy
    """
    
    data = sio.loadmat(data_file)
    label = sio.loadmat(labels_file)
    data_o = data['DataSet']
    labels = label['ClsID']
    classes = np.max(labels)
    print('there are ' + str(classes) + 'class in the data set.')
    rows, cols = labels.shape()
    
    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label != 0:
                data_temp = data_o[i,j]
                if neighbor > 0:
                    center_data = data_temp
                    data1 = []
                    data2 = []
                    data3 = []
                    data4 = []
                    data5 = []
                    data6 = []
                    data7 = []
                    data8 = []
                    #data1
                    if 

def load_data(dataset, ratio):
    """
    """