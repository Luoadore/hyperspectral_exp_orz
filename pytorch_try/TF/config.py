# coding: utf-8
"""
Config of deviding dataset for training and testing.

    data_dir: The filename of raw data.
    label_dir: The filename of corresponding label.
    train_dir: The train result save file.
    num_classes: Categroies.
    bands: Nums of bands of pixel.
    conv1_kernel: Kernels of convolution layer 1.
    learning_rate: Learning rate of training.
"""

root = '/media/luo/result/hsi/'
result_root = '/media/luo/result/hsi_transfer/'

ksc = {
    'data_dir': root + 'KSC/KSCData.mat',
    'label_dir': root + 'KSC/KSCGt.mat',
    'train_dir': result_root + 'ksc/',
    'num_classes': 13,
    'bands': 176,
    'rectangle_size_train':[112, 269, 270, 403],
    'rectangle_size_test':[305, 443, 434, 545],
    'conv1_kernel': [10, 10, 10, 10, 10, 27, 27, 27, 27, 27, 21, 21, 21, 21, 21, 12, 12, 12, 12, 12],
    'learning_rate': 0.1,
}

ip = {
    'data_dir': root + 'IP/IPdata.mat',
    'label_dir': root + 'IP/IPGt.mat',
    'train_dir': result_root + 'ip/',
    'num_classes': 16,
    'bands': 200,
    'rectangle_size_train':[],
    'rectangle_size_test':[],
    'conv1_kernel': [11, 11, 11, 11, 11, 15, 15, 15, 15, 15, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14],
    'learning_rate': 0.1,
}

pu = {
    'data_dir': root + 'PU/PUData.mat',
    'label_dir': root + 'PU/PUGt.mat',
    'train_dir': result_root + 'pu/',
    'num_classes': 9,
    'bands': 103,
    'rectangle_size_train':[],
    'rectangle_size_test':[],
     'conv1_kernel': [15, 15, 15, 15, 15, 12, 12, 12, 12, 12, 20, 20, 20, 20, 20, 35, 35, 35, 35, 35],
    'learning_rate': 0.1,
}

sa = {
    'data_dir': root + 'SA/SAData.mat',
    'label_dir': root + 'SA/SAGt.mat',
    'train_dir': result_root + 'sa/',
    'num_classes': 16,
    'bands': 204,
    'rectangle_size_train':[],
    'rectangle_size_test':[],
    'conv1_kernel': [13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 19, 19, 19, 19, 19],
    'learning_rate': 0.1,
}