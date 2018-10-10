# coding: utf-8
"""
Config of datasets.

    data_dir: The filename of data.
    train_dir: The train result save file.
    num_classes: Categroies.
    bands: Nums of bands of pixel.
    conv1_kernel: Kernels of convolution layer 1.
    learning_rate: Learning rate of training.
"""

root = '/media/luo/result/hsi/extracted_data/'
result_root = '/media/luo/result/hsi_ppr_result/'

ksc = {
    'data_dir': root + 'KSCdata.mat',
    'train_dir': result_root + 'ksc/',
    'num_classes': 13,
    'bands': 176,
    'conv1_kernel': [10, 10, 10, 10, 10, 27, 27, 27, 27, 27, 21, 21, 21, 21, 21, 12, 12, 12, 12, 12],
    'learning_rate': 0.1,
}

ip = {
    'data_dir': root + 'IPdata.mat',
    'train_dir': result_root + 'ip/',
    'num_classes': 16,
    'bands': 200,
    'conv1_kernel': [11, 11, 11, 11, 11, 15, 15, 15, 15, 15, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14],
    'learning_rate': 0.1,
}

pu = {
    'data_dir': root + 'PUdata.mat',
    'train_dir': result_root + 'pu/',
    'num_classes': 9,
    'bands': 103,
    'conv1_kernel': [15, 15, 15, 15, 15, 12, 12, 12, 12, 12, 20, 20, 20, 20, 20, 35, 35, 35, 35, 35],
    'learning_rate': 0.1,
}

sa = {
    'data_dir': root + 'SAdata.mat',
    'train_dir': result_root + 'sa/',
    'num_classes': 16,
    'bands': 204,
    'conv1_kernel': [13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 19, 19, 19, 19, 19],
    'learning_rate': 0.1,
}
