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
    # 'conv1_kernel': [10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15,
    #                 10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15],
    # 'conv1_kernel': [10, 27, 21, 12],
    # 'conv1_kernel': [10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15],
    'conv1_kernel': [10, 10, 10, 10, 10, 27, 27, 27, 27, 27, 21, 21, 21, 21, 21, 12, 12, 12, 12, 12],
    'learning_rate': 0.1,
}

ip = {
    'data_dir': root + 'IPdata.mat',
    'train_dir': result_root + 'ip/',
    'num_classes': 16,
    'bands': 200,
    #'conv1_kernel': [11, 15, 13, 14, 21, 23, 12, 22, 8, 24, 26, 18, 25, 30, 34, 28, 10, 20, 27, 40,
    #                 11, 15, 13, 14, 21, 23, 12, 22, 8, 24, 26, 18, 25, 30, 34, 28, 10, 20, 27, 40],
    # 'conv1_kernel': [11, 15, 13, 14],
    # 'conv1_kernel': [11, 15, 13, 14, 21, 23, 12, 22, 8, 24, 26, 18, 25, 30, 34, 28, 10, 20, 27, 40],
    'conv1_kernel': [11, 11, 11, 11, 11, 15, 15, 15, 15, 15, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14],
    'learning_rate': 0.1,
}

pu = {
    'data_dir': root + 'PUdata.mat',
    'train_dir': result_root + 'pu/',
    'num_classes': 9,
    'bands': 103,
    #'conv1_kernel': [15, 12, 20, 35, 45, 22, 16, 19, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36,
    #                 15, 12, 20, 35, 45, 22, 16, 19, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36],
    # 'conv1_kernel': [15, 12, 20, 35],
    # 'conv1_kernel': [15, 12, 20, 35, 45, 22, 16, 19, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36],
     'conv1_kernel': [15, 15, 15, 15, 15, 12, 12, 12, 12, 12, 20, 20, 20, 20, 20, 35, 35, 35, 35, 35],
    'learning_rate': 0.1,
}

sa = {
    'data_dir': root + 'SAdata.mat',
    'train_dir': result_root + 'sa/',
    'num_classes': 16,
    'bands': 204,
    #'conv1_kernel': [13, 15, 24, 19, 23, 30, 14, 26, 11, 20, 27, 29, 34, 32, 25, 36, 41, 33, 16, 17,
    #                 13, 15, 24, 19, 23, 30, 14, 26, 11, 20, 27, 29, 34, 32, 25, 36, 41, 33, 16, 17],
    # 'conv1_kernel': [13, 15, 24, 19],
    # 'conv1_kernel': [13, 15, 24, 19, 23, 30, 14, 26, 11, 20, 27, 29, 34, 32, 25, 36, 41, 33, 16, 17],
    'conv1_kernel': [13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 19, 19, 19, 19, 19],
    'learning_rate': 0.1,
}
