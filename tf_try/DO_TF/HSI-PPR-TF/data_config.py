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

root = '/media/luo/result/hsi_transfer/'
result_root = '/media/luo/result/hsi_transfer/'

ksc = {
    'data_dir': root + 'ksc/data/data_baseline.mat',
    'train_dir': result_root + 'ksc/tf-results/',
    'num_classes': 13,
    'bands': 176,
    # 'conv1_kernel': [10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15,
    #                 10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15],
    # 'conv1_kernel': [10, 27, 21, 12],
    # 'conv1_kernel': [10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15],
    'conv1_kernel': [10, 10, 10, 10, 10, 27, 27, 27, 27, 27, 21, 21, 21, 21, 21, 12, 12, 12, 12, 12],
    #'conv1_kernel': [15, 15, 15, 15, 15, 24, 24, 24, 24, 24, 39, 39, 39, 39, 39, 62, 62, 62, 62, 62],
    'learning_rate': 0.1,
}
