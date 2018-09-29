# coding: utf-8

root = '/media/luo/result/hsi/extracted_data/'

ksc = {
    'data_dir': root + 'KSCdata.mat',
    'num_classes': 13,
    'bands': 176,
    'conv1_kernel': [10, 27, 21, 12, 20, 22, 14, 19, 17, 18, 13, 16, 24, 26, 30, 7, 29, 32, 40, 15],
    'learning_rate': 0.1,
}

ip = {
    'data_dir': root + 'IPdata.mat',
    'num_classes': 16,
    'bands': 200,
    'conv1_kernel': [11, 15, 13, 14, 21, 23, 12, 22, 8, 24, 26, 18, 25, 30, 34, 28, 10, 20, 27, 40],
    'learning_rate': 0.1,
}

pu = {
    'data_dir': root + 'PUdata.mat',
    'num_classes': 9,
    'bands': 103,
    'conv1_kernel': [15, 12, 20, 35, 45, 22, 16, 19, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36],
    'learning_rate': 0.1,
}

sa = {
    'data_dir': root + 'SAdata.mat',
    'num_classes': 16,
    'bands': 204,
    'conv1_kernel': [13, 15, 24, 19, 23, 30, 14, 26, 11, 20, 27, 29, 34, 32, 25, 36, 41, 33, 16, 17],
    'learning_rate': 0.1,
}
