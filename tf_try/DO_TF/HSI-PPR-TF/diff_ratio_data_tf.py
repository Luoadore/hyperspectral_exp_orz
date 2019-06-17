# coding: utf-8

from select_domaindata import *
from normalize import *
from config import ksc
import subprocess

def divide_data(ratio, scaler):
    """
    Divide data set into train and test according to ratio.

    :param ratio: %
    :param scaler: data mean and std
    :return: nothing
    """
    dataset = ksc
    neighbor = 8

    # Get the sets of data
    print('Source')
    center = [170, 307]
    print('center: ', center)
    s_data_set, s_data_pos = select_data(center, [200, 300], dataset['data_dir'], dataset['label_dir'], neighbor)
    s_train_data, s_train_label, s_train_pos, s_test_data, s_test_label, s_test_pos = load_data(s_data_set, s_data_pos,
                                                                                                ratio)
    s_train_data, s_test_data = normalize_dataset(scaler, s_train_data, s_test_data)
    print('train label length: ' + str(len(s_train_label)) + ', train data length: ' + str(len(s_train_data)))
    print('test label length:' + str(len(s_test_label)) + ', test data length: ' + str(len(s_test_data)))
    print('label num: ', set(s_train_label))

    # Get the sets of data
    print('Target')
    center = [342, 460]
    print('center: ', center)
    t_data_set, t_data_pos = select_data(center, [200, 300], dataset['data_dir'], dataset['label_dir'], neighbor)
    t_train_data, t_train_label, t_train_pos, t_test_data, t_test_label, t_test_pos = load_data(t_data_set, t_data_pos,
                                                                                                ratio)
    t_train_data, t_test_data = normalize_dataset(scaler, t_train_data, t_test_data)
    print('train label length: ' + str(len(t_train_label)) + ', train data length: ' + str(len(t_train_data)))
    print('test label length:' + str(len(t_test_label)) + ', test data length: ' + str(len(t_test_data)))
    print('label num: ', set(t_train_label))

    sio.savemat(dataset['train_dir'] + '/data' + str(ratio) + '.mat', {
        'source_train_data': s_train_data, 'source_train_label': s_train_label, 'source_train_pos': s_train_pos,
        'source_test_data': s_test_data, 'source_test_label': s_test_label, 'source_test_pos': s_test_pos,
        'target_train_data': t_train_data, 'target_train_label': t_train_label, 'target_train_pos': t_train_pos,
        'target_test_data': t_test_data, 'target_test_label': t_test_label, 'target_test_pos': t_test_pos,
    })

if __name__ == '__main__':

    # prepare
    # ratio = [18, 15, 12, 10, 8, 5, 3, 2, 1]
    ratio = [18, 15, 12]
    dataset = ksc
    neighbor = 8
    # scaler, _, _ = mean_std(dataset['data_dir'], dataset['label_dir'], neighbor)

    """
    for r in ratio:
        print('Start dividing dataset into:', ratio)
        divide_data(r, scaler)
        print('Done.')"""

    """
    # different ratio
    for i in range(len(ratio)):

        print('Train S-net: -----------------------------')

        p = subprocess.Popen('python train_original.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps 5000 '
                             + '--data_name data' + str(ratio[i]) + '.mat'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/0320_' + str(ratio[i]) + '/'
                             + ' --is_training True',
                             shell=True)
        p.wait()


        print('Train T-net: -----------------------------')

        print('All parameters training:')
        p = subprocess.Popen('python train_transfer.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps 5000 '
                             + '--data_name data' + str(ratio[i]) + '.mat'
                             + ' --ckpt_dir ' + ksc['train_dir'] + 'results/ratio_test/0320_' + str(ratio[i]) + '/'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/ratio_test/0320_true_' + str(ratio[i]) + '/',
                             shell=True)
        p.wait()


        print('All parameters not training:')
        p = subprocess.Popen('python train_transfer_false.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps 5000 '
                             + '--data_name data' + str(ratio[i]) + '.mat'
                             + ' --ckpt_dir ' + ksc['train_dir'] + 'results/0320_' + str(ratio[i]) + '/'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/0320_false_' + str(ratio[i]) + '/',
                             shell=True)
        p.wait()"""


    # different learning rate of 1%
    lr = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    iters = [20000, 20000, 10000, 10000, 10000, 10000, 5000, 5000]

    for i in range(len(lr)):
        print('Train S-net: -----------------------------')

        p = subprocess.Popen('python train_original.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps ' + str(iters[i])
                             + ' --data_name data20.mat'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/lr_test/0326_' + str(lr[i]) + '/'
                             + ' --is_training True',
                             shell=True)
        p.wait()

        print('Train T-net: -----------------------------')

        print('All parameters training:')
        p = subprocess.Popen('python train_transfer.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps ' + str(iters[i])
                             + ' --data_name data20.mat'
                             + ' --ckpt_dir ' + ksc['train_dir'] + 'results/lr_test/0326_' + str(lr[i]) + '/'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/lr_test/0326_true_' + str(lr[i]) + '/',
                             shell=True)
        p.wait()

        print('All parameters not training:')
        p = subprocess.Popen('python train_transfer_false.py '
                             + '--learning_rate 0.1 '
                             + '--max_steps ' + str(iters[i])
                             + ' --data_name data20.mat'
                             + ' --ckpt_dir ' + ksc['train_dir'] + 'results/lr_test/0326_' + str(lr[i]) + '/'
                             + ' --train_dir ' + ksc['train_dir'] + 'results/lr_test/0326_false_' + str(lr[i]) + '/',
                             shell=True)
        p.wait()
