# coding: utf-8
"""
Call CMD to run train and test with 10 circulations at least.
"""

import subprocess

dataset = ['ksc', 'ip', 'pu', 'sa']
"""
# train
for n in range(5):

    for i in range(4):
        p = subprocess.Popen('python train.py '
                             + '--PPR_block cd '
                             + '--dataset_name ' + dataset[i]
                             + ' --save_name ' + dataset[i] + '_1118_' + str(n) + '/',
                             shell= True)
        print('--------------------dataset----------------------', dataset[i])
        p.wait()


    for i in range(4):
        p = subprocess.Popen('python train.py '
                             + '--PPR_block cd '
                             + '--dataset_name ' + dataset[i]
                             + ' --neighbor 8 --conv1_stride 9'
                             + ' --save_name ' + dataset[i] + '_1118_' + str(n) + '_cube/',
                             shell= True)
        p.wait()
    """
# fine-tuning
for n in range(1, 2):

    for i in range(4):
        p = subprocess.Popen('python train.py '
                             + '--PPR_block cd '
                             + '--learning_rate 0.01 '
                             + '--ckpt_dir ' + dataset[i] + '_1107_' + str(n) + '/'
                             + ' --dataset_name ' + dataset[i]
                             + ' --save_name ' + dataset[i] + '_1118_' + str(n) + '/',
                             shell= True)
        print('--------------------dataset----------------------', dataset[i])
        p.wait()


    for i in range(4):
        p = subprocess.Popen('python train.py '
                             + '--PPR_block cd '
                             + '--learning_rate 0.01 '
                             + '--ckpt_dir ' + dataset[i] + '_1107_' + str(n) + '_cube/'
                             + ' --dataset_name ' + dataset[i]
                             + ' --neighbor 8 --conv1_stride 9'
                             + ' --save_name ' + dataset[i] + '_1118_' + str(n) + '_cube/',
                             shell= True)
        p.wait()