# coding: utf-8
"""
Call CMD to run train and test with 10 circulations at least.
"""

import subprocess

dataset = ['ksc', 'ip', 'pu', 'sa']

for i in range(4):
    p = subprocess.Popen('python train_original.py '
                         + '--dataset_name ' + dataset[i]
                         + ' --save_name ' + dataset[i] + '_1010/',
                         shell= True)
    p.wait()


for i in range(4):
    p = subprocess.Popen('python train_original.py '
                         + '--dataset_name ' + dataset[i]
                         + ' --neighbor 8 --conv1_stride 9'
                         + ' --save_name ' + dataset[i] + '_1010_cube/',
                         shell= True)
    p.wait()
