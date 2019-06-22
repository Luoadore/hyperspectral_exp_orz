# coding: utf-8
"""
Call CMD to run train and alternate kernel size each time.
"""

import subprocess


# kernels = [15, 24, 39]
"""for i in range(15, 40):
    p = subprocess.Popen('python /media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/CNN/train_original.py '
                         + '--max_steps 20000 '
                         + '--conv1_kernel ' + str(i * 9) + ' '
                         + '--data_dir /media/luo/result/hsi/extracted_data/KSCdata.mat '
                         + '--train_dir /media/luo/result/hsi_kernels_r/ksc/exp_' + str(i),
                         shell= True)
    print('EXP ' + str(i) + ' is proceeding...')
    p.wait()
"""

for i in range(5, 51):
    p = subprocess.Popen('python /media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/CNN/train_original.py '
                         + '--max_steps 20000 '
                         + '--conv1_kernel ' + str(i * 9) + ' '
                         + '--data_dir /media/luo/result/hsi/extracted_data/PUdata.mat '
                         + '--train_dir /media/luo/result/hsi_kernels_r/PU_2/exp_' + str(i),
                         shell= True)
    print('EXP ' + str(i) + ' is proceeding...')
    p.wait()