# coding: utf-8
"""
Call CMD to run train and test with 10 circulations at least.
"""

import subprocess

for i in range(1, 3):
    p = subprocess.Popen('F:\Anaconda2\envs\py3\python.exe train_original.py --max_steps 100 '
                         + '--data_dir F:\hsi_data\KennedySpaceCenter(KSC)\KSCData.mat '
                         + '--label_dir F:\hsi_data\KennedySpaceCenter(KSC)\KSCGt.mat --train_dir F:\\result\exp_' + str(i),
                         shell= True)
    p.wait()