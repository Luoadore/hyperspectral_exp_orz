# coding: utf-8
"""
Call CMD to run train multiple times.
"""

import subprocess


for i in range(5):
    p = subprocess.Popen('python /media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/wgan_tf.py '
                         + '--iter 20000 '
                         + '--model_path /media/luo/result/hsi-wgan/test/exp_' + str(i) + ' ',
                         shell= True)
    print('EXP ' + str(i) + ' is proceeding...')
    p.wait()
