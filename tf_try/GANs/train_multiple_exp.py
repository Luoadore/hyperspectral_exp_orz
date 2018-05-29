# coding: utf-8
"""
Call CMD to run train multiple times or different class.
"""

import subprocess
"""
# Experiments many times.
for i in range(5):
    p = subprocess.Popen('python /media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/wgan_tf.py '
                         + '--iter 20000 '
                         + '--model_path /media/luo/result/hsi-wgan/test/exp_' + str(i) + ' ',
                         shell= True)
    print('EXP ' + str(i) + ' is proceeding...')
    p.wait()
"""

# Train different classes.
for i in range(13):
    p = subprocess.Popen('python /media/luo/cs/codestore/hyperspectral_exp_orz/tf_try/GANs/wgan_tf.py '
                         + '--iter 2000 '
                         + '--model_path /media/luo/result/hsi-wgan/test/exp_' + str(i) + ' '
                         + '--class_number ' + str(i) + ' '
                         + '--train_dir /media/luo/result/hsi_gan_result/KSC/hsi_data' + str(i) + '.mat',
                         shell= True)
    print('Class ' + str(i) + ' is proceeding...')
    p.wait()