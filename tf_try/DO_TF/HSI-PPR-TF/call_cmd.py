# coding: utf-8
"""
Call CMD to run train and test with 10 circulations at least.
"""

import subprocess

results = ['1st', '2nd', '3rd']
"""
# train baseline
for i in range(3):


    p = subprocess.Popen('python train_original.py '
                         + '--S_or_T s '
                         + '--train_ratio 80 '
                         + '--test_ratio 20 '
                         + ' --save_name ' + 'baseline/' + results[i] + '/ksc_0520_s80/',
                         shell= True)
    p.wait()

    p = subprocess.Popen('python train_original.py '
                         + '--S_or_T t '
                         + '--train_ratio 80 '
                         + '--test_ratio 20 '
                         + ' --save_name ' + 'baseline/' + results[i] + '/ksc_0520_t80/',
                         shell=True)
    p.wait()

    p = subprocess.Popen('python train_original.py '
                         + '--S_or_T s '
                         + '--train_ratio 20 '
                         + '--test_ratio 80 '
                         + ' --save_name ' + 'baseline/' + results[i] + '/ksc_0520_s20/',
                         shell= True)
    p.wait()

    p = subprocess.Popen('python train_original.py '
                         + '--S_or_T t '
                         + '--train_ratio 20 '
                         + '--test_ratio 80 '
                         + ' --save_name ' + 'baseline/' + results[i] + '/ksc_0520_t20/',
                         shell=True)
    p.wait()


# transfer learning
for i in range(3):


    p = subprocess.Popen('python train_transfer.py '
                         + '--PPR_block cd '
                         + '--learning_rate 0.1 '
                         + '--ckpt_dir ' + 'baseline/' + results[i] + '/ksc_0520_s80/'
                         + ' --save_name ' + 'transfer/' + results[i] + '/ksc_0520_strue/',
                         shell= True)
    p.wait()

    p = subprocess.Popen('python train_transfer_false.py '
                         + '--PPR_block cd '
                         + '--learning_rate 0.1 '
                         + '--ckpt_dir ' + 'baseline/' + results[i] + '/ksc_0520_s80/'
                         + ' --save_name ' + 'transfer/' + results[i] + '/ksc_0520_sfalse/',
                         shell=True)
    p.wait()

    p = subprocess.Popen('python train_transfer.py '
                         + '--PPR_block cd '
                         + '--learning_rate 0.1 '
                         + '--ckpt_dir ' + 'baseline/' + results[i] + '/ksc_0520_t80/'
                         + ' --save_name ' + 'transfer/' + results[i] + '/ksc_0520_ttrue/',
                         shell=True)
    p.wait()

    p = subprocess.Popen('python train_transfer_false.py '
                         + '--PPR_block cd '
                         + '--learning_rate 0.1 '
                         + '--ckpt_dir ' + 'baseline/' + results[i] + '/ksc_0520_t80/'
                         + ' --save_name ' + 'transfer/' + results[i] + '/ksc_0520_tfalse/',
                         shell=True)
    p.wait()"""

p = subprocess.Popen('python train_transfer_false.py '
                         + '--PPR_block cd '
                         + '--learning_rate 0.1 '
                         + '--ckpt_dir ' + 'baseline/' + results[0] + '/ksc_0520_s80/'
                         + ' --save_name ' + 'transfer/' + results[0] + '/ksc_0520_sfalse/',
                         shell=True)
p.wait()

p = subprocess.Popen('python train_transfer_false.py '
                     + '--PPR_block cd '
                     + '--learning_rate 0.1 '
                     + '--ckpt_dir ' + 'baseline/' + results[0] + '/ksc_0520_t80/'
                     + ' --save_name ' + 'transfer/' + results[0] + '/ksc_0520_tfalse/',
                         shell=True)
p.wait()