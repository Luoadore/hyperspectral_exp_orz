# coding: utf-8

import caffe
import numpy as np
import scipy.io as sio

root = 'F:\caffe-master\exp_orz'
model_def_d = root + '\hsi_deploy.prototxt'
model_def_o = root + '\hsi_train_test.prototxt'
model_weights = root + '\\results_orignal\KSC_1_iter_10000.caffemodel'

caffe.set_mode_cpu()
net_d = caffe.Net(model_def_d, model_weights, caffe.TEST)
net_o = caffe.Net(model_def_o, model_weights, caffe.TEST)

conv1_d_w = net_d.params['conv1'][0].data
conv1_d_b = net_d.params['conv1'][1].data
ip1_d_w = net_d.params['ip1'][0].data
ip1_d_b = net_d.params['ip1'][1].data
ip2_d_w = net_d.params['ip2'][0].data
ip2_d_b = net_d.params['ip2'][1].data

conv1_o_w = net_o.params['conv1'][0].data
conv1_o_b = net_o.params['conv1'][1].data
ip1_o_w = net_o.params['ip1'][0].data
ip1_o_b = net_o.params['ip1'][1].data
ip2_o_w = net_o.params['ip2'][0].data
ip2_o_b = net_o.params['ip2'][1].data

sio.savemat(root + '\Deploy_params.mat', {'conv1_w': conv1_d_w, 'conv1_b': conv1_d_b, 'ip1_w': ip1_d_w, 'ip1_b': ip1_d_b, 'ip2_w': ip2_d_w, 'ip2_b': ip2_d_b})
# sio.savemat(root + 'original_params.mat', {'conv1_w': conv1_o_w, 'conv1_b': conv1_o_b, 'ip1_w': ip1_o_w, 'ip1_b': ip1_o_b, 'ip2_w': ip2_o_w, 'ip2_b': ip2_o_b})

if (conv1_d_w == conv1_o_w).all():
    print 'conv1 weights the same.'
if (conv1_d_b == conv1_o_b).all():
    print 'conv1 bias the same.'
if (ip1_d_w == ip1_o_w).all():
    print 'ip1 weights the same.'
if (ip1_d_b == ip1_o_b).all():
    print 'ip1 bias the same.'
if (ip2_d_w == ip2_o_w).all():
    print 'ip2 weights the same.'
if (ip2_d_b == ip2_o_b).all():
    print 'ip2 bias the same.'