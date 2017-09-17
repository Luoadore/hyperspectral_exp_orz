# coding=utf-8
# not correct
import caffe
import numpy as np
import scipy.io as sio

data_root = 'F:\caffe-master\exp_orz\KSC_modified'
model_root = 'F:\caffe-master\exp_orz'
weight_root = 'F:\caffe-master\exp_orz'

caffe.set_mode_cpu()
model_def = model_root + '\hsi_deploy.prototxt'
model_weights = weight_root + '\\results_orignal\KSC_1_iter_10000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)
data = sio.loadmat(data_root + '\\1_1\\1\HSIKSCtestinglmdb\data.mat')
original_Data = data['data']
original_label = data['label']
num, _, _, bands = original_Data.shape
feature = []
pred = []
count = 0
for i in range(num):
    data_o = original_Data[i][0][0]
    label_o = original_label[0][i]
    #data_o = data_o.reshape(bands, 1, 1)
    data_o = np.float32(data_o.reshape(1, 1, 1, bands)) / 255.0
    #scores = net.predict([data_o])
    net.blobs['data'].data[...] = data_o
    output = net.forward()
    output_prob = output['loss'][0]
    feat = net.blobs['ip1'].data[0]
    feature.append(feat)
    #order = scores.argmax()
    pred.append(output_prob.argmax())
    if label_o == pred[-1]:
        count += 1
# sio.savemat(result_root + '\\feat_pred.mat', {'feature': feature, 'predict': pred})
print count
print pred
print len(pred)
# print feature
print 'accuracy: ', float(count) / float(len(pred))