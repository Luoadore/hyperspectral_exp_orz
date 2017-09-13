# coding=utf-8
import caffe
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Set Caffe to CPU mode and load the net from disk.
caffe.set_mode_cpu()
# defines the structure of the model
model_def = 'F:\caffe-master\exp_orz\hsi_deploy.prototxt'
# contains the trained weights
model_weights = 'F:\caffe-master\exp_orz\\results_orignal\KSC_1_iter_10000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)  # use test mode

data = sio.loadmat('F:\caffe-master\exp_orz\KSC_modified\\1_1\\1\HSIKSCtestinglmdb\data.mat')
testData = data['data']
testlabel = data['label']
data_1 = testData[0][0][0]
label_1 = testlabel[0][0]
print data_1
print len(data_1)
print label_1
data_1 = data_1.reshape(1, 1, 176, 1)
print data_1
lmdbData = caffe.io.datum_to_array(data_1, label_1)
net.blobs['data'].data[...] = transformer.preprocess('data')   # 将数据载入blob中
#data = 'F:\caffe-master\exp_orz\KSC_modified\\1_1\\1\HSIKSCtestinglmdb'
out = net.forward()  #执行测试


# net = caffe.Classifier(model_def, model_weights)
# net.set_phase_test()
# output = open('F:\caffe-master\exp_orz\\results_orignal\\feature.txt', 'w')
prob = net.blobs['loss'].data[0].flatten()   #取出最后一层属于某个类别的概率值
print prob
order = prob.argsort()[-1]
allLabel = np.linspace(0, 12, 13)
print allLabel
print 'the predict class is: ' + allLabel(order)

conv1_f = net.blobs['conv1'].data[0]

# sio.savemat('F:\caffe-master\exp_orz\\results_orignal\parameters.mat', {'conv1_w': conv1_w, 'conv1_b': conv1_b})
print conv1_f