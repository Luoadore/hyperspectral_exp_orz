#coding: utf-8
import numpy as np
import lmdb
import caffe
import scipy.io as sio

dataset = []
labels = []
env = lmdb.open('F:\caffe-master\exp_orz\KSC_modified\\1_1\\1\HSIKSCtestinglmdb', readonly = True)
with env.begin() as txn:
    #生成迭代器指针
    cursor = txn.cursor()
    for key, value in cursor:# 循环获取数据
        #print key,len(value)
        print 'key: ',key
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value) #从value中获取数据
        label = datum.label
        labels.append(label)
        #print label
        #flat_x = np.fromstring(datum.data, dtype = np.uint8)
        flat_x = np.fromstring(datum.data, dtype = np.uint16)
        #print flat_x.shape
        print flat_x
        data_x = flat_x.reshape(datum.channels,datum.height, datum.width)
        dataset.append(data_x)
        #data = caffe.io.datum_to_array(datum)
        #print data_x.shape
        #print datum.channels
        #print datum.height
        #print datum.width
    sio.savemat('F:\caffe-master\exp_orz\KSC_modified\\1_1\\1\HSIKSCtestinglmdb\data.mat', {'data': dataset, 'label': labels})
#print zip(dataset, labels)
#with env.begin() as txn:
#   raw_datum = txn.get()
#   datum = caffe.proto.caffe_pb2.Datum()
#   datum.ParseFromString(raw_datum)



