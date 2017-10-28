# coding=utf-8
# This code mainly use to validate the data transformed from lmdb right or not.
import numpy as np
import scipy.io as sio

DataSetMat = sio.loadmat('H:\luoadore\hyperspectralWork\KSC\Data.mat')
LabelsMat = sio.loadmat('H:\luoadore\hyperspectralWork\KSC\Gt.mat')
DataSet = DataSetMat['DataSet']
Labels = LabelsMat['ClsID']
rows = len(Labels)
lines = len(Labels[0])
DataList = []
print 'size of the dataset: ' + str(rows), str(lines)

for indexRow in range(rows):
    #print indexRow
    for indexLine in range(lines):
        #print indexLine
        label = Labels[indexRow, indexLine]
        #store non-zero data
        if label == 9:
            #for test purpose printing...
            data = DataSet[indexRow, indexLine]
            DataList.append(data)
    
print 'data loaded.'
print 'spectral length now is: ' + str(len(DataList))
print str(len(DataList[0]))

sample = sio.loadmat('F:\caffe-master\exp_orz\KSC_modified\\1_1\\1\HSIKSCtestinglmdb\data.mat')
sample_data = sample['data'][0][0][0]
sample_label = sample['label'][0]
print sample_label
print sample_data

count = 0
for eachline in DataList:
    if (sample_data == eachline).all():
        print 'good job.'
        count += 1
print count


