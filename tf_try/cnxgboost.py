# coding: utf-8
"""
Use the full-connection layer's output of CNN network as feature, feeding into xgboost train and classify.
"""

#import python_analyze.read_data as rd
import read_data as rd
import xgboost as xgb
import numpy as np


# 获得实验数据
_, tr_lab, tr_pred, tr_data, _, te_lab, te_pred, te_data = rd.get_data('F:\\tf-try\\result\\data.mat')
dtrain = xgb.DMatrix(tr_data, label=np.transpose(tr_lab))
dtest = xgb.DMatrix(te_data, label=np.transpose(te_lab))

param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 1
param['num_class'] = 13

watchlist = [(dtest, 'test'), (dtrain, 'train')]

# 训练分类器
num_round = 5
bst = xgb.train(param, dtrain, num_round, watchlist)

# 测试分类器
pred = bst.predict(dtest)
print(pred)
print(te_lab)
error_rate = np.sum([pred != te_lab]) / te_lab.shape[1]
print('\nTest error using softmax = {}\n'.format(error_rate))

# 使用输出分类概率
param['objective'] = 'multi:softprob'
bst = xgb.train(param, dtrain, num_round, watchlist)
# 测试预测
pred_prob = bst.predict(dtest).reshape(te_lab.shape[1], 13)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != te_lab) / te_lab.shape[1]
print('\nTest error using softprob = {}\n'.format(error_rate))