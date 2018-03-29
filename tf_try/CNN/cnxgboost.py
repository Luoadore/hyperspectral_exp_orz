# coding: utf-8
"""
Use the full-connection layer's output of CNN network as feature, feeding into xgboost train and classify.
"""


#import python_analyze.read_data as rd
import xgboost as xgb
import numpy as np
import scipy.io as sio

# 获得实验数据
#_, tr_lab, tr_pred, tr_data, _, te_lab, te_pred, te_data = rd.get_data('F:\\tf-try\\result\\exp1\data.mat')
for i in range(1, 10):
    data = sio.loadmat('E:\exp_result\SA\exp_' + str(i) + '\data.mat')
    tr_data = data['train_fea']
    tr_lab = data['train_label']
    te_data = data['test_fea']
    te_lab = data['test_label']
    tr_data = tr_data.reshape(tr_data.shape[0], -1)
    te_data = te_data.reshape(te_data.shape[0], -1)
    print((tr_data).shape)
    dtrain = xgb.DMatrix(tr_data, label=np.transpose(tr_lab))
    dtest = xgb.DMatrix(te_data, label=np.transpose(te_lab))

    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    #param['nthread'] = 4 运行的最大线程数
    param['num_class'] = 26
    #param['eval_metric'] = 'mlogloss'  #mlogloss or merror used for multi-class classfication
    #param['scale_pos_weight'] = 1
    #param['colsample_bytree'] = 0.9

    watchlist = [(dtest, 'test'), (dtrain, 'train')]

    # 训练分类器
    num_round = 500
    bst = xgb.train(param, dtrain, num_round, watchlist)

    # 测试分类器
    pred = bst.predict(dtest)
    print(pred)
    print(te_lab)
    error_rate = np.sum([pred != te_lab]) / te_lab.shape[1]
    acc = np.sum([pred == te_lab]) / te_lab.shape[1]
    print('\nTest error using softmax = {}\n'.format(error_rate))
    print('\nTest accuracy using softmax = {}\n'.format(acc))
    pred = bst.predict(dtrain)
    acc_t = np.sum([pred == tr_lab]) / tr_lab.shape[1]
    print('\nTrain accuracy using softmax = {}\n'.format(acc_t))

    sio.savemat('E:\exp_result\SA\pred_' + str(i), {'te_pred': pred, 'te_lab': te_lab, 'te_acc': acc, 'tr_acc': acc_t})


"""
# 使用输出分类概率
param['objective'] = 'multi:softprob'
bst = xgb.train(param, dtrain, num_round, watchlist)
# 测试预测
pred_prob = bst.predict(dtest).reshape(te_lab.shape[1], 26)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != te_lab) / te_lab.shape[1]
acc = np.sum([pred_label == te_lab]) / te_lab.shape[1]
print('\nTest error using softprob = {}\n'.format(error_rate))
print('\nTest accuracy using softprob = {}\n'.format(acc))
pred_prob = bst.predict(dtrain).reshape(tr_lab.shape[1], 13)
pred_label = np.argmax(pred_prob, axis=1)
acc = np.sum([pred_label == tr_lab]) / tr_lab.shape[1]
print('\nTrain accuracy using softmax = {}\n'.format(acc))


# 调参
from xgboost.sklearn import XGBClassifier"""

