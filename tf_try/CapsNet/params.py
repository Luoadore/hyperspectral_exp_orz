# -*- coding: utf-8 -*-


class Params:
    pass


param = Params()
param.pre ='data'
param.generations = 30001
param.output_every = 200
param.save_every = 1000
param.batch_size = 100
param.neighbors = 1
param.learning_rate = 0.005
param.m_plus = 0.9
param.m_minus = 0.1
param.down_val = 0.5
param.regular_scale = 0.01
param.vector = 16
param.channel = 8
param.length = 16
param.routing = 3


_pu = Params()
_pu.data_file = 'pu.mat'
_pu.class_number = 9
_pu.feature_number = 103
_pu.kernel1 = 24 * param.neighbors
_pu.stride1 = param.neighbors
_pu.kernel2 = 32
_pu.stride2 = 2
_pu.fc1 = 64
_pu.fc2 = 128

_pc = Params()
_pc.data_file = 'pc.mat'
_pc.class_number = 9
_pc.feature_number = 102
_pc.kernel1 = 23 * param.neighbors
_pc.stride1 = param.neighbors
_pc.kernel2 = 32
_pc.stride2 = 2
_pc.fc1 = 64
_pc.fc2 = 128

_ksc = Params()
_ksc.data_file = 'ksc.mat'
_ksc.class_number = 13
_ksc.feature_number = 176
_ksc.kernel1 = 24 * param.neighbors
_ksc.stride1 = param.neighbors
_ksc.kernel2 = 57
_ksc.stride2 = 4
_ksc.fc1 = 64
_ksc.fc2 = 128

_ip = Params()
_ip.data_file = 'ip.mat'
_ip.class_number = 16
_ip.feature_number = 200
_ip.kernel1 = 24 * param.neighbors
_ip.stride1 = param.neighbors
_ip.kernel2 = 57
_ip.stride2 = 5
_ip.fc1 = 64
_ip.fc2 = 256

_sa = Params()
_sa.data_file = 'ip.mat'
_sa.class_number = 16
_sa.feature_number = 204
_sa.kernel1 = 24 * param.neighbors
_sa.stride1 = param.neighbors
_sa.kernel2 = 61
_sa.stride2 = 5
_sa.fc1 = 64
_sa.fc2 = 256


param.data = {'pu': _pu, 'pc': _pc, 'ksc': _ksc, 'ip': _ip, 'sa': _sa}
param.hsi = _pu
