dataset = ["ksc", "ip", "sa", "pu"]
root = '/media/luo/result/hsi_ppr_result/'
for i= 1:4
    path = root + data + '/' + data + '_1008/data.mat'
    load(path)
    plot_acc(data)
    path = root + data + '/' + data + '_1008_cube/data.mat'
    load(path)
    plot_acc([data, '_cube'])
end