# coding： utf-8
import scipy.io as sio
import xlwt

def write_xls(col_name, data, root, xls_name):
    file = xlwt.Workbook(encoding='utf-8')
    table = file.add_sheet(xls_name)
    for i in range(len(col_name)):
        for j in range(len(data[0])):
            print(data[i][j])
            table.write(i, j, data[i][j])

    file.save(root + xls_name + '_test.xls')
    print('Write done.')


ksc_dir = '/media/luo/result/hsi_transfer/ksc/results/'

ratio = [20, 18, 15, 12, 10, 8, 5, 3, 2, 1]
acc = []
for i, r in enumerate(ratio):
    S = sio.loadmat(ksc_dir + 'ratio_test/0320_' + str(r) + '/original_data.mat')
    T_true = sio.loadmat(ksc_dir + 'ratio_test/0320_true_' + str(r) + '/transfer_data.mat')
    T_false = sio.loadmat(ksc_dir + 'ratio_test/0320_false_' + str(r) + '/transfer_data.mat')
    acc_each = [max(S['train_acc'][0]), max(S['source_test_acc'][0]), max(S['target_test_acc'][0]),
                max(T_true['train_acc'][0]), max(T_true['source_test_acc'][0]), max(T_true['target_test_acc'][0]),
                max(T_false['train_acc'][0]), max(T_false['source_test_acc'][0]), max(T_false['target_test_acc'][0])]
    acc.append(acc_each)
    print('load different ratio acc.')
write_xls(ratio, acc, ksc_dir, 'diff_ratio')


lr = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
acc = []
for i, l in enumerate(lr):
    S = sio.loadmat(ksc_dir + 'lr_test/0320_' + str(l) + '/original_data.mat')
    T_true = sio.loadmat(ksc_dir + 'lr_test/0320_true_' + str(l) + '/transfer_data.mat')
    T_false = sio.loadmat(ksc_dir + 'lr_test/0320_false_' + str(l) + '/transfer_data.mat')
    acc_each = [max(S['train_acc'][0]), max(S['source_test_acc'][0]), max(S['target_test_acc'][0]),
                max(T_true['train_acc'][0]), max(T_true['source_test_acc'][0]), max(T_true['target_test_acc'][0]),
                max(T_false['train_acc'][0]), max(T_false['source_test_acc'][0]), max(T_false['target_test_acc'][0])]
    acc.append(acc_each)
    print('load different learning rate acc.')
write_xls(lr, acc, ksc_dir, 'diff_lr')
