# coding: utf-8
"""
Plotting loss and acc.
"""
import scipy.io as sio
import matplotlib.pyplot as plt
ROOT = '/media/luo/result/hsi_ppr_result/'

def load_data(filename):
    data = sio.loadmat(filename)
    test_step = data['test_step'][0]
    train_acc = data['train_acc'][0]
    test_acc = data['test_acc'][0]
    test_loss = data['test_loss'][0]
    return test_step, train_acc, test_acc, test_loss

def plot_acc_loss(dataset):

    # load data
    path_pixel = ROOT + dataset + '/' + dataset + '_1118_1/data.mat'
    path_cube = ROOT + dataset + '/' + dataset + '_1118_1_cube/data.mat'
    test_step_pixel, train_acc_pixel, test_acc_pixel, test_loss_pixel = load_data(path_pixel)
    test_step_cube, train_acc_cube, test_acc_cube, test_loss_cube = load_data(path_cube)

    plt.clf()
    plt.cla()
    # accuracy
    plt.plot(test_step_pixel, train_acc_pixel, label='train-pixel')
    plt.plot(test_step_pixel, test_acc_pixel, label='test-pixel')
    plt.plot(test_step_cube, train_acc_cube, label='train_cube')
    plt.plot(test_step_cube, test_acc_cube, label='test_cube')


    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy of PPR in ' + dataset)
    plt.savefig(ROOT + '/figs/accuracy-' + dataset)

    print('Draw accuracy fig:', dataset)

    plt.clf()
    plt.cla()
    # loss
    plt.plot(test_step_pixel, test_loss_pixel, label='pixel')
    plt.plot(test_step_cube, test_loss_cube, label='cube')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss of PPR in ' + dataset)
    plt.savefig(ROOT + '/figs/loss-' + dataset)

    print('Draw loss fig:', dataset, max(test_acc_pixel), max(test_acc_cube))
    return max(test_acc_pixel), max(test_acc_cube)

def multi_exp_result(dataset, exp_times, time):
    test_acc = []
    for i in range(exp_times):
        # load data
        path_cube = ROOT + dataset + '/' + dataset + '_' + time + '_' + str(i) + '_cube/data.mat'
        path_finetune = ROOT + dataset + '/' + dataset + '_' + time + '_finetune_' + str(i) + '_cube/data.mat'
        test_step_cube, train_acc_cube, test_acc_cube, test_loss_cube = load_data(path_cube)
        test_step_finetune, train_acc_finetune, test_acc_finetune, test_loss_finetune = load_data(path_finetune)
        test_acc.append([max(test_acc_cube), max(test_acc_finetune)])
        print('The ' + str(i) + ' exp acc.')
    return test_acc

def record_max(data):
    with open(ROOT + '/figs/result.txt', 'wt') as f:
        for each in data:
            f.write(str(each[0]) + '\t' + str(each[1]) + '\n')
    print('Write down.')

if __name__ == '__main__':
    dataset = ['ksc', 'ip', 'pu', 'sa']
    max_acc = []
    """
    for each in dataset:
        max_pixel, max_cube = plot_acc_loss(each)
        max_acc.append([max_pixel, max_cube])
    """
    for each in dataset:
        test_acc = multi_exp_result(each, 4, '0314')
        print(each + ':')
        print(test_acc)
        max_acc.extend(test_acc)
    record_max(max_acc)