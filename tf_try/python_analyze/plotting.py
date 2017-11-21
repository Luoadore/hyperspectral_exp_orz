# coding: utf-8
"""
Plotting the loss-iterations and accuracy-iterations curve figure of training and testing.
Training data is extracted from TensorBoard and testing data is extract from the training program.
目前还没有写好，不太熟练，画图还是matlab好用。。。
"""

import matplotlib.pyplot as plt
import pandas
import numpy

root = 'F:/hsi_result/original/lossAndaccuracy_train'

def showXY(x, y, title, ylabel, curve_color, imageName = ''):
    """
    Show figure, single curve.

    Args:
        x: X axis data.
        y: Y axis data.
        title: Figure title.
        ylabel: Y axis's name.
        curve_color: Curve color.
        imageName:Image name for saving.
    """
    assert len(x) > 0 and len(y) > 0 and len(x) == len(y)
    dpi = 72.
    width = 300
    height = 300
    g = plt.figure(figsize = (width/dpi, height/dpi))
    plt.clf()
    plt.cla()
    plt.axis([0, max(x), 0, 1])
    plt.xticks(range(0, int(max(x) + 1), max(1, int(max(x) / 10.0))), size = 10)
    plt.title(title, size = 10)
    plt.xlabel('iterations', size = 10)
    plt.ylabel(ylabel, size = 10)
    plt.plot(x, y, color = curve_color)
    axis = plt.gca()
    xaxis = axis.xaxis
    xaxis.grid(False)
    yaxis = axis.yaxis
    yaxis.grid(True)

    if imageName is not '':
        plt.savefig(root + '/' + imageName, dpi = dpi)

    plt.show()

def load_data(neighbor):
    """
    Load data set from csv, and return column array.

    Args:
        neighbor: Pixel neighbor, include 1, 4, 8.

    Return:
        steps: Corresponding iterations.
        acc: Accuracy of per 100 iterations.
        loss: Loss of per 100 iterations.
    """
    acc_path = root + '/run_' + str(neighbor) + '-tag-accuracy.csv'
    loss_path = root + '/run_' + str(neighbor) + '-tag-training-loss.csv'
    accuracy_data = pandas.read_csv(acc_path)
    loss_data = pandas.read_csv(loss_path)
    steps = accuracy_data['Step']
    acc = accuracy_data['Value']
    loss = loss_data['Value']

    return steps, acc, loss

if __name__ == '__main__':
    steps, acc_1, loss_1 = load_data(1)
    _, acc_4, loss_4 = load_data(4)
    _, acc_8, loss_8 = load_data(8)
    showXY(steps, acc_1, 'Accuracy of original method net of neighbor method', 'accuracy', 'b', 'train_accVSiters.jpg')