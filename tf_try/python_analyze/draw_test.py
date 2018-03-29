# coding: utf-8
"""
Draw the ground turth label image and the test result image.
"""
from PIL import Image
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# RGB color
colors = [[0, 0, 0],
          [0, 0, 139],
          [139, 0, 139],
          [0,255,0],
          [255,255,0],
          [0,139,0],
          [0,139,139],
          [125,38,205],
          [255,69,0],
          [238,232,170],
          [220,220,220],
          [187,255,255],
          [205,149,12],
          [50,205,50],
          [199,21,133],
          [0,0,139],
          [99, 184, 255],
          [255, 255, 255],
          [205, 55, 0],
          [0, 255, 127]
          ]

def draw_picture(file_name, image_name):
    """
    Draw the HSI as a color image.
    """
    data_gt = sio.loadmat(file_name)['DataSet']
    image_data = data_gt[:, :, 0:2]
    plt.imshow(image_data)

def draw_groundturth(file_name, image_name):
    """
    Draw ground turth label image.

    Args:
        file_name: Label file path.
        image_name: Image saved name.

    Return:
        image_len: Image's length.
        image_wid: Image's width.
    """
    #label_gt = sio.loadmat(file_name)['ClsID']
    label_gt = sio.loadmat(file_name)['indian_pines_gt']
    x, y = label_gt.shape
    im = Image.new("RGB", (x, y))
    for i in range(x):
        for j in range(y):
            # axis and corresponding RGB value
            im.putpixel((i, j), (colors[label_gt[i, j]][0], colors[label_gt[i, j]][1], colors[label_gt[i, j]][2]))

    im.save(image_name + "gt_all.jpeg", "JPEG")
    image_len, image_wid = x, y
    print('Draw ground turth done.')
    return image_len, image_wid

def draw_test(pos_dir, label_dir, image_len, image_wid, image_name):
    """
    Draw classification test result image.

    Args:
        pos_dir: Data file path, data predicted label's position.
        label_dir: Data file path, data predicted label and original label.
        image_name: Saved image name.
        image_len, image_wid: The image's length and width.
    """
    data = sio.loadmat(pos_dir)
    pos = data['test_pos']
    data = sio.loadmat(label_dir)
    label = np.transpose(data['test_label'])
    pred = np.transpose(data['test_prediction'])
    #label = np.transpose(data['te_lab'])
    #pred = np.transpose(data['te_pred'])
    test_num = len(label)

    im = Image.new("RGB", (image_len, image_wid))

    for i in range(test_num):
        im.putpixel(pos[i], (colors[label[i][0] + 1][0], colors[label[i][0] + 1][1], colors[label[i][0] + 1][2]))
    for i in range(image_len):
        for j in range(image_wid):
            if im.getpixel((i, j)) == (0, 0, 0):
                im.putpixel((i, j), (colors[0][0], colors[0][1], colors[0][2]))
    im.save(image_name + "_gt.jpeg", "JPEG")
    print('Done.')

    for i in range(test_num):
        im.putpixel(pos[i], (colors[int(pred[i][0] + 1)][0], colors[int(pred[i][0] + 1)][1], colors[int(pred[i][0] + 1)][2]))
    for i in range(image_len):
        for j in range(image_wid):
            if im.getpixel((i, j)) == (0, 0, 0):
                im.putpixel((i, j), (colors[0][0], colors[0][1], colors[0][2]))
    im.save(image_name + "_pred.jpeg", "JPEG")
    print('Perfect.')


if __name__ == '__main__':
    # load data
    ksc_posdir = 'F:\data_pos\KSCdata.mat'
    ksc_labeldir = 'E:\exp_result\exp_result\KSC\pred_2.mat'
    ip_posdir = 'F:\data_pos\IPdata.mat'
    ip_labeldir = 'E:\exp_result\IP\exp_12\data.mat'
    """pu_posdir = 'F:\data_pos\PUdata.mat'
    pu_labeldir = 'E:\exp_result\exp_result\PU\pred_1.mat'
    sa_posdir = 'F:\data_pos\SAdata.mat'
    sa_labeldir = 'E:\exp_result\exp_result\SA\pred_3.mat'
    ksc_label_dir = 'F:\hsi_data\KennedySpaceCenter(KSC)\KSCGt.mat'"""
    ip_label_dir = 'F:\hsi_data\Indian Pine\Indian_pines_gt.mat'
    """pu_label_dir = 'F:\hsi_data\Pavia University scene\PUGt.mat'
    sa_label_dir = 'F:\hsi_data\Salinas scene\SAGt.mat'
    # draw ground-turth and test
    x, y = draw_groundturth(ksc_label_dir, 'KSC')
    draw_test(ksc_posdir, ksc_labeldir, x, y, 'KSC')
    x, y = draw_groundturth(pu_label_dir, 'PU')
    draw_test(pu_posdir, pu_labeldir, x, y, 'PU')
    x, y = draw_groundturth(sa_label_dir, 'SA')
    draw_test(sa_posdir, sa_labeldir, x, y, 'SA')"""
    x, y = draw_groundturth(ip_label_dir, 'IP')
    draw_test(ip_posdir, ip_labeldir, x, y, 'IP')
