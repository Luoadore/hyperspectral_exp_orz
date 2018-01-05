# coding: utf-8
"""
Draw the ground turth label image and the test result image.
"""
from PIL import Image
import scipy.io as sio
import numpy as np

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
          [255, 255, 255],
          [99, 184, 255],
          [205, 55, 0],
          [0, 255, 127]
          ]

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
    label_gt = sio.loadmat(file_name)['ClsID']
    #label_gt = sio.loadmat(file_name)['indian_pines_gt']
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
    test_num = len(label)

    im = Image.new("RGB", (image_len, image_wid))

    for i in range(test_num):
        im.putpixel(pos[i], (colors[label[i][0]][0], colors[label[i][0]][1], colors[label[i][0]][2]))
    for i in range(image_len):
        for j in range(image_wid):
            if im.getpixel((i, j)) == (0, 0, 0):
                im.putpixel((i, j), (colors[0][0], colors[0][1], colors[0][2]))
    im.save(image_name + "_gt.jpeg", "JPEG")
    print('Done.')

    for i in range(test_num):
        im.putpixel(pos[i], (colors[pred[i][0]][0], colors[pred[i][0]][1], colors[pred[i][0]][2]))
    for i in range(image_len):
        for j in range(image_wid):
            if im.getpixel((i, j)) == (0, 0, 0):
                im.putpixel((i, j), (colors[0][0], colors[0][1], colors[0][2]))
    im.save(image_name + "_pred.jpeg", "JPEG")
    print('Perfect.')


if __name__ == '__main__':
    # load data
    posdir = 'F:\hsi_result\original\KSC\data\\1st\data8.mat'
    labeldir = 'F:\hsi_result\original\KSC\data\\1st\data8.mat'
    original_label_dir = 'F:\hsi_data\Kennedy Space Center (KSC)\KSCGt.mat'
    # draw ground-turth
    x, y = draw_groundturth(original_label_dir, 'KSC')
    draw_test(posdir, labeldir, x, y, 'KSC')
