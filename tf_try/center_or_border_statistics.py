# coding: utf-8
"""Do whether the sample is a center or border point statistics:
    extract single-pixel's class as sample, and extract 3 * 3 and 5 * 5 pixels classes and count the number of the center and border point.
"""

import scipy.io as sio
import numpy as np

def square3_category(labels):
    """
    Measure a pixel's class from 3 * 3 square around it.
    The shape:
     data1     data2     data3
     data4  target_data  data5
     data6     data7     data8
    If the 9 samples all belong to one class, determining a center data; else a border data.

    Args:
        label: Data label set, [rows,cols] means the pixel's class.

    Return:
        center_points: A list including numbers of classes list, every list means thenumber of the i-th class's center points.
        border_points: A list like center_points, but the meaning is the number of the border points.

    """
    rows, cols = labels.shape
    classes = np.max(labels)
    print('There are ' + str(classes) + ' classes in the data set.')
    center_points = []
    border_points = []
    for mark in range(classes):
        center_points.append([0])
        border_points.append([0])

    for r in range(rows):
        for c in range(cols):
            label = labels[r, c]
            if label != 0:
                # judgment standard, how to choose neighbors' coordinates
                lessThan = lambda x: x - 1 if x > 0 else 0
                greaterThan_row = lambda x: x + 1 if x < rows - 1 else rows - 1
                greaterThan_col = lambda x: x + 1 if x < cols - 1 else cols - 1

                data_label = []
                data_label.extend([labels[lessThan(r), lessThan(c)], labels[lessThan(r), c], labels[lessThan(r), greaterThan_col(c)],
                                  labels[r, lessThan(c)], labels[r, greaterThan_col(c)],
                                  labels[greaterThan_row(r), lessThan(c)], labels[greaterThan_row(r), c], labels[greaterThan_row(r), greaterThan_col(c)]])

                data_flag = 0
                for i in range(len(data_label)):
                    if data_label[i] != label:
                        data_flag = 1
                    else:
                        continue

                if data_flag == 0:
                    center_points[label - 1][0] += 1
                else:
                    border_points[label - 1][0] += 1

    print('3-Square done.')
    return center_points, border_points

def square5_category(labels):
    """
    Measure a pixel's class from 3 * 3 square around it.
    The shape:
     data1     data2     data3      data4     data5
     data6     data7     data8      data9     data10
     data11    data12  target_data  data13    data14
     data15    data16    data17     data18    data19
     data20    data21    data22     data23    data24
    If the 25 samples all belong to one class, determining a center data; else a border data.

    Args:
        label: Data label set, [rows,cols] means the pixel's class.

    Return:
        center_points: A list including numbers of classes list, every list means thenumber of the i-th class's center points.
        border_points: A list like center_points, but the meaning is the number of the border points.

    """
    rows, cols = labels.shape
    classes = np.max(labels)
    print('There are ' + str(classes) + ' classes in the data set.')
    center_points = []
    border_points = []
    for mark in range(classes):
        center_points.append([0])
        border_points.append([0])

    for r in range(rows):
        for c in range(cols):
            label = labels[r, c]
            if label != 0:
                # judgment standard, how to choose neighbors' coordinates
                lessThan = lambda x: x - 1 if x > 0 else 0
                greaterThan_row = lambda x: x + 1 if x < rows - 1 else rows - 1
                greaterThan_col = lambda x: x + 1 if x < cols - 1 else cols - 1
                lessThan_outer = lambda  x: x - 2 if x > 1 else 0
                greaterThan_row_outer = lambda x: x + 2 if x < rows - 2 else rows - 1
                greaterThan_col_outer = lambda x: x + 2 if x < cols - 2 else cols - 1

                data_label = []
                data_label.extend([labels[lessThan_outer(r), lessThan_outer(c)], labels[lessThan_outer(r), lessThan(c)],
                                  labels[lessThan_outer(r), c], labels[lessThan_outer(r), greaterThan_col(c)],
                                  labels[lessThan_outer(r), greaterThan_col_outer(c)], labels[lessThan(r), lessThan_outer(c)],
                                  labels[lessThan(r), lessThan(c)], labels[lessThan(r), c], labels[lessThan(r), greaterThan_col(c)],
                                  labels[lessThan(r), lessThan_outer(c)], labels[r, lessThan_outer(c)], labels[r, lessThan(c)],
                                  labels[r, greaterThan_col(c)], labels[r, greaterThan_col_outer(c)], labels[greaterThan_row(r), lessThan_outer(c)],
                                  labels[greaterThan_row(r), lessThan(c)], labels[greaterThan_row(r), c],
                                  labels[greaterThan_row(r), greaterThan_col(c)], labels[greaterThan_row(r), greaterThan_col_outer(c)],
                                  labels[greaterThan_row_outer(r), lessThan_outer(c)], labels[greaterThan_row_outer(r), lessThan(c)],
                                  labels[greaterThan_row_outer(r), c], labels[greaterThan_row_outer(r), greaterThan_col(c)],
                                  labels[greaterThan_row_outer(r), greaterThan_col_outer(c)]])

                data_flag = 0
                for i in range(len(data_label)):
                    if data_label[i] != label:
                        data_flag = 1
                    else:
                        continue

                if data_flag == 0:
                    center_points[label - 1][0] += 1
                else:
                    border_points[label - 1][0] += 1

    print('5-Square done.')
    return center_points, border_points

if __name__ == '__main__':
    root = 'D:/hsi/dataset/'
    # data = sio.loadmat(root + 'Kennedy Space Center (KSC)/KSCData.mat')
    label = sio.loadmat(root + 'Kennedy Space Center (KSC)/KSCGt.mat')
    labels = label['ClsID']
    classes = np.max(labels)

    center_3, border_3 = square3_category(labels)
    center_5, border_5 = square5_category(labels)
    
    count_3 = 0
    count_5 = 0
    for i in range(classes):
        print('3 * 3 :' + str(i) + ' class has ' + str(center_3[i][0]) + ' center points and ' + str(border_3[i][0]) + ' border points.')
        print('5 * 5 :' + str(i) + ' class has ' + str(center_5[i][0]) + ' center points and ' + str(border_5[i][0]) + ' border points.')
        print('There are ' + str(center_3[i][0] + border_3[i][0]) + ' in ' + str(i) + ' class of 3-Square.')
        print('There are ' + str(center_5[i][0] + border_5[i][0]) + ' in ' + str(i) + ' class of 5-Square.')
        count_3 = count_3 + center_3[i][0] + border_3[i][0]
        count_5 = count_5 + center_5[i][0] + border_5[i][0]

    print('There are ' + str(count_3) + ' pixcels in 3-Square.')
    print('There are ' + str(count_5) + ' pixcels in 5-Square.')
