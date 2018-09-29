# coding: utf-8
"""
Calculating kernels set for exact dataset according to excel results.
"""

import xlrd

base_dir = '/media/luo/result/excelresult/'
ksc_xlsx_file = base_dir + 'kernels-by-class (ksc).xlsx'
ip_xlsx_file = base_dir + 'kernels-by-class (IP).xlsx'
pu_xlsx_file = base_dir + 'kernels-by-class(PU).xlsx'
sa_xlsx_file = base_dir + 'kernels-by-class (SA).xlsx'

bands = 16 # ksc_13, ip_16, pu_9, sa_16
row_start = 28 # ksc_21,ip_26, pu_17, sa_28

workbook = xlrd.open_workbook(sa_xlsx_file)

kernels = [[] for _ in range(3)]
for i in range(3):
    kernels[i] = [[] for _ in range(bands)]


for i in range(3):
    sheet = workbook.sheet_by_index(i)
    nums = sheet.col_values(1)[row_start: row_start + bands]
    print('sheet {} has {} kernel nums {}'.format(str(i), str(len(nums)), nums))
    print(nums[0])
    for index, each in enumerate(nums):
        # print('index {} each {} type {}'.format(index, each, type(each)))
        if type(each) == str:
            if each[2: -7] != '46':
                kernels[i][index] = sheet.row_values(row_start + index)[2: ]
            else:
                kernels[i][index] = []
        else:
            if type(each) != list:
                kernels[i][index] = [each]
            else:
                kernels[i][index] = each
    print('sheet {} exact kernel {}'.format(str(i), kernels[i]))

final_kernels = set(kernels[0][0])
for i in range(3):
    for j in range(bands):
        if kernels[i][j] != []:
            final_kernels = final_kernels | set(kernels[i][j])
print('final kernels {}, {}'.format(len(final_kernels), final_kernels))

print('count---------------')
kernels_count = {}
for i in range(3):
    for j in range(bands):
        for each in kernels[i][j]:
            if each != '':
                if each > 0:
                    if each in kernels_count.keys():
                        kernels_count[each] += 1
                    else:
                        kernels_count[each] = 1
                else:
                    if -each in kernels_count.keys():
                        kernels_count[-each] -= 1
print('kernels count {} for specific {}'.format(len(kernels_count), kernels_count))
print('sort----------------')
kernels_sorted = sorted(kernels_count.items(), key=lambda item:item[1], reverse=True)
print(kernels_sorted)
selected_kernels = [int(x[0]) for x in kernels_sorted[: 20]]
print('selected kernels', selected_kernels)