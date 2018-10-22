#!/usr/bin/python
# _*_ coding: utf-8 _*_
# Edited by Jlx 2018_04_22
import numpy as np
import scipy.io as sio
import math
import random
#初始化数据
#       F             G             H            I              J            K            M             N
x_init = [(356, 349., 1.), (433, 300, 1.), (366, 222, 1.), (426, 205, 1.), (415, 190, 1.), (440, 170, 1.),
          (410, 443, 1.), (377, 354, 1.), (461, 325, 1.), (357, 244, 1.), (445, 235, 1.), (410, 222, 1.),
          (363, 190., 1.), (450, 190, 1.), (450, 177, 1.)]

y_init = [(120, 90, 1), (120, 175, 1), (40, 400, 1), (66, 470, 1), (87, 500, 1), (50, 600, 1), (70, 25, 1),
    (80, 90, 1), (132, 130, 1), (40, 347, 1), (130, 365, 1), (92, 405, 1), (40, 375, 1), (135, 535, 1), (140, 600, 1)]\
#测试算法的数据
# x_init = [(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)]
# #
# y_init = [(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2),(2,9,2)]
x = x_init
y = y_init


para = 100.0
for i in range(len(x_init)):#缩小倍数，防止梯度爆炸
    x[i] = (x_init[i][0]/para, x_init[i][1]/para, x_init[i][2]/para)
    y[i] = (y_init[i][0]/para, y_init[i][1]/para, y_init[i][2]/para)

#alpha = [[0.209817839, 0.0180384599, 0.603687291], [2.00529854, -1.9104164, 0.717344544], [0.00166577734, 0.000361302171, 0.219214705]]

# 学习率
a = 0.001
error_total = 0
cnt = 0
m = len(x)
#batch_size
random_num = 2
# 随机初始化参数
alpha = np.random.random((3, 3))

while cnt < 100000:
    cnt += 1
    for i in range(3):
        for j in range(random_num):
            index = random.randint(0, m - 1)
            diff = (alpha[i][0] * x[index][0] + alpha[i][1] * x[index][1] + alpha[i][2] * x[index][2]) - y[index][i]
            alpha[i][0] -= a * diff * x[index][0]/m
            alpha[i][1] -= a * diff * x[index][1]/m
            alpha[i][2] -= a * diff * x[index][2]/m  #随机梯度下降

    # 计算损失函数
    error1 = [0, 0, 0]

    for lp in range(3):
        for lp2 in range(m):
            tmp = (y[lp2][lp] - (alpha[lp][0] * x[lp2][0] + alpha[lp][1] * x[lp2][1] + alpha[lp][2] * x[lp2][2]))**2
            error1[lp] += math.sqrt(tmp)/m

    error_total = (error1[1] + error1[0])/2

    print('error_total : %f inter: %d learning rate: %f' % (error_total, cnt, a))


sio.savemat('result.mat', {'alpha': alpha, 'error': error_total})
for kk in range(m):
    result = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            result[i] += alpha[i][j] * x[kk][j]
    print(y[kk])
    print(result)
print(alpha)
print('迭代次数: %d' % cnt)

