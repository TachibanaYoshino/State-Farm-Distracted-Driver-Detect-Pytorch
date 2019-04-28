# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#这里导入你自己的数据
#......
#......
#x_axix，train_pn_dis这些都是长度相同的list()
model1 = 'MobileNet-V2'
model2 = 'MobileNet-V2 + Residual mask'
acc_file_path1 = 'V2' + '_loss.txt'
acc_file_path2 = 'V2_m' + '_loss.txt'

def get_loss_list(path):
    index =[]
    losses = []
    with open(path, 'r') as f:
        while True:                  #直到读取完文件
            line = f.readline()  # 读取一行文件，包括换行符
            if not line:
                break
            line = line[:-1]    # 去掉换行符，也可以不去

            line = line.split(",")

            index.append(int(line[0].strip()))
            losses.append(float(line[1].strip()))

    return index, losses





x1,y1 = get_loss_list(acc_file_path1)
x2,y2 = get_loss_list(acc_file_path2)
assert  x1 ==x2
#开始画图

# plt.title('Result Analysis')
plt.plot(x1, y1, color='green', label=model1,linewidth=1.0)
plt.plot(x1, y2, color='red', label=model2,linewidth=1.0)
plt.grid(linestyle='-.')
plt.legend() # 显示图例

plt.xlabel('step')
plt.ylabel('loss')
# plt.show()
plt.savefig('./loss1.png')