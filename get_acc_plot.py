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
acc_file_path1 = './trained_models/' + 'V2' #model1
acc_file_path2 = './trained_models/' + 'V2_m' #model2

def get_acc_list(path):
    for root,dirs,files in os.walk(path):
        pass

    l = sorted(files, key=lambda x:int(x[:-13]))

    acc = []
    for x in l:
        acc.append(float(x[-12:-4]))

    return acc





x = [ i for i in range(1,101)]
y1 = get_acc_list(acc_file_path1)
y2 = get_acc_list(acc_file_path2)
#开始画图

# plt.title('Result Analysis')
plt.plot(x, y1, color='blue', label=model1,linewidth=1.0)
plt.plot(x, y2, color='orange', label=model2,linewidth=1.0)
plt.grid(linestyle='-.')
plt.legend() # 显示图例

plt.xlabel('epoch')
plt.ylabel('accuracy')
# plt.show()
plt.savefig('./acc1.png')