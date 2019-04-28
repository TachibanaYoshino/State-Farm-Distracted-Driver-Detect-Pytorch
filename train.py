# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:28:51 2018
@brief : 1）项目：kaggle平台上的驾驶员注意力状态检测【State Farm Distracted Driver Detection】，详见https://www.kaggle.com/c/state-farm-distracted-driver-detection
         2）项目介绍:需建立一个模型，对一张图片进行分类。共有10类：安全驾驶，右手打字，右手打电话...
         3）本部分代码主要是根据竞赛所提供的imgs.zip中train数据集，训练一个finetune的renet34的模型；
@environment: linux pytorch0.4 python3.5
@author:
"""

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch as t
from data_loader import train_data_loader
from config import cfg
import os, time

# get net
# from net_mask import resnet18
# model = resnet18.lamda()

from mobilenet import V2_m
model = V2_m.mobilenet_v2()
if cfg.TRAIN.use_gpu:
    model.cuda()

model_path = './trained_models/'+'95_0.950829.pkl'
if os.path.exists(model_path):
    model = t.load(model_path)
    print('loading model : %s' % model_path)
    start_epoch = int(model_path[18:-13]) + 1
else:
    print('Training from scratch')
    start_epoch = 0
    
"""
===============================================================================
2 定义LOSS和优化器
===============================================================================
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

"""
===============================================================================
3 定义评估函数
===============================================================================
"""
def val(model, dataloader):
    model.eval()
    acc_sum = 0
    for ii, (input, label) in enumerate(dataloader):
        val_input = input
        val_label = label
        if cfg.TRAIN.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        output = model(val_input)
        acc_batch = t.mean(t.eq(t.max(output, 1)[1], val_label).float())
        acc_sum += acc_batch
        
    acc_vali = acc_sum / (ii + 1)
    model.train()
    return acc_vali
"""
===============================================================================
4 训练
===============================================================================
"""
if __name__ == '__main__':

    '''加载数据'''
    train_data_path = cfg.TRAIN.train_data_path

    train_data = train_data_loader.DriverDataset(train_data_path, train=True)
    train_dataloader = DataLoader(dataset=train_data,shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)

    vali_data = train_data_loader.DriverDataset(train_data_path, train=False)
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)
    
    print(model)
    
    '''训练'''
    loss_print = []
    j = 0

    f = open('./V2_m_loss.txt','w')
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
        st = time.time()
        for (data_x, label) in train_dataloader:
            j += 1
            optimizer.zero_grad()
            #pdb.set_trace()
            input = data_x
            label = label
            if cfg.TRAIN.use_gpu:
                input = input.cuda()
                label = label.cuda()
        
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_print.append(loss)
            
            '''print loss'''
            if j % cfg.TRAIN.frequency_print == 0:
                loss_mean = t.mean(t.Tensor(loss_print))
                loss_print = []
                print('第 %d epoch, step : %d' % (epoch, j), 'train_loss: %f'%loss_mean)
                f.write(str(j)+','+'%f'%loss_mean+'\n')

        print("epoch time : %f s" % (time.time()-st))
        '''可视化模型在验证集上的准确率'''
        acc_vali = val(model, vali_dataloader)
        print('第 %d epoch, acc_vali : %f' % (epoch, acc_vali))

        '''每epoch,保存已经训练的模型'''
        trainedmodel_path = './trained_models/V2_m/'
        if not os.path.isdir(trainedmodel_path):
            os.makedirs(trainedmodel_path)
        t.save(model, trainedmodel_path + '%d'%epoch + '_' + '%f'%acc_vali + '.pkl')
    f.close()

        
