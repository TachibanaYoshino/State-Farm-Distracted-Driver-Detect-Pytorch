# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import time
import os, cv2
import torch as t
from thop import profile

from data_loader import test_data_loader
from config import cfg

class_dict = {  '0': 'safe driving',
                '1': 'texting - right',
                '2': 'talking on the phone-right',
                '3': 'texting - left',
                '4': 'talking on the phone-left',
                '5': 'operating the radio',
                '6': 'drinking',
                '7': 'reaching behind',
                '8': 'hair and makeup',
                '9': 'talking to passenger',
            }

def get_class(index):
    assert index>=0 and index <=9
    return class_dict[str(index)]



"""
===============================================================================
 test
===============================================================================
"""
if __name__ == '__main__':
    # model_path = './trained_models/V2/'+'99_0.948312.pkl'
    model_path = './trained_models/V2_m/'+'97_0.965492.pkl'

    '''加载模型'''
    model = t.load(model_path)
    print('loading model : %s' % model_path)
    model.eval().cuda()

    flops, params = profile(model, input_size=(1, 3, 224, 224))

    '''加载数据'''
    test_data_path = cfg.TEST.test_data_path
    test_data = test_data_loader.DriverDataset(test_data_path)
    test_dataloader = DataLoader(dataset=test_data, shuffle=False, batch_size=cfg.TEST.BATCH_SIZE, num_workers=4)

    '''test'''

    j = 0

    for (data_x, data_path) in test_dataloader:
        j += 1
        if j == 1001:
            break
        start_t = time.time()
        # pdb.set_trace()
        input = data_x


        if cfg.TEST.use_gpu:
            input = input.cuda()
        output = model(input)
        m = t.nn.Softmax(dim=1)
        output = m(output)

        value, index = t.max(output, 1)

        value, index = value.cpu().detach().numpy()[0], index.cpu().detach().numpy()[0]
        if index == 2 or index==4:  #'标签太长要换行显示'
            text = get_class(index) + ': \n' + '%s' % (str(value))
        else:
            text = get_class(index) + ': ' + '%s' % (str(value))
        # print(text)


        '''save test images'''
        # image = input.mul(255).byte().cpu()
        # image = image.squeeze(0).permute(1, 2, 0)
        # image = image.numpy()
        image = cv2.imread(data_path[0])
        if index == 0:
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        elif index == 2 or index==4: #'标签太长要换行显示'
            for i, txt in enumerate(text.split('\n')):
                y = 50 + i * 60
                cv2.putText(image, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # test_out_path = './test_output/V2/'
        test_out_path = './test_output/V2_m/'
        if not os.path.isdir(test_out_path):
            os.makedirs(test_out_path)

        cv2.imwrite(test_out_path+'img_'+str(j)+'.jpg', image)

        end_t = time.time()- start_t
        print('testing   %d    picture ------------------ time: %f s' %(j,end_t))

        if j==499:
            t1 = time.time()
        if j == 1000:
            t2 =time.time()
            print('mean time:  %f s' % ((t2-t1)/500))

    print('Gflops, params : ', flops/(pow(10.0, 9)), params)







