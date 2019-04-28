
from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import os, random

def get_filepath(dir_root):
    ''''获取一个目录下所有文件的路径，并存储到List中'''
    file_paths = []
    for root, dirs, files in os.walk(dir_root):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class DriverDataset(data.Dataset):
    '''
    1 加载数据
    2 对数据进行预处理
    3 进行训练集/验证集的划分
    '''

    def __init__(self, data_root, transforms=None):

        self.imgs_in = get_filepath(data_root)

        if transforms is None:
            self.transforms = T.Compose([T.Resize(size=(224,224)),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])

    def __getitem__(self, index):
        img_path = self.imgs_in[index]

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, img_path

    def __len__(self):
        return len(self.imgs_in)