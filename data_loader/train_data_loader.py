
from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import os, random

random.seed(1)

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

    def __init__(self, data_root, transforms=None, train=True):
        self.train = train
        imgs_in = get_filepath(data_root)
        random.shuffle(imgs_in)
        imgs_num = len(imgs_in)

        if transforms is None and self.train:
            self.transforms = T.Compose([T.RandomHorizontalFlip(),
                                         T.RandomResizedCrop(224),
                                         T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                         T.RandomRotation(degrees=60, resample=False, expand=False),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
        else:
            self.transforms = T.Compose([T.Resize(size=(224, 224)),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
        if self.train: # 训练图片
            self.imgs = imgs_in[:int(0.7 * imgs_num)]
        else:          # 验证图片
            self.imgs = imgs_in[int(0.7 * imgs_num):]

    def __getitem__(self, index):
        img_path = self.imgs[index]

        label = int(img_path.split('/')[-2][1])
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)