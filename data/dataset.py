
import os
from torch.utils import data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T

# Define dataset

class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, mode='train'):
        self.mode = mode

        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # train: data/train/cat.10004.jpg
        # test: data/test/12906.jpg
        if self.mode == 'test':
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        # img_nums: Number of images in dataset
        img_nums = len(imgs)

        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        if self.mode = 'test':
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

        else:
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomSizedCrop(224), # TODO: Remove these two steps?
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

    # Return data of one picture
    def __getitem__(self, index):
        img_path = self.imgs[index]

        # train: 
        # test: "data/test/12.jpg" returns "12"
        if self.mode == 'test':
            label = int(self.imgs[index].split('/')[2].split('.')[0])
        else:
            label = int(self.imgs)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label


    def __len__(self):
        return len(self.imgs)



