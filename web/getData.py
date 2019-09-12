import os
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms

# Default size of input image
H = 200
W = 200

# Transform images to Pytorch Tensors
data_transform = transforms.Compose([
    transforms.ToTensor()
])

class DCDataset(data.Dataset):
    def __init__(self, mode, dir):
        # initialize variables: list of images, size of dataset
        self.mode = mode
        self.list_img = [] # Stores the dir of all images
        self.list_labels = [] # Stores the labels of all images
        self.dataset_size = 0
        self.transform = data_transform

        if self.mode == 'train':
            dir = dir + '/train/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.dataset_size += 1
                name = file.split(sep='.')
                if name[0] == 'cat':
                    self.list_labels.append(0)
                elif name[0] == 'dog':
                    self.list_labels.append(1)
                else:
                    print("Error: Incorrect file name")


        elif self.mode == 'test':
            dir = dir + '/test/'
            for file in os.listdir(dir):
                self.list_img.append(dir + file)
                self.dataset_size += 1
                self.list_labels.append(2)

        else:
            return print("Undefined Operation")

    def __getitem__(self, item):
        img = Image.open(self.list_img[item]).resize((H,W))
        img = np.array(img)[:,:,:3]
        
        if self.mode == 'train':
            label = self.list_labels[item]
            return self.transform(img), torch.LongTensor([label])

        elif self.mode == 'test':
            return self.transform(img)
        else:
            print("Error")


    def __len__(self):
        return self.dataset_size
            