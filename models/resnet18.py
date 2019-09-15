from .basic_module import BasicModule
import torch.nn as nn
from torch.nn import functional as F

from ipdb import set_trace

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1by1=False):
        # use_1by1: Whether to use a 1*1 conv 
        super(ResidualBlock, self).__init__()
        self.plain = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = None
        if use_1by1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.plain(x)
        # If we use 1 by 1 filter
        if self.conv:
            x = self.conv(x)
        return F.relu(y+x)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        # use_1by1: Whether to use a 1*1 conv 
        super(ResNet18, self).__init__()
        self.model_name = 'resnet18'

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)
        
            

    # num_layers: Number of residual blocks used in current layer
    def make_layer(self, in_channels, out_channels, num_layers=2, stride=1):
        #set_trace()
        layers = []
        for i in range(num_layers):
            if i==0:
                layers.append(ResidualBlock(in_channels, out_channels, use_1by1=True, stride=stride))
            else:
                layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        # print(x.shape)
        x = self.pre(x)
        # print('Layer 1')
        # print(x.shape)
        x = self.layer1(x)
        # print('Layer 2')
        # print(x.shape)
        x = self.layer2(x)
        # print('Layer 3')
        # print(x.shape)
        x = self.layer3(x)
        # print('Layer 4')
        # print(x.shape)
        x = self.layer4(x)
        # print('AvgPool')
        # print(x.shape)
        x = F.avg_pool2d(x, 7)
        # print('Flatten')
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('FC')
        # print(x.shape)
        return self.fc(x)
