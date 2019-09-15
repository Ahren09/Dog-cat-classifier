# coding:utf8
from .basic_module import BasicModule
import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x
        if self.shortcut is not None:
            x = self.shortcut(x)
        out += residual
        return F.relu(out)


class ResNet(BasicModule):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.model_name = 'resnet'

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self.make_layer(64, 128, 3)
        self.layer2 = self.make_layer(128, 256, 2, stride=2)
        self.layer3 = self.make_layer(256, 512, 2, stride=2)
        self.layer4 = self.make_layer(512, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), # kernel size=1
            nn.BatchNorm2d(out_channels),
        )
        layers = []

        # There are block_num layers Residual Blocks stacked together.
        # The input has different number of layers from output
        layers.append(ResidualBlock(in_channels, out_channels, stride, shortcut))
        for i in range(1, block_num): # TODO: Why range(block_num-1) not working?
            layers.append(ResidualBlock(out_channels, out_channels)) # stride=1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        print("After 1")
        x = self.layer2(x)
        print("After 2")
        x = self.layer3(x)
        print("After 3")
        x = self.layer4(x)
        print("After 4")
        print(x.shape)
        x = F.avg_pool2d(x, 7, stride=1)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        return self.fc(x)


