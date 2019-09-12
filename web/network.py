# network.py
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

# TODO: Forward pass using nn.Sequential

# The new network must inherit its parent class - nn.Module 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(50*50*16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x) # 16*16*200*200
        x = F.relu(x) # 16*16*200*200
        x = F.max_pool2d(x, kernel_size=2) # 16*16*100*100

        x = self.conv2(x) # 16*16*100*100
        x = F.relu(x) # 16*16*100*100
        x = F.max_pool2d(x, kernel_size=2) # 16*16*50*50

        x = x.view(x.size()[0], -1) # 16*40000
        x = F.relu(self.fc1(x)) # 16*128
        x = F.relu(self.fc2(x)) # 16*64
        x = self.fc3(x) # 2

        return F.softmax(x, dim=1)
