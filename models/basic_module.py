import torch
import torch.nn as nn
import time


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        # Get the type of NN
        self.model_name = str(type(self))

    # TODO: model saving according to time period
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):

        torch.save(self.state_dict(), name)

        return name


# Flat: turn input into shape (batch_size, dim_len)
class Flat(nn.Module):
    def __init__(self):
        super(Flat, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
