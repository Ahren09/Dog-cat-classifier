# train.py
from getData import DCDataset
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dir = '/Users/jinyiqiao/Desktop/Git/Kaggle/dogs-vs-cats-redux-kernels-edition/'
model_dir = dir+'model'
lr = 1e-4
batch_size = 16
N = 50

def train():
    datafile = DCDataset('train', dir)
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True)

    model = Net()
    # model = model.cuda() # If CUDA is enabled (GPU must be present)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print('Dataset loaded. Length of training set is {0}'.format(len(datafile)))
    count = 0

    for img, label in dataloader:
        img, label = Variable(img), Variable(label)
        out = model(img)
        loss = criterion(out, label.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count += 1

        if count%N ==0:
            print('Frame {0}, Training Loss = {1}'.format(count*batch_size, loss/batch_size))

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_dir))

if __name__ == '__main__':
    train()