import os
import torch
import torch.nn as nn
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import ipdb
# from torchnet import meter
# from utils.visualize import Visualizer
# from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-4
LR_DECAY = 0.5
MAX_EPOCHS = 100
ROOT = '/Users/jinyiqiao/Desktop/Git/DL/dogs-vs-cats-redux-kernels-edition/'
PRINT_EVERY = 2
SAVE_EVERY = 1000
PRETRAINED = True # Whether or not to use pretrained networks
MODEL_NAME = 'resnet'
def train():
    # ipdb.set_trace()

    # Load training data
    train_data = DogCat(ROOT, mode='train')
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Load validation data
    val_data = DogCat(ROOT, mode='val')
    
    val_data_len = len(val_data)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    net = models.ResNet18()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Calculates training loss
    criterion = nn.CrossEntropyLoss() 

    plt.title('Training Loss')

    max_epoch = -1
    
    # Load pretrained network with greatest number of training epochs
    pretrained_root = ROOT+'models/pretrained_'+net.model_name+'/'
    if PRETRAINED and os.path.exists(pretrained_root):
        # model_list = [os.path.join(pretrained_root, model) for model in os.listdir(pretrained_root)]
        file_list = [file for file in os.listdir(pretrained_root) if file.endswith('.pth')]
        if file_list != []:
            index_list = [int(file.split('.pth')[0].split('_')[1]) for file in file_list]
            max_epoch = max(index_list)
            print('Using mode:', 'model_%s.pth'%max_epoch)
            net.load_state_dict(torch.load(pretrained_root+'model_%s.pth'%max_epoch))
        else:
            print('Train from Epoch 0')


    # Store steps and loss for plotting
    # loss_list = []
    plt.ion()

    previous_loss = 1e3


    # TODO: start from some step
    for epoch in range(MAX_EPOCHS):
        if epoch < max_epoch+1:
            continue

        loss_list = []
        for step, (input, label) in enumerate(train_dataloader):
            print("step", step)
            optimizer.zero_grad()
            score = net(input)
            loss = criterion(score, label) # loss is a torch tensor
            loss.backward()
            optimizer.step()

            
            # print(len(step_list))
            # print(len(loss_list))
            
            
            # Plot training loss
            if step%PRINT_EVERY == 0:
                print('Epoch', epoch, 'Step: ', step, ' | Loss: ', loss.data.numpy())
                loss_list.append(loss.data.numpy())
                plt.plot(range(int(step/PRINT_EVERY+1)), loss_list, 'r-')
                plt.xlabel('Number of iterations')
                plt.ylabel('Loss')
                plt.ylim(0, 1)
                plt.draw()
                plt.pause(0.01)
                # plt.ioff()
                # plt.show()

            # TODO: plot loss and accuracy in parallel
        val_accuracy = val(net, val_dataloader, val_data_len)
        print("=====")
        print('Epoch', epoch, 'Step: ', step, "Accuracy: %s" % val_accuracy)
        print("=====")
        # Save the model after finishing an epoch
        torch.save(net.state_dict(), ROOT+'models/pretrained/'+'model_%s.pth' % epoch)

        # loss_value: float
        loss_value = float(loss.data.numpy())
        if loss_value > previous_loss:
            lr = LR * LR_DECAY
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss
            
            # Save networks

def val(net, dataloader, val_data_len):
    net.mode = 'val'
    net.eval()
    score=0

    for step, (input, labels) in enumerate(dataloader):
        out = net(input)
        out_labels=torch.max(out, dim=1).indices
        # print(out)
        # print(out_labels)
        # print(labels)
        # print((out_labels==labels).sum())
        
        score += (out_labels == labels).sum()

    net.train()
    accuracy = 100. * (float(score)/val_data_len)
    return accuracy


def test():
    pass
if __name__ == '__main__':
    train()


