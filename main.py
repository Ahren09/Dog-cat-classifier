import os
import torch
import torch.nn as nn
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from torchnet import meter
# from utils.visualize import Visualizer
# from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-2
MAX_EPOCHS = 3
ROOT = '/Users/jinyiqiao/Desktop/Git/DL/dogs-vs-cats-redux-kernels-edition/'
PRINT_EVERY = 2
SAVE_EVERY = 1000
PRETRAINED = False

def train():
    train_data = DogCat(ROOT, mode='train')
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    net = models.SimpleNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Calculates training loss
    criterion = nn.CrossEntropyLoss() 

    plt.title('Training Loss')

    '''
    # Load pretrained net
    net = None
    
    # Load pretrained network with greatest number of steps
    pretrained_root = ROOT+'models/pretrained/'
    if os.path.exists(pretrained_root):
        # model_list = [os.path.join(pretrained_root, model) for model in os.listdir(pretrained_root)]
        file_list = [file for file in os.listdir(pretrained_root) if file.endswith('.pth')]
        
        index_list = [int(file.split('.pth')[0].split('_')[1]) for file in file_list]
        max_index = max(index_list)
        print('model_%s.pth'%max_index)
        net = torch.load('model_%s.pth'%max_index)
    '''


    # For plotting
    step_list, loss_list = [], []

    plt.ion()

    # TODO: start from some step
    for epoch in range(MAX_EPOCHS):
        for step, (input, label) in enumerate(dataloader):
            optimizer.zero_grad()
            score = net(input)
            loss = criterion(score, label) # loss is a torch tensor
            loss.backward()
            optimizer.step()

            step_list.append(step)
            loss_list.append(loss.data.numpy())
            # print(len(step_list))
            # print(len(loss_list))
            
            
            # Plot training loss
            if step%PRINT_EVERY == 0:
                print('Epoch', epoch, 'Step: ', step, ' | Loss: ', loss.data.numpy())

                
                plt.plot(step_list, loss_list, 'r-')
                plt.xlabel('Number of iterations')
                plt.ylabel('Loss')
                plt.ylim(0, 1)
                plt.draw()
                plt.pause(0.01)
                # plt.ioff()
                # plt.show()
            
            # Save networks
            if step%SAVE_EVERY == 0:
                torch.save(net.state_dict(), 'model_%s.pth'%(step+1))


def test():
    pass
if __name__ == '__main__':
    train()


