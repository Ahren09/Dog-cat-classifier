# test.py
from getData import DCDataset
from network import Net
import numpy as np
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torch, random, csv, os
import torchvision.transforms as transforms

dir = './'
model_dir = dir + 'model/model.pth'
N_SAMPLES = 10

MODE = 'All'
DIR = '/Users/jinyiqiao/Desktop/Git/Kaggle/dogs-vs-cats-redux-kernels-edition/test/4.jpeg'

data_transform = transforms.Compose([
    transforms.ToTensor()
])

def test():
    model = Net()
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    
    if MODE == 'Single':
        img = Image.open(DIR).resize((200,200))
        img = np.array(img)[:,:,:3]
        img = data_transform(img)
        img = img.unsqueeze(0)
        #img = Variable(img)
        out = model(img)
        print("Output:")
        if out[0,0]>out[0,1]:
            print('The image is a cat')
        else:
            print('The image is a dog')

        img = Image.open(DIR)
        plt.figure('image')
        plt.imshow(img)
        plt.show()

    else:
        datafile = DCDataset('test', dir)
        print('Dataset loaded! Length of training set is {0}'.format(len(datafile)))
        
        if MODE == 'Random':
            List = random.sample(range(len(datafile)), N_SAMPLES)
            for index in List:
                img = datafile.__getitem__(index)
                img = img.unsqueeze(0)
                img = Variable(img)
                out = model(img)
                print("Output:")
                print(out)
                if out[0,0]>out[0,1]:
                    print('The image is a cat')
                else:
                    print('The image is a dog')

                img = Image.open(datafile.list_img[index])
                plt.figure('image')
                plt.imshow(img)
                plt.show()

        elif MODE == 'All':

            List = range(len(datafile))
            results = []
            with torch.no_grad():
                for index in List:
                    if index%500 == 0:
                        print("Testing", index)
                    img = datafile.__getitem__(index)
                    img = img.unsqueeze(0)
                    img = Variable(img)
                    out = model(img)
                    predicted = out.data.tolist()
                    results.extend([[index+1, predicted[0][0]]])
            
            out_csv = os.path.join(os.path.expanduser('.'), 'submission.csv')
            with open(out_csv, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                q = ("id", "label")
                writer.writerow(q)
                for x in results:
                    writer.writerow(x)

if __name__ == '__main__':
    test()