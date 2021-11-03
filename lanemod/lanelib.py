import numpy as np
import math
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from tqdm import tqdm_notebook
from sklearn.cluster import DBSCAN
from tqdm import notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image

DEVICE = torch.device('cuda')

def preX(names, dataFile='data'):
    X = np.zeros((len(names), 256, 512, 3))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/TrainData/Train/{name}')
        X[i] = np.array(img)[:,:,:3]/255
    X = np.transpose(X, (0, 3, 1, 2))
    X = torch.tensor(X, dtype=torch.float, requires_grad=True)
    return X

def preY(names, dataFile='data'):
    Y = np.zeros((len(names), 256, 512))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/MaskedData/Masked/{name}')
        Y[i] = np.array(img)/255
    Y = torch.tensor(Y, dtype=torch.float, requires_grad=True)
    return Y
    
def getNames(names, n, preshuffle=True):
    if preshuffle:
        random.shuffle(names)
        return names[:n], names[n:]
    else:
        return random.shuffle(names[:n]), names[n:]
    
def WBCE(y_true, y_pred):
    epsilon = torch.tensor(1e-7).to(DEVICE)
    return ((-1)*torch.mean(torch.multiply(torch.log(y_pred + epsilon), y_true)*20 + torch.multiply(torch.log(1-y_pred + epsilon), 1-y_true))).to(DEVICE)

@torch.no_grad()
def show_pred_image(X, Y_pred, Y):
    X = np.transpose(X.cpu().detach().numpy(), (1,2,0))
    plt.imshow(X, cmap='gray')
    plt.show()
    
    Y_pred = np.squeeze(Y_pred.cpu().detach().numpy())    
    plt.imshow(Y_pred, cmap= 'gray')
    plt.show()

    Y = np.squeeze(Y.cpu().detach().numpy())
    plt.imshow(Y, cmap = 'gray')
    plt.show()

    
class LaneDataset(Dataset):
    def __init__(self, names, train_dir, masked_dir, batch_size = 4, DEVICE = torch.device('cuda')):
        self.names = names
        self.dataset_size = len(self.names)
        self.train_dir = train_dir
        self.masked_dir = masked_dir
        self.batch_size = batch_size
        self.DEVICE = DEVICE

        self.train_names = []
        self.masked_names = []
        for i in range(self.dataset_size):
            train_data = train_dir + names[i]
            mask_data = masked_dir + names[i]
            self.train_names.append(train_data)
            self.masked_names.append(mask_data)
        
    def __len__(self):
        return int(np.floor(self.dataset_size) / self.batch_size)

    def __getitem__(self, idx):
        train_list = self.train_names[idx * self.batch_size: (idx+1) * self.batch_size]
        masked_list = self.masked_names[idx * self.batch_size: (idx+1) * self.batch_size]

        for idx, path in enumerate(zip(train_list, masked_list)):
            train_path, masked_path = path
            if idx == 0:
                train_image = read_image(train_path)
                train_image = torch.unsqueeze(train_image, 0)

                masked_image = read_image(masked_path)
                masked_image = torch.unsqueeze(masked_image, 0)
            else:
                train_temp = read_image(train_path)
                train_temp = torch.unsqueeze(train_temp, 0)
                train_image = torch.cat((train_image, train_temp), 0)

                masked_temp = read_image(masked_path)
                masked_temp = torch.unsqueeze(masked_temp, 0)
                masked_image = torch.cat((masked_image, masked_temp), 0)
        
        train_image = (train_image/255.).to(DEVICE)
        masked_image = (masked_image/255.).to(DEVICE)
        sample = {"train": train_image, "masked": masked_image}
        return sample
    
    
def Train(model, img_names, valid_names=None, epochs=1):
    print('######### Train Start #########')
    for e in range(epochs):
        print(f'######### Epoch {e + 1}/{epochs} Train Start #########')
        for idx in notebook.tqdm(range(len(train_loader))):
            sample = next(iter(train_loader))
            train, masked = sample['train'], sample['masked']
            if idx == 0:
                Y_pred = model(train)
                loss = WBCE(masked, Y_pred)

            optimizer.zero_grad()
            Y_pred = model(train)
            loss = WBCE(masked, Y_pred)
            loss.backward()
            optimizer.step()

            print('\r', f'[Train] Epoch : {e + 1}/{epochs},\tBatch : {idx+1}/{len(train_loader)},\tWBCE : {loss.item()}', end = '')
            if (idx+1)%(len(train_loader)//10) == 0:
                print()
        
        print(f'######### Epoch {e + 1}/{epochs} Valid Start #########')            
        with torch.no_grad():
            val = []
            for idx in notebook.tqdm(range(len(valid_loader))):
                sample = next(iter(valid_loader))
                train, masked = sample['train'], sample['masked']

                Y_pred = model(train)
                loss = WBCE(masked, Y_pred)
                val.append(loss.item())
                print('\r', f'[Valid] Epoch : {e + 1}/{epochs},\tBatch : {idx+1}/{len(valid_loader)},\tWBCE : {loss.item()}', end = '')
            print()
    
    
    
    
    
    
    
