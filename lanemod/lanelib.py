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

    
    
DEVICE = torch.device('cuda')


def Train(model, img_names, valid_names=None, epochs=1, chunk_size=500, batch_size=5, optimizer=None):
    if optimizer is None:
        learning_rate = 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_chunks = len(img_names)//chunk_size
    if len(img_names)%chunk_size:
        num_chunks += 1
    
    total_batch_num = len(img_names)//batch_size + len(img_names)%batch_size

    if valid_names is not None:
        num_chunks_val = len(valid_names)//chunk_size
        if len(valid_names)%chunk_size:
            num_chunks_val += 1
    
    model.train(True)
    for e in range(epochs):
        print(f'##### Epoch {e}/{epochs} Train Start #####')
        for chunk in notebook.tqdm(range(num_chunks)):
            names = img_names[chunk_size*chunk: chunk_size*(chunk+1)]
            X_chunk = preX(names)
            Y_chunk = preY(names)

            num_batchs = chunk_size//batch_size
            if chunk_size%batch_size:
                num_batchs += 1
            
            for batch in notebook.tqdm(range(num_batchs)):
                X = X_chunk[batch_size*batch: batch_size*(batch + 1)].to(DEVICE)
                Y = Y_chunk[batch_size*batch: batch_size*(batch + 1)].to(DEVICE)

                optimizer.zero_grad()
                Y_pred = model(X)
                loss = WBCE(Y, Y_pred)
                loss.backward()
                optimizer.step()
                print('\r', f'[Train] Current Epoch : {e + 1}/{epochs}, Current Batch : {chunk*num_batchs+batch+1}/{total_batch_num}, WBCE : {loss.cpu()}', end = '')
                if (chunk*num_batchs+batch+1) % (total_batch_num//20) == 0:
                    print()
                
                break
            break

        # Validation
        with torch.no_grad():
            if valid_names is not None:
                print(f"##### Epoch {e}/{epochs} Validation Start #####")
                val = []
                for chunk_val in notebook.tqdm(range(num_chunks_val)):
                    names = valid_names[batch_size*chunk_val: batch_size*(chunk_val + 1)]
                    X_chunk = preX(names)
                    Y_chunk = preY(names)

                    num_batchs = chunk_size//batch_size
                    if chunk_size%batch_size:
                        num_batchs += 1

                    for batch in notebook.tqdm(range(num_batchs)):
                        X = X_chunk[batch_size*batch: batch_size*(batch + 1)].to(DEVICE)
                        Y = Y_chunk[batch_size*batch: batch_size*(batch + 1)].to(DEVICE)
                    
                        Y_pred = model(X)
                        loss = WBCE(Y, Y_pred)
                        val.append(loss.item())
                        
                    
                print(f'[Validation] Current Epoch : {e}/{epochs}, WBCE : {sum(val)/len(val)}')
                show_pred_image(X[0],Y_pred[0], Y[0])
    
    
    
    
    
    
    
