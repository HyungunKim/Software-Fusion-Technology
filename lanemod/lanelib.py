import numpy as np
import math
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from tqdm import tqdm_notebook
import tensorflow as tf
from sklearn.cluster import DBSCAN

def preX(names, dataFile='data'):
    X = np.zeros((len(names), 256, 512, 3))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/TrainData/Train/{name}')
        X[i] = np.array(img)[:,:,:3]/255
    return X

def preY(names, dataFile='data'):
    Y = np.zeros((len(names), 256, 512))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/MaskedData/Masked/{name}')
        y = np.array(img)/255
        Y[i] = y
    return Y
    
def getNames(names, n):
    random.shuffle(names)
    return names[:n], names[n:]
    
    
def WBCE(y_true, y_pred):
    epsilon = 1e-7
    return (-1)*tf.math.reduce_mean(tf.math.multiply(tf.math.log(y_pred + epsilon), y_true)*20 + tf.math.multiply(tf.math.log(1-y_pred + epsilon), 1-y_true))
    
    
    
    
    
    
    
    
    
    
    
    
