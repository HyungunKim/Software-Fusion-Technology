import numpy as np
import math
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from tqdm import tqdm_notebook
import tensorflow as tf
from sklearn.cluster import DBSCAN

def preX(names):
    X = np.zeros((len(names), 256, 512, 3))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/new_rgb_output/{name}')
        X[i] = np.array(img)[:,:,:3]/255
    return X

def preY(names):
    Y = np.zeros((len(names), 256, 512))
    for i, name in enumerate(names):
        img = Image.open(f'{dataFile}/new_sem_output/{name}')
        y = np.array(img)[:,:,:3]
        yb = np.ones((256,512))
        yb[y[:,:,0] != 157] = 0
        yb[y[:,:,1] != 234] = 0
        yb[y[:,:,2] != 50] = 0
        Y[i] = yb
    return Y
    
def getNames(names, n):
    random.shuffle(names)
    return names[:n], names[n:]
    
    
def WBCE(y_true, y_pred):
    epsilon = 1e-7
    return (-1)*tf.math.reduce_mean(tf.math.multiply(tf.math.log(y_pred + epsilon), y_true)*20 + tf.math.multiply(tf.math.log(1-y_pred + epsilon), 1-y_true))
    
    
    
    
    
    
    
    
    
    
    
    
