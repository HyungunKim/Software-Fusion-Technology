import numpy as np
import math
import tensorflow as tf
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def getIndicies_tf(A):
    zero = tf.constant(0, dtype=tf.float32)
    A = tf.cast(A + 0.5, tf.int32)
    where = tf.not_equal(A,zero)
    indicies = tf.where(where)
    return indicies.numpy()

def getIndicies_np(A):
    return np.transpose(np.nonzero(np.where(A[:,:,0]>0.8, 1, 0)))

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
