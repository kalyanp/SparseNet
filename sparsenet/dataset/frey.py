import numpy as np
import gnumpy as gp
import scipy as scp
import scipy.misc
import cPickle
import sparsenet.nn_utils as nn
import os
import random
import scipy.io
import time
import threading
import multiprocessing
import ctypes
import pylab as plt
import h5py
import tables
from tables import *

class Frey():
    @staticmethod
    def load(backend="numpy", bias=None,raw = False):

        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/Frey/frey_rawface.mat")
        X = f['ff'].T
        X_mean=X.mean(axis=1)

        if not raw: X_std = (X.var(1)+10)**.5
        if not raw: X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        else: X = (X-X_mean[:, np.newaxis])

        # M = X.mean(axis=0)
        # X -= M

        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        if not raw and bias:
            sigma = cov(X)
            u,s,v=np.linalg.svd(sigma)
            P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+bias)))),u.T)
            X=np.dot(X,P)        

        X = X.reshape(1965,1,28,20)
        X = X[:,:,3:-5,:]    
        if backend=="numpy":
            return X
        else: 
            return nn.garray(X)