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

class TinyImages():
    @staticmethod
    class data_provider_tiny_images:
        def __init__(self,address,train_range=[0,10],mini_batch = 100):
            self.train_range = train_range
            self.address = address
            self.data_file = open(self.address, "rb")        
            # f=np.load(work_address()+"/save/MP.npz"); 
            self.train_range_id = train_range[0]              
            f=np.load(work_address()+"/Dataset/CIFAR10/cifar_whitened.npz"); 
            self.X_cifar=f['X'];self.T=f['T'];self.X_test_cifar=f['X_test'];self.T_test=f['T_test'];
            self.T_train_labels=f['T_train_labels'];self.T_labels=f['T_labels']
            self.M = f['M']
            self.P = f['P'] 

        def train(self):
            train_range_id_temp = self.train_range_id      
            self.train_range_id = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]          
            # t = time.time()
            X = np.zeros((10000,3072))
            for i in range(10000):
                x = self.sliceToBin(train_range_id_temp*10000+i)
            #     nn.show_images(np.swapaxes(x.reshape(1,3,32,32),2,3),(1,1),unit= 0)
                X[i:i+1,:] = np.swapaxes(x.reshape(3,32,32),1,2).ravel()
            


            X_mean=X.mean(axis=1)
            X_std = (X.var(1)+10)**.5
            X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]

            X -= self.M

            # sigma = cov(X)
            # u,s,v=np.linalg.svd(sigma)
            # P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+.1)))),u.T)

            X=np.dot(X,self.P)
            X = X.reshape(10000,3,32,32)
            # print time.time()-t    
            return X,X,train_range_id_temp

        def sliceToBin(self,indx):
            offset = indx * 3072
            self.data_file.seek(offset)
            data = self.data_file.read(3072) 
            return np.fromstring(data, dtype='uint8').astype(np.float64)

