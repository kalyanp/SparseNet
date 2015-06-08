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

class MNIST():

    @staticmethod
    def test():
        # X = TorontoFace.load(want_mean = True)
        # X = TorontoFace.load_whiten(p=1)
        # X = TorontoFace.load_contrast(n=21,k=.01)
        X,T,X_test,T_test,T_train_labels,T_labels = MNIST.semi(N=100,want_dense = False)
        dp = nn.dp_ram(X,T,data_batch=100)
        while 1:
            x,t,id = dp.train()
            nn.show_images(x,(10,10))
            print t
            nn.show()        
        
    @staticmethod
    def semi(N, want_dense = True):
        k = N/10
        # print k
        X,T,X_test,T_test,T_train_labels,T_labels = MNIST.load(want_dense = True)
      
        count = [0]*10
        lst = []
        index = 0
        while len(lst)<N:
            label = int(T_train_labels[index])        
            if count[label] < k: 
                lst.append(index)
                count[label]+=1
            index += 1

        X_semi = X[lst]
        T_semi = T[lst]

        if want_dense==False:
            X_semi = X_semi.reshape(10*k,1,28,28)
            X_test = X_test.reshape(10000,1,28,28)

        # print count
        return X_semi,T_semi,X_test,T_test,None,None,lst


    @staticmethod
    def load(backend="numpy",binary = False, want_dense = False):

        s=60000

        T=np.zeros((s,10))
        
        data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/MNIST/train-images.idx3-ubyte','r')
        data=data_.read() 
        data_=np.fromstring(data[16:], dtype='uint8')
        X=np.reshape(data_,(s,784))/255.0
     
        data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/MNIST/train-labels.idx1-ubyte','r')
        data=data_.read()
        T_train_labels = np.fromstring(data[8:], dtype='uint8')

        for n in range(s):
            T[n,T_train_labels[n]]=1
            


        s_test=10000
        X_test=np.zeros((s_test,784))
        T_test=np.zeros((s_test,10))


        data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/MNIST/t10k-images.idx3-ubyte','r')
        data=data_.read()
        data_=np.fromstring(data[16:], dtype='uint8')
        X_test=np.reshape(data_,(s_test,784))/255.0


        data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/MNIST/t10k-labels.idx1-ubyte','r')
        data=data_.read()
        T_labels = np.fromstring(data[8:], dtype='uint8')
        T_labels = T_labels.astype("float32")

        for n in range(s_test):
            T_test[n,T_labels[n]]=1

        if binary:
            X[X>.5]=1
            X[X<.5]=0    
            X_test[X_test>.5]=1.0
            X_test[X_test<.5]=0.0    


        if want_dense==False:
            X = X.reshape(60000,1,28,28)
            X_test = X_test.reshape(10000,1,28,28)            
        
        if backend=="numpy": X=nn.array(X);T=nn.array(T);X_test=nn.array(X_test);T_test=nn.array(T_test);T_train_labels=nn.array(T_train_labels);T_labels=nn.array(T_labels)
        if backend=="gnumpy": X=nn.garray(X);T=nn.garray(T);X_test=nn.garray(X_test);T_test=nn.garray(T_test);T_train_labels=nn.garray(T_train_labels);T_labels=nn.garray(T_labels)
       
        # print X.dtype,T.dtype
        return X,T,X_test,T_test,T_train_labels,T_labels

