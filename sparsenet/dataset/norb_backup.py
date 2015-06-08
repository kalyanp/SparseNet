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

class NORB():
    @staticmethod
    def load(size=64,want_mean=False,want_dense=False,serial=True):

        if serial: X = np.zeros((48600,size**2))        
        else:      X = np.zeros((24300,size**2))        

        data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
        data = data.read() 
        data = np.fromstring(data[24:], dtype='uint8')
        data = data.reshape(24300,2,96,96).astype("float32")

        for n in range(24300): 
            X[n] = scipy.misc.imresize(data[n,0,16:80,16:80], (size,size) , 'bilinear').flatten()
        if serial:
            for n in range(24300): 
                X[n+24300] = scipy.misc.imresize(data[n,1,16:80,16:80], (size,size) , 'bilinear').flatten()            
            
        data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')
        data = data.read()
        data = np.fromstring(data[20:], dtype='uint32')
        

        T_train_labels[:24300] = data
        if serial: T_train_labels[24300:] = data

        if serial:
            T = np.zeros((48600,5))
            for n in range(48600):
                T[n,T_train_labels[n]] = 1
        else:
            T = np.zeros((24300,5))
            for n in range(24300):
                T[n,T_train_labels[n]] = 1
        ###################################

        if serial: X_test = np.zeros((48600,size**2))        
        else:      X_test = np.zeros((24300,size**2))    

        data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
        data = data.read() 
        data = np.fromstring(data[24:], dtype='uint8')
        data = data.reshape(24300,2,96,96).astype("float32")    

        for n in range(24300): 
            X_test[n] = scipy.misc.imresize(data[n,0,16:80,16:80], (size,size) , 'bilinear').flatten()
        if serial:
            for n in range(24300): 
                X_test[n+24300] = scipy.misc.imresize(data[n,1,16:80,16:80], (size,size) , 'bilinear').flatten() 

        data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')
        data = data.read()
        data = np.fromstring(data[20:], dtype='uint32')

        T_labels[:24300] = data
        if serial: T_labels[24300:] = data
        
        if serial:
            T_test = np.zeros((48600,5))
            for n in range(48600):
                T_test[n,T_labels[n]]=1
        else:
            T_test = np.zeros((24300,5))
            for n in range(24300):
                T_test[n,T_labels[n]]=1
        
        if want_mean:
            X_mean= X.mean(0)
            X_std = X.std(0)
            X = (X-X_mean)/X_std
            X_test = (X_test-X_mean)/X_std


        if not want_dense:
            X = X.reshape(24300,1,size,size)
            X_test = X_test.reshape(24300,1,size,size)

        return X,T,X_test,T_test,T_train_labels,T_labels



    @staticmethod
    def load_whiten(size = 32, want_dense = False,bias  = .1):

        X,T,X_test,T_test,T_train_labels,T_labels=NORB.load(size=size, want_mean=False, want_dense=True)
        # X,T,X_test,T_test,T_train_labels,T_labels        
        # print X.shape

        #normalize for contrast
        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)

        X_std = (X.var(1)+10)**.5
        X_test_std = (X_test.var(1)+10)**.5

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]

        #covariance
        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        #whiten
        
        M = X.mean(axis=0)
        X -= M
        X_test -= M    

        sigma = cov(X)
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+bias)))),u.T)
        
        X=np.dot(X,P)
        X_test=np.dot(X_test,P)

        # X=nn.garray(X)
        # print X.shape
        # np.savez("cifar10_adam", X = X)
        # print X.min(0)
        # print (X**2).sum(1)[:100]    
        if not want_dense:
            X = X.reshape(24300,1,size,size)
            X_test = X_test.reshape(24300,1,size,size)

        return X,T,X_test,T_test,T_train_labels,T_labels




    @staticmethod
    def load_norb(size=64,mode="single",want_dense=False):

        rnd_permute = np.arange(24300)

        data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')
        data = data.read()
        data = np.fromstring(data[20:], dtype='uint32')
        T_train_labels = data[rnd_permute]
        if (mode == "single" or mode == "parallel" or mode == "binocular"): 
            T = np.zeros((24300,5))
            for n in range(24300):
                T[n,T_train_labels[n]] = 1
        elif mode == "serial":
            T_train_labels = np.concatenate((T_train_labels,T_train_labels),axis=1)
            T = np.zeros((48600,5))
            for n in range(48600):
                T[n,T_train_labels[n]]=1     
        
        if mode == "single": X = np.zeros((24300,size**2))
        elif mode == "parallel": X = np.zeros((24300,2*size**2))
        elif mode == "serial": X = np.zeros((48600,size**2))
        elif mode == "binocular": X = np.zeros((24300,2*size**2))
        
        data_ = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
        data_ = data_.read() 
        data_ = np.fromstring(data_[24:], dtype='uint8')
        data_ = np.reshape(data_,(24300,2,96,96))
        

        #X = X[rnd_permute,:]
        if mode == "serial":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X[n] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
            data = data_[:,1,16:80,16:80]
            for n in range(24300,48600):
                X[n] = scipy.misc.imresize(data[n-24300,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "parallel":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X[n][:size**2] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
            data = data_[:,1,16:80,16:80]
            for n in range(24300):
                X[n][-size**2:] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "single":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X[n] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "binocular":
            data0 = data_[:,0,16:80,16:80]
            data1 = data_[:,1,16:80,16:80]
            for n in range(24300):
                a = scipy.misc.imresize(data0[n,:,:], (size,size) , 'bilinear')
                b = scipy.misc.imresize(data1[n,:,:], (size,size) , 'bilinear')
                X[n] = np.concatenate((a,b),axis=1).ravel()
            
        # X = X/255.0


        data=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')
        data=data.read() 
        data=np.fromstring(data[20:], dtype='uint32')
        T_labels=data
        if (mode == "single" or mode == "parallel" or mode == "binocular"): 
            T_test = np.zeros((24300,5))
            for n in range(24300):
                T_test[n,T_labels[n]]=1
        elif mode == "serial":
            T_labels = np.concatenate((T_labels,T_labels),axis=1)
            T_test = np.zeros((48600,5))
            for n in range(48600):
                T_test[n,T_labels[n]]=1        
        # print T_test.shape    

        if mode == "single": X_test = np.zeros((24300,size**2))
        elif mode == "parallel": X_test = np.zeros((24300,2*size**2))
        elif mode == "serial": X_test = np.zeros((48600,size**2))
        elif mode == "binocular": X_test = np.zeros((24300,2*size**2))

        data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
        data_=data_.read() 
        data_=np.fromstring(data_[24:], dtype='uint8')
        data_=np.reshape(data_,(24300,2,96,96))
        data=data_[:,0,16:80,16:80]
        if mode == "serial":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X_test[n] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
            data = data_[:,1,16:80,16:80]
            for n in range(24300,48600):
                X_test[n] = scipy.misc.imresize(data[n-24300,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "parallel":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X_test[n][:resize**2] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
            data = data_[:,1,16:80,16:80]
            for n in range(24300):
                X_test[n][-resize**2:] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "single":
            data  = data_[:,0,16:80,16:80]
            for n in range(24300):
                X_test[n] = scipy.misc.imresize(data[n,:,:], (size,size) , 'bilinear').flatten()
        elif mode == "binocular":
            data0 = data_[:,0,16:80,16:80]
            data1 = data_[:,1,16:80,16:80]
            for n in range(24300):
                a = scipy.misc.imresize(data0[n,:,:], (size,size) , 'bilinear')
                b = scipy.misc.imresize(data1[n,:,:], (size,size) , 'bilinear')
                X_test[n] = np.concatenate((a,b),axis=1).ravel()
        # X = X[rnd_permute,:]
        # X_test = X_test/255.0
        # print X.shape # print X_test.shape

        # if backend=="numpy": X=np.array(X,dtype);T=np.array(T,dtype);X_test=np.array(X_test,dtype);T_test=np.array(T_test,dtype);T_train_labels=np.array(T_train_labels,dtype);T_labels=np.array(T_labels,dtype)
        # if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T)
        # if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
        
        #return X,T
        X = X.astype("float32")
        X_test = X_test.astype("float32")

        X = X/255.0
        X_test = X_test/255.0

        if not want_dense:
            X = X.reshape(-1,1,size,size)
            X_test = X_test.reshape(-1,1,size,size)


        return X,T,X_test,T_test,T_train_labels,T_labels


    @staticmethod
    def test():
        X,T,X_test,T_test,T_train_labels,T_labels = NORB.load_norb(size=32,mode="serial",want_dense=False)
        # X,T,X_test,T_test,T_train_labels,T_labels = NORB.load_whiten(size=32)
        print X.dtype,X.max()
        dp = nn.dp_ram(X=X,T=T,X_test=X_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,data_batch=25)
        while 1:
            x,t,id = dp.train()
            print x.shape
            nn.show_images(x,(5,5))
            print t
            nn.show()   

# np.savez("./dataset/norb_binocular_small_normalized", X=X,T=T,X_test=X_test,T_test=T_test,T_train_labels=T_train_labels,T_labels=T_labels)
# f=np.load(work_address+'./dataset/norb_single_small.npz')
# X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
# f=np.load(work_address+'./dataset/norb_binocular_small.npz')
# X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
# X_std=X.std(axis=0)
# X_mean=X.mean(axis=0)
# X_test_std=X_test.std(axis=0)
# X_test_mean=X_test.mean(axis=0)
# X=(X-X_mean)/X_std
# X_test=(X_test-X_test_mean)/X_test_std

# np.savez("./dataset/norb_binocular_small_normalized", X=X,T=T,X_test=X_test,T_test=T_test,T_train_labels=T_train_labels,T_labels=T_labels)
# f=np.load(work_address+'./dataset/norb_single_small.npz')
# X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
# f=np.load(work_address+'./dataset/norb_binocular_small.npz')
# X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!numpy object dtype correct it