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
#####################################
from svhn import SVHN
from frey import Frey
from mnist import MNIST
from imagenet import ImageNet
from cifar10 import CIFAR10
from norb import NORB
from toronto_face import TorontoFace
from natural import Natural
#####################################
def extract_patch(img,backend="numpy",num_patch=10000,size=7):
    # work_address = os.environ["WORK"]

    # assert whitened == True
    # if whitened: 
    #     f=np.load(work_address+"./Dataset/CIFAR10/cifar_bias_.1.npz")
    #     img=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
    # else:
    #     img,T,X_test,T_test,T_train_labels,T_labels = dataset.load_cifar10(raw=True)
    assert(img.ndim)==4

    
    # img = img.reshape(50000,3,32,32)
    # print img.max()
    # nn.show_images(img[:9,:,:,:],(3,3))

    X = np.zeros((num_patch,img.shape[1],size,size))

    # extract random patches
    for index in xrange(num_patch):
        if index%10000==0: print index
        x = random.randint(0,img.shape[2]-size)
        y = random.randint(0,img.shape[3]-size)    
        patch = img[index%img.shape[0],:,x:x+size,y:y+size]
        X[index,:,:,:] = patch

    if backend=="numpy":
        return X
    if backend=="gnumpy":
        return nn.garray(X)



# def load_norb(backend,dtype,resize=64,mode="single"):

#         rnd_permute = np.arange(24300)
#         # rnd_permute = np.random.permutation(24300)
#         data = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')
#         data = data.read()
#         data = np.fromstring(data[20:], dtype='uint32')
#         T_train_labels = data[rnd_permute]
#         if (mode == "single" or mode == "parallel" or mode == "binocular"): 
#             T = np.zeros((24300,5))
#             for n in range(24300):
#                 T[n,T_train_labels[n]] = 1
#         elif mode == "serial":
#             T_train_labels = np.concatenate((T_train_labels,T_train_labels),axis=1)
#             T = np.zeros((48600,5))
#             for n in range(48600):
#                 T[n,T_train_labels[n]]=1     
        
#         if mode == "single": X = np.zeros((24300,resize**2))
#         elif mode == "parallel": X = np.zeros((24300,2*resize**2))
#         elif mode == "serial": X = np.zeros((48600,resize**2))
#         elif mode == "binocular": X = np.zeros((24300,2*resize**2))
        
#         data_ = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
#         data_ = data_.read() 
#         data_ = np.fromstring(data_[24:], dtype='uint8')
#         data_ = np.reshape(data_,(24300,2,96,96))
        

#         #X = X[rnd_permute,:]
#         if mode == "serial":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#             data = data_[:,1,16:80,16:80]
#             for n in range(24300,48600):
#                 X[n] = scipy.misc.imresize(data[n-24300,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "parallel":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X[n][:resize**2] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#             data = data_[:,1,16:80,16:80]
#             for n in range(24300):
#                 X[n][-resize**2:] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "single":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "binocular":
#             data0 = data_[:,0,16:80,16:80]
#             data1 = data_[:,1,16:80,16:80]
#             for n in range(24300):
#                 a = scipy.misc.imresize(data0[n,:,:], (resize,resize) , 'bilinear')
#                 b = scipy.misc.imresize(data1[n,:,:], (resize,resize) , 'bilinear')
#                 X[n] = np.concatenate((a,b),axis=1).ravel()
            
#         X = X/255.0


#         data=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')
#         data=data.read() 
#         data=np.fromstring(data[20:], dtype='uint32')
#         T_labels=data
#         if (mode == "single" or mode == "parallel" or mode == "binocular"): 
#             T_test = np.zeros((24300,5))
#             for n in range(24300):
#                 T_test[n,T_labels[n]]=1
#         elif mode == "serial":
#             T_labels = np.concatenate((T_labels,T_labels),axis=1)
#             T_test = np.zeros((48600,5))
#             for n in range(48600):
#                 T_test[n,T_labels[n]]=1        
#         # print T_test.shape    

#         if mode == "single": X_test = np.zeros((24300,resize**2))
#         elif mode == "parallel": X_test = np.zeros((24300,2*resize**2))
#         elif mode == "serial": X_test = np.zeros((48600,resize**2))
#         elif mode == "binocular": X_test = np.zeros((24300,2*resize**2))

#         data_=open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
#         data_=data_.read() 
#         data_=np.fromstring(data_[24:], dtype='uint8')
#         data_=np.reshape(data_,(24300,2,96,96))
#         data=data_[:,0,16:80,16:80]
#         if mode == "serial":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X_test[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#             data = data_[:,1,16:80,16:80]
#             for n in range(24300,48600):
#                 X_test[n] = scipy.misc.imresize(data[n-24300,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "parallel":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X_test[n][:resize**2] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#             data = data_[:,1,16:80,16:80]
#             for n in range(24300):
#                 X_test[n][-resize**2:] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "single":
#             data  = data_[:,0,16:80,16:80]
#             for n in range(24300):
#                 X_test[n] = scipy.misc.imresize(data[n,:,:], (resize,resize) , 'bilinear').flatten()
#         elif mode == "binocular":
#             data0 = data_[:,0,16:80,16:80]
#             data1 = data_[:,1,16:80,16:80]
#             for n in range(24300):
#                 a = scipy.misc.imresize(data0[n,:,:], (resize,resize) , 'bilinear')
#                 b = scipy.misc.imresize(data1[n,:,:], (resize,resize) , 'bilinear')
#                 X_test[n] = np.concatenate((a,b),axis=1).ravel()
#         # X = X[rnd_permute,:]
#         X_test = X_test/255.0
#         # print X.shape # print X_test.shape

#         if backend=="numpy": X=np.array(X,dtype);T=np.array(T,dtype);X_test=np.array(X_test,dtype);T_test=np.array(T_test,dtype);T_train_labels=np.array(T_train_labels,dtype);T_labels=np.array(T_labels,dtype)
#         # if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T)
#         if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
        
#         #return X,T
#         return X,T,X_test,T_test,T_train_labels,T_labels




