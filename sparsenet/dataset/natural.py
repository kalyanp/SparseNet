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

class Natural():
    
    @staticmethod
    def test():
        # X = Natural.load()
        X = Natural.extract_patch(num_patch=400000,size=7)
        nn.show_images(X[:100,:,:,:],(10,10))
        nn.show()
      

    @staticmethod
    def load():   
        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/Natural/IMAGES.mat")
        X = f['IMAGES']
        X = X.reshape(512**2,10).T.reshape(10,1,512,512)
        print X.max()
        return X

        # return X_cn,T,X_test_cn,T_test,T_train_labels,T_labels
    
    @staticmethod
    def extract_patch(backend="numpy",num_patch=10000,size=7,want_desne=True):
        img = Natural.load()
        assert(img.ndim)==4

        
        # img = img.reshape(50000,3,32,32)
        # print img.max()
        # nn.show_images(img[:9,:,:,:],(3,3))

        X = np.zeros((num_patch,img.shape[1],size,size))

        # extract random patches
        for index in xrange(num_patch):
            if index%100000==0: print index
            x = random.randint(0,img.shape[2]-size)
            y = random.randint(0,img.shape[3]-size)    
            patch = img[index%img.shape[0],:,x:x+size,y:y+size]
            X[index,:,:,:] = patch

        if want_desne:
            X = X.reshape(num_patch,size*size)

        if backend=="numpy":
            return X
        if backend=="gnumpy":
            return nn.garray(X)