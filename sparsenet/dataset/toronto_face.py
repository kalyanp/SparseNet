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

class TorontoFace():

    # def __init__(self,want_mean=False):
    #     # pass
    #     return self.load(want_mean)

    @staticmethod
    def load(want_mean=False,want_dense=False):
        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/Toronto-Face/TFD_ranzato_48x48.mat")

        labs_ex = f['labs_ex']
        # print labs_ex.shape
        folds = f['folds']
        # print folds.shape
        labs_id = f['labs_id']
        # print labs_id.shape

        X = f['images']
        X = X.reshape(-1,1,48,48).astype("float32")
        print X.shape

        # print X.max()
        if want_mean:
            X_mean= X.mean(0)
            X_std = X.std(0)
            X = (X-X_mean)/X_std
            # X_test = (X_test-X_mean)/X_std

        if want_dense:
            X = X.reshape(-1,2304)
        # print X.max()
        return X


    @staticmethod
    def load_whiten2(p=.1):
        X = TorontoFace.load(want_mean=False,want_dense=True)

        # X_mean=X.mean(axis=1)
        # X_std = (X.var(1)+10)**.5
        # X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]

        #covariance
        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        #whiten
        
        X_mean = X.mean(axis=0)
        X -= X_mean

        sigma = cov(X)
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+p)))),u.T)
        
        X=np.dot(X,P)

        X = X.reshape(-1,1,48,48)
        return X

    @staticmethod
    def load_contrast(n=13,k=.01,filter="box",contrast="mean",want_dense = False):
        X = TorontoFace.load(want_mean=False,want_dense=True)
        X = X[:100000]

        X_mean=X.mean(axis=1)
        X_std = (X.var(1)+10)**.5

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X = X.reshape(-1,1,48,48)

        if filter=="box":
            filters = nn.ones((n,n))
        elif filter=="gaussian":
            filters = nn.gaussian2D(n,k)
        else: 
            raise Exception("unhandled case")


        X_cn = nn.zeros(X.shape)
        for i in xrange(10):
            if contrast=="mean":
                X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrastMean(X[10000*i:10000*(i+1),:,:,:], filters)
            elif contrast=="mean-var":
                X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X[10000*i:10000*(i+1),:,:,:], filters)
            else:             
                raise Exception("unhandled case")


        if want_dense: X_cn = X_cn.reshape(-1,48*48)
        return X_cn

    # @staticmethod
    # def load_whiten(backend="numpy",bias = .1):

    #     X,T,X_test,T_test,T_train_labels,T_labels=CIFAR10.load(want_mean=False,want_dense=True)

    #     #normalize for contrast
    #     X_mean=X.mean(axis=1)
    #     X_test_mean=X_test.mean(axis=1)
    #     X_std = X.std(1)
    #     X_test_std = X_test.std(1)

    #     X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
    #     X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]

    #     #covariance
    #     def cov(X):
    #         X_mean = X.mean(axis=0)
    #         X -= X_mean
    #         return np.dot(X.T,X)/(1.0*X.shape[0]-1)

    #     #whiten       
    #     X_mean = X.mean(axis=0)
    #     X -= X_mean
    #     X_test -= X_mean    

    #     sigma = cov(X)
    #     u,s,v=np.linalg.svd(sigma)
    #     P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+bias)))),u.T)
        
    #     X=np.dot(X,P)
    #     X_test=np.dot(X_test,P)

    #     X = X.reshape(50000,3,32,32)
    #     X_test = X_test.reshape(10000,3,32,32)

    #     if backend=="numpy":
    #         return X,T,X_test,T_test,T_train_labels,T_labels
    #     if backend=="gnumpy":
    #         return nn.garray(X),nn.garray(T),nn.garray(X_test),nn.garray(T_test),nn.garray(T_train_labels),nn.garray(T_labels)


    @staticmethod
    def load_whiten(bias=.1,want_dense=False):
        X = TorontoFace.load(want_mean=False,want_dense=True)
        # nn.show_images(X[63766].reshape(1,1,48,48),(1,1))
        # nn.show()
        # print X[63766].max()
        X_mean = X.mean(1)
        X_std = X.std(1)+10
        # print np.nonzero(X_std==0)
        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        # print X.min(),X.max()

        #covariance
        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        #whiten
        
        X_mean = X.mean(axis=0)
        X -= X_mean

        sigma = cov(X)+bias*np.identity(X.shape[1])
        # print X.min(),X.max()
        # print sigma.shape, type(sigma)
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s)))),u.T)

        # Q = P[0].reshape(48,48)
        # print Q[:4,:4].sum()
        # nn.show_images(P[0].reshape(1,1,48,48),(1,1))
        # nn.show()
        
        X=np.dot(X,P)

        # X = X.reshape(-1,1,48,48)
        if want_dense: X = X.reshape(-1,48*48)        
        return X


    @staticmethod
    def test():
        # X = TorontoFace.load(want_mean = True)
        X = TorontoFace.load_whiten(bias=.1)
        # X = TorontoFace.load_contrast(n=21,k=.01)
        # X = TorontoFace.load_contrast(n=3,k=.01,filter="box",contrast="mean")
        dp = nn.dp_ram(X=X,T=X,data_batch=9)
        while 1:
            X,T,_ = dp.train()
            nn.show_images(X,(3,3))
            nn.show()        


if __name__ == "__main__":
    nn.set_backend("numpy",board=0)
    # print nn.backend
    TorontoFace.test()