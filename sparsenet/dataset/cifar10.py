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

class CIFAR10():
    
    @staticmethod
    def test():
        # X,T,X_test,T_test,T_train_labels,T_labels = CIFAR10.load(want_mean=True,want_dense=False)
        # X,T,X_test,T_test,T_train_labels,T_labels = CIFAR10.load_whiten(bias=.1)
        X,T,X_test,T_test,T_train_labels,T_labels = CIFAR10.load_contrast(n=13,k=.01)
        # X,T,X_test,T_test,T_train_labels,T_labels = NORB.load_whiten(size=32,bias=.1)
        # X = CIFAR10.load_cifar10_adam_patch(num_patch=100000,size=11,backend="numpy")
        # print X.dtype,X.max()
        dp = nn.dp_ram(X=X,X_test=X,data_batch=25)
        while 1:
            x,t,id = dp.train()
            print x.shape
            nn.show_images(x,(5,5))
            print t
            nn.show()        

    @staticmethod
    def load_contrast(n=13,k=.01,want_dense=False,filter="gaussian",contrast="mean"):

        X,T,X_test,T_test,T_train_labels,T_labels = CIFAR10.load(want_mean=False,want_dense=True)

        # for i in xrange(3):
        #     mean = X[:,i,:,:].mean()
        #     std = X[:,i,:,:].std()
        #     X[:,i,:,:] = (X[:,i,:,:] - mean)/std
        #     X_test[:,i,:,:] = (X_test[:,i,:,:] - mean)/std

        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)

        X_std = X.std(1)
        X_test_std = X_test.std(1)

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]

        X = X.reshape(50000,3,32,32)
        X_test = X_test.reshape(10000,3,32,32)

        if filter=="box":
            filters = nn.ones((n,n))
        elif filter=="gaussian":
            filters = nn.gaussian2D(n,k)
        else: 
            raise Exception("unhandled case")

        X_cn = nn.zeros(X.shape)
        X_test_cn = nn.zeros(X_test.shape)

        for i in xrange(5):
            if contrast=="mean":
                X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrastMean(X[10000*i:10000*(i+1),:,:,:], filters)
            elif contrast=="mean-var":
                X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X[10000*i:10000*(i+1),:,:,:], filters)
            else:             
                raise Exception("unhandled case")
        for i in xrange(1):
            if contrast=="mean":
                X_test_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrastMean(X_test[10000*i:10000*(i+1),:,:,:], filters)
            elif contrast=="mean-var":
                X_test_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X_test[10000*i:10000*(i+1),:,:,:], filters)
            else:             
                raise Exception("unhandled case")

        if want_dense:
            X_cn = X_cn.reshape(50000,3072)
            X_test_cn = X_test_cn.reshape(10000,3072)

        return X_cn,T,X_test_cn,T_test,T_train_labels,T_labels


    # @staticmethod
    # def load_contrast(n=13,k=.01,filter="box",contrast="mean"):
    #     X = TorontoFace.load(want_mean=False,want_dense=True)
    #     X = X[:100000]

    #     X_mean=X.mean(axis=1)
    #     X_std = (X.var(1)+10)**.5

    #     X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
    #     X = X.reshape(-1,1,48,48)

    #     if filter=="box":
    #         filters = nn.ones((n,n))
    #     elif filter=="gaussian":
    #         filters = nn.gaussian2D(n,k)
    #     else: 
    #         raise Exception("unhandled case")


    #     X_cn = nn.zeros(X.shape)
    #     for i in xrange(10):
    #         if contrast=="mean":
    #             X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrastMean(X[10000*i:10000*(i+1),:,:,:], filters)
    #         elif contrast=="mean-var":
    #             X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X[10000*i:10000*(i+1),:,:,:], filters)
    #         else:             
    #             raise Exception("unhandled case")


    #     X_cn = X_cn.reshape(-1,1,48,48)
    #     return X_cn









    @staticmethod
    def load(backend="numpy",want_mean=True,want_dense = False):
        work_address = os.environ["WORK"]

        X=np.zeros((50000,3072))
        T=np.zeros((50000,10))
        T_train_labels=np.zeros(50000)

        X_test=np.zeros((10000,3072))
        T_test=np.zeros((10000,10))
        T_labels=np.zeros(10000)

    # nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/alex/

        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_1', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[:10000]=dict['data'].T
        T_train_labels[:10000]= dict['labels']
        for i in range(10000):
            T[i,dict['labels'][i]]= 1
            
        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_2', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[10000:20000]=dict['data'].T
        T_train_labels[10000:20000]= dict['labels']
        for i in range(10000):
            T[i+10000,dict['labels'][i]]= 1
            
        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_3', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[20000:30000]=dict['data'].T
        T_train_labels[20000:30000]= dict['labels']
        for i in range(10000):
            T[i+20000,dict['labels'][i]]= 1
            
        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_4', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[30000:40000]=dict['data'].T
        T_train_labels[30000:40000]= dict['labels']
        for i in range(10000):
            T[i+30000,dict['labels'][i]]= 1
            
        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_5', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[40000:50000]=dict['data'].T
        T_train_labels[40000:50000]= dict['labels']
        for i in range(10000):
            T[i+40000,dict['labels'][i]]= 1
            
        fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_6', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X_test[:10000]=dict['data'].T
        T_labels[:10000]= dict['labels']
        for i in range(10000):
            T_test[i,dict['labels'][i]]= 1

        if want_mean:
            fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/batches.meta', 'rb')
            dict = cPickle.load(fo)
            fo.close()
            X_mean=dict['data_mean']
            X-=X_mean.T
            X_test-=X_mean.T

            # print X_mean.max()

            X = X/255.0
            X_test = X_test/255.0
            # print "Dataset mean subtracted."
        else:   
            pass     
            # print "Dataset mean NOT subtracted."  

        if not want_dense:
            X = X.reshape(50000,3,32,32)
            X_test = X_test.reshape(10000,3,32,32)

        if backend=="numpy": X=np.array(X);T=np.array(T);X_test=np.array(X_test);T_test=np.array(T_test);T_train_labels=np.array(T_train_labels);T_labels=np.array(T_labels)
        if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
        
        return X,T,X_test,T_test,T_train_labels,T_labels

    @staticmethod
    def load_whiten(backend="numpy",bias = .1):

        X,T,X_test,T_test,T_train_labels,T_labels=CIFAR10.load(want_mean=False,want_dense=True)

        #normalize for contrast
        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)
        X_std = X.std(1)
        X_test_std = X_test.std(1)

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]

        #covariance
        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        #whiten       
        X_mean = X.mean(axis=0)
        X -= X_mean
        X_test -= X_mean    

        sigma = cov(X)
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+bias)))),u.T)
        
        X=np.dot(X,P)
        X_test=np.dot(X_test,P)

        X = X.reshape(50000,3,32,32)
        X_test = X_test.reshape(10000,3,32,32)

        if backend=="numpy":
            return X,T,X_test,T_test,T_train_labels,T_labels
        if backend=="gnumpy":
            return nn.garray(X),nn.garray(T),nn.garray(X_test),nn.garray(T_test),nn.garray(T_train_labels),nn.garray(T_labels)


    @staticmethod    
    def load_pylearn2(bias=.1):

        X,T,X_test,T_test,T_train_labels,T_labels=CIFAR10.load(want_mean=False,want_dense=True)

        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)
        X=(X-X_mean[:,np.newaxis])
        X_test=(X_test-X_test_mean[:,np.newaxis])    

        normalizers = np.sqrt((X ** 2).sum(axis=1))/55.
        normalizers_test = np.sqrt((X_test ** 2).sum(axis=1))/55.

        X /= normalizers[:, np.newaxis]
        X_test /= normalizers_test[:, np.newaxis]


        def cov(X):
            return np.dot(X.T,X)/(1.0*X.shape[0])

        X_mean = X.mean(axis=0)
        X -= X_mean
        X_test -= X_mean

        sigma = cov(X)+bias*np.identity(X.shape[1])
        u,s,v=np.linalg.svd(sigma)

        P = np.dot(np.dot(u,np.diag(np.sqrt(1./s))),u.T)

        X=np.dot(X,P)
        X_test = np.dot(X_test,P)
        
        X = X.reshape(50000,3,32,32)
        X_test = X_test.reshape(10000,3,32,32)
        return X,T,X_test,T_test,T_train_labels,T_labels        
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################

    @staticmethod
    def load_whiten_disk():

        f=np.load(nn.work_address()+"/Dataset/CIFAR10/cifar_bias_.1_new.npz")
        X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
        X = X.reshape(3072,50000).T.reshape(50000,3,32,32)
        X_test = X_test.reshape(3072,10000).T.reshape(10000,3,32,32)        
        return X,T,X_test,T_test,T_train_labels,T_labels
        # X=nn.garray(X);T=nn.garray(T);X_test=nn.garray(X_test);T_test=nn.garray(T_test);T_train_labels=nn.garray(T_train_labels);T_labels=nn.garray(T_labels)

    @staticmethod
    def load_cifar10(backend, want_normalized = False, alex = False):
        X=np.zeros((50000,3072))
        T=np.zeros((50000,10))
        T_train_labels=np.zeros(50000)

        X_test=np.zeros((10000,3072))
        T_test=np.zeros((10000,10))
        T_labels=np.zeros(10000)

        fo = open(work_address+'./Dataset/CIFAR10/data_batch_1', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[:10000]=dict['data']
        T_train_labels[:10000]= dict['labels']
        for i in range(10000):
            T[i,dict['labels'][i]]= 1
            
        fo = open(work_address+'./Dataset/CIFAR10/data_batch_2', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[10000:20000]=dict['data']
        T_train_labels[10000:20000]= dict['labels']
        for i in range(10000):
            T[i+10000,dict['labels'][i]]= 1
            
        fo = open(work_address+'./Dataset/CIFAR10/data_batch_3', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[20000:30000]=dict['data']
        T_train_labels[20000:30000]= dict['labels']
        for i in range(10000):
            T[i+20000,dict['labels'][i]]= 1
            
        fo = open(work_address+'./Dataset/CIFAR10/data_batch_4', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[30000:40000]=dict['data']
        T_train_labels[30000:40000]= dict['labels']
        for i in range(10000):
            T[i+30000,dict['labels'][i]]= 1
            
        fo = open(work_address+'./Dataset/CIFAR10/data_batch_5', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X[40000:50000]=dict['data']
        T_train_labels[40000:50000]= dict['labels']
        for i in range(10000):
            T[i+10000,dict['labels'][i]]= 1
            
        fo = open(work_address+'./Dataset/CIFAR10/test_batch', 'rb')
        dict = cPickle.load(fo)
        fo.close()
        X_test[:10000]=dict['data']
        T_labels[:10000]= dict['labels']
        for i in range(10000):
            T_test[i,dict['labels'][i]]= 1

        X = X/255.0
        X_test = X_test/255.0

        if want_normalized:
            X_std=X.std(axis=0)
            X_mean=X.mean(axis=0)
            X_test_std=X_test.std(axis=0)
            X_test_mean=X_test.mean(axis=0)
            X=(X-X_mean)/X_std
            X_test=(X_test-X_test_mean)/X_test_std

        if alex:
            X_mean=X.mean(axis=0)
            X_test_mean=X_test.mean(axis=0)
            X=(X-X_mean)
            X_test=(X_test-X_test_mean)

        if backend=="numpy": X=np.array(X);T=np.array(T);X_test=np.array(X_test);T_test=np.array(T_test);T_train_labels=np.array(T_train_labels);T_labels=np.array(T_labels)
        if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)

        return X,T,X_test,T_test,T_train_labels,T_labels

    @staticmethod
    def load_cifar10_adam_patch(num_patch,size,backend="numpy",want_dense = True):

        img,T,X_test,T_test,T_train_labels,T_labels=CIFAR10.load(want_mean = False, want_dense = True)
        img = img.reshape(50000,3,32,32)
        # print img.max()
        # nn.show_images(img[:9,:,:,:],(3,3))



        X = np.zeros((num_patch,3,size,size))

        # extract random patches
        # for index in xrange(num_patch):
        #     x = random.randint(0,31-size)
        #     y = random.randint(0,31-size)    
        #     patch = img[index%50000,:,x:x+size,y:y+size]
        #     X[index,:,:,:] = patch

        for index in xrange(num_patch):
            if index%10000==0: print index
            x = random.randint(0,img.shape[2]-size)
            y = random.randint(0,img.shape[3]-size)    
            patch = img[index%img.shape[0],:,x:x+size,y:y+size]
            X[index,:,:,:] = patch


        X = X.reshape(num_patch,-1)

        #normalize for contrast
        X_mean=X.mean(axis=1)
        print X_mean.shape
        # X_std = (X.std(1)+10)**.5
        # X_std = X.std(1)
        X_std = (X.var(1)+10)**.5
        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]

        #covariance
        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        #whiten
        sigma = cov(X)+.1*np.identity(X.shape[1])
        M = X.mean(axis=0)
        X -= M
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./s))),u.T)
        X=np.dot(X,P)

        # X=nn.garray(X)
        # print X.shape
        # np.savez("cifar10_adam", X = X)
        # print X.min(0)
        # print (X**2).sum(1)[:100]    
        if not want_dense:
            X = X.reshape(num_patch,3,size,size)
        if backend=="numpy":
            return X,M,P
        if backend=="gnumpy":
            return nn.garray(X)
  










    @staticmethod
    def load_cifar10_adam(backend="numpy",bias  = .1):

        X,T,X_test,T_test,T_train_labels,T_labels=load_cifar10(raw=True)
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

        X = X.reshape(50000,3,32,32)
        X_test = X_test.reshape(10000,3,32,32)


        # np.savez(nn.work_address()+"/Dataset/CIFAR10/cifar_bias_.1_adam", X=X,T=T,X_test=X_test,T_test=T_test,T_train_labels=T_train_labels,T_labels=T_labels)

        if backend=="numpy":
            return X,T,X_test,T_test,T_train_labels,T_labels,M,P
        # if backend=="gnumpy":
        #     return nn.garray(X),nn.garray(T),nn.garray(X_test),nn.garray(T_test),nn.garray(T_train_labels),nn.garray(T_labels)

    @staticmethod

    def load_cifar10_filter(backend="numpy", size = 5):

        assert backend == "gnumpy"
        X,T,X_test,T_test,T_train_labels,T_labels=load_cifar10(raw=True)
        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)

        X_std = (X.var(1)+10)**.5
        X_test_std = (X_test.var(1)+10)**.5

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]    

        X = X.reshape(50000,3,32,32)
        X_test = X_test.reshape(10000,3,32,32)
        
        filter = np.zeros((3,3,size,size))
        T=nn.garray(T);T_test=nn.garray(T_test);T_train_labels=nn.garray(T_train_labels);T_labels=nn.garray(T_labels)

        for i in xrange(3):
            b = np.ones((size,size))*(1.0/size**2)
            a = np.zeros((size,size))
            a[(size-1)/2,(size-1)/2]=1
            filter[i,i,:,:] = a-b 


        # X_filter = nn.ConvUp(X, filter, moduleStride = 1, paddingStart = (size-1)/2)        
        # X_filter_test = nn.ConvUp(X_test, filter, moduleStride = 1, paddingStart = (size-1)/2)        
      
        X_filter = nn.GnumpyBackend.zeros((50000,3,32,32))
        for i in range(250):
            X_filter[i*200:(i+1)*200,:,:,:] = nn.ConvUp(nn.garray(X[i*200:(i+1)*200,:,:,:]), filter, moduleStride = 1, paddingStart = (size-1)/2)
        X_filter_test = nn.GnumpyBackend.zeros((10000,3,32,32))
        for i in range(50):
            X_filter_test[i*200:(i+1)*200,:,:,:] = nn.ConvUp(nn.garray(X_test[i*200:(i+1)*200,:,:,:]), filter, moduleStride = 1, paddingStart = (size-1)/2)
        
        if backend=="gnumpy":
            return X_filter,T,X_filter_test,T_test,T_train_labels,T_labels        






    # def load_cifar10_raw(backend):
    #     X=np.zeros((50000,3072))
    #     T=np.zeros((50000,10))
    #     T_train_labels=np.zeros(50000)

    #     X_test=np.zeros((10000,3072))
    #     T_test=np.zeros((10000,10))
    #     T_labels=np.zeros(10000)

    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_1', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X[:10000]=dict['data'].T
    #     T_train_labels[:10000]= dict['labels']
    #     for i in range(10000):
    #         T[i,dict['labels'][i]]= 1
            
    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_2', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X[10000:20000]=dict['data'].T
    #     T_train_labels[10000:20000]= dict['labels']
    #     for i in range(10000):
    #         T[i+10000,dict['labels'][i]]= 1
            
    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_3', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X[20000:30000]=dict['data'].T
    #     T_train_labels[20000:30000]= dict['labels']
    #     for i in range(10000):
    #         T[i+20000,dict['labels'][i]]= 1
            
    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_4', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X[30000:40000]=dict['data'].T
    #     T_train_labels[30000:40000]= dict['labels']
    #     for i in range(10000):
    #         T[i+30000,dict['labels'][i]]= 1
            
    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_5', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X[40000:50000]=dict['data'].T
    #     T_train_labels[40000:50000]= dict['labels']
    #     for i in range(10000):
    #         T[i+40000,dict['labels'][i]]= 1
            
    #     fo = open(work_address+'./Dataset/CIFAR10/alex/data_batch_6', 'rb')
    #     dict = cPickle.load(fo)
    #     fo.close()
    #     X_test[:10000]=dict['data'].T
    #     T_labels[:10000]= dict['labels']
    #     for i in range(10000):
    #         T_test[i,dict['labels'][i]]= 1

    #     # fo = open(work_address+'./Dataset/CIFAR10/alex/batches.meta', 'rb')
    #     # dict = cPickle.load(fo)
    #     # fo.close()
    #     # X_mean=dict['data_mean']
    #     # X-=X_mean.T
    #     # X_test-=X_mean.T
        
    #     if backend=="numpy": X=np.array(X);T=np.array(T);X_test=np.array(X_test);T_test=np.array(T_test);T_train_labels=np.array(T_train_labels);T_labels=np.array(T_labels)
    #     if backend=="gnumpy": X=gp.garray(X);T=gp.garray(T);X_test=gp.garray(X_test);T_test=gp.garray(T_test);T_train_labels=gp.garray(T_train_labels);T_labels=gp.garray(T_labels)
        
    #     return X,T,X_test,T_test,T_train_labels,T_labels    

    @staticmethod

    class data_provider:

        def __init__(self,train_range=None,test_range=None,mini_batch = None,crop_size = 24,want_auto = False,bias=None):
            self.bias = bias
            if bias:
                # print nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/CIFAR10/MP_"+str(bias)
                try:
                    f = np.load(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/CIFAR10/MP_"+str(bias)+".npz")
                    self.M = f['M']; self.P = f['P'];
                except:                
                    # print "hello"
                    _,_,_,_,_,_,self.M,self.P = load_cifar10_adam(backend="numpy",bias = bias)
                    print "M and P saved to NAS."
                    np.savez(nn.nas_address+"/PSI-Share-no-backup/Ali/Dataset/CIFAR10/MP_.1",M=M,P=P)                
            # self.want_whiten = want_whiten

            self.num_threads = 5
            self.want_auto = want_auto

            self.start = True
            self.switch = 0
            self.train_range = train_range
            self.test_range = test_range
            assert train_range!=None
            self.train_range_id = self.train_range[0]
            self.test_range_id = self.test_range[0]
            self.mini_batch = mini_batch

            self.crop_size = crop_size
            self.crop_offset = (32-crop_size)

            shared_array_base_X0 = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*10000)
            shared_array_X0 = np.ctypeslib.as_array(shared_array_base_X0.get_obj())
            self.X0 = shared_array_X0.reshape(10000,3,self.crop_size,self.crop_size)

            shared_array_base_X1 = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*10000)
            shared_array_X1 = np.ctypeslib.as_array(shared_array_base_X1.get_obj())
            self.X1 = shared_array_X1.reshape(10000,3,self.crop_size,self.crop_size)

            shared_array_base_T0 = multiprocessing.Array(ctypes.c_double, 10000*10)
            shared_array_T0 = np.ctypeslib.as_array(shared_array_base_T0.get_obj())
            self.T0 = shared_array_T0.reshape(10000,10)

            shared_array_base_T1 = multiprocessing.Array(ctypes.c_double, 10000*10)
            shared_array_T1 = np.ctypeslib.as_array(shared_array_base_T1.get_obj())
            self.T1 = shared_array_T1.reshape(10000,10)

            if test_range!=None:
                shared_array_base_X_test = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*10000)
                shared_array_X_test = np.ctypeslib.as_array(shared_array_base_X_test.get_obj())
                self.X_test = shared_array_X_test.reshape(10000,3,self.crop_size,self.crop_size)
                assert self.X_test.base.base is shared_array_base_X_test.get_obj()  

                shared_array_base_T_test = multiprocessing.Array(ctypes.c_double, 10000*10000)
                shared_array_T_test = np.ctypeslib.as_array(shared_array_base_T_test.get_obj())
                self.T_test = shared_array_T_test.reshape(10000,10000)
                assert self.T_test.base.base is shared_array_base_T_test.get_obj()

                shared_array_base_T_labels_test = multiprocessing.Array(ctypes.c_double, 10000)
                shared_array_T_labels_test = np.ctypeslib.as_array(shared_array_base_T_labels_test.get_obj())
                self.T_labels_test = shared_array_T_labels_test.reshape(10000)
                assert self.T_labels_test.base.base is shared_array_base_T_labels_test.get_obj()                                              
         

        def __len__(self): return len(self.train_range) 

        def train(self): 
            if self.start:
                offset_x=np.random.randint(0,self.crop_offset+1) 
                offset_y=np.random.randint(0,self.crop_offset+1)                
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,self.train_range[0],None,offset_x,offset_y))
                self.p_load.start()
                self.p_load.join()
                self.start = False
            else: self.p_load.join()

            if self.switch == 0:
                train_range_id_next = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]
                offset_x=np.random.randint(0,self.crop_offset+1) 
                offset_y=np.random.randint(0,self.crop_offset+1)    
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X1,self.T1,train_range_id_next,None,offset_x,offset_y))
                self.p_load.start()
                self.switch = 1
                temp = self.train_range_id
                self.train_range_id = train_range_id_next
                if self.want_auto:                                                                               
                    return self.X0,self.X0.copy(),temp   
                else:
                    return self.X0,self.T0,temp  
            else: 
                train_range_id_next = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]
                offset_x=np.random.randint(0,self.crop_offset+1) 
                offset_y=np.random.randint(0,self.crop_offset+1)            
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,train_range_id_next,None,offset_x,offset_y))
                self.p_load.start()
                self.switch = 0
                temp = self.train_range_id           
                self.train_range_id = train_range_id_next
                if self.want_auto:                                                                               
                    return self.X1,self.X1.copy(),temp   
                else:
                    return self.X1,self.T1,temp               

        def test(self): 
            self.load(self.X_test,self.T_test,self.test_range_id,self.T_labels_test,offset_x=16,offset_y=16)
            test_range_id_next = self.test_range_id+1 if self.test_range_id != self.test_range[1]-1 else self.test_range[0]
            temp = self.test_range_id           
            self.test_range_id = test_range_id_next
            return self.X_test,self.T_test,self.T_labels_test,temp 

        def load(self,X,T,train_range_id,T_labels=None,offset_x=0,offset_y=0): 
            # print train_range_id
            T[:]=0    
            fo = open(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_'+str(train_range_id), 'rb')
            dict = cPickle.load(fo)
            fo.close()
            # print dict['data'].T.shape
            temp = dict['data'].T
            
            if self.bias:
                temp_mean=temp.mean(axis=1)
                temp_std = (temp.var(1)+10)**.5
                temp = (temp-temp_mean[:, np.newaxis])/temp_std[:, np.newaxis]
                temp -= self.M
                temp = np.dot(temp,self.P)

            temp = temp.reshape(10000,3,32,32)
            for i in xrange(10000):        
                if np.random.rand()>.5:
                    # print i,
                    for j in xrange(3):
                        temp[i,j,:,:]=np.fliplr(temp[i,j,:,:])   
            # print X.shape,temp.shape
            X[:]=temp[:,:,offset_x:self.crop_size+offset_x,offset_y:self.crop_size+offset_y]
            if T_labels: T_labels[:]= dict['labels']
            for i in range(10000):
                T[i,dict['labels'][i]]= 1
