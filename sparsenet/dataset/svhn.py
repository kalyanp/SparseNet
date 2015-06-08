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




class SVHN(object):

    @staticmethod
    def test(want_dense=False):
        # X,T,X_test,T_test,T_train_labels,T_labels = dataset.load_svhn_contrast_extra(n=21,k=.01,want_dense=False)
        # dataset.load_svhn_contrast_extra_make()
        # X,T,X_test,T_test,T_train_labels,T_labels = SVHN.load_contrast(n=13,k=.01,want_dense=False)
        X,T,X_test,T_test,_,_,lst = SVHN.semi(N=20)
        # X,T,X_test,T_test,T_train_labels,T_labels = dataset.load_svhn_contrast(n=13,want_dense=False)
        # X,T,X_test,T_test,T_train_labels,T_labels = dataset.load_svhn_pylearn2(want_dense=False)
        # X,T,X_test,T_test,T_train_labels,T_labels = dataset.load_svhn_torch(want_dense=False)
        # n = 0
        # print X.shape
        dp = nn.dp_ram(X,T,data_batch=20)        
        x,t,_ = dp.train()
        nn.show_images(x,(2,10))
        nn.show()

    @staticmethod
    def semi(N):
        k = N/10
        # print k
        X,T,X_test,T_test,T_train_labels,T_labels = SVHN.load_contrast()
      
        count = [0]*10
        lst = []
        index = 0
        while len(lst)<N:
            label = int(T_train_labels[index])        
            if count[label] < k: 
                lst.append(index)
                count[label]+=1
            index += 1
        # print lst

        X_semi = X[lst,:,:,:]
        T_semi = T[lst]

        # if want_dense==False:
        #     X_semi = X_semi.reshape(10*k,1,28,28)
        #     X_test = X_test.reshape(10000,1,28,28)

        # print count
        return X_semi,T_semi,X_test,T_test,None,None,lst


    @staticmethod
    def load_pylearn2(want_dense=False):

        f_train = open_file(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/mirzamom/splitted_train_32x32.h5", mode = "r")
        X = np.array(f_train.root.Data.X).reshape(-1,3,32,32)[:70000,:,:,:]
     
        T_train_labels = None
        T = f_train.root.Data.y[:10000]


        f_test = open_file(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/mirzamom/test_32x32.h5", mode = "r")
        X_test = np.array(f_test.root.Data.X).reshape(-1,3,32,32)[:10000,:,:,:]
        T_labels = None
        T_test = f_test.root.Data.y[:10000]

        if want_dense:
            X = X.reshape(70000,3072)
            X_test = X_test.reshape(10000,3072)

        return X,T,X_test,T_test,T_train_labels,T_labels


    @staticmethod
    def load_torch(want_dense=False,want_bw=False):

        myFile = h5py.File(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/SVHN/torch/svhn_rgb_21.h5', 'r')
        X = np.array(myFile['X'])[:70000,:,:,:]

        T_train_labels = np.array(myFile['T_train_labels'])[:70000]
        T_train_labels = T_train_labels%10
        
        T = np.zeros((70000,10))
        for i in range(70000):
            T[i,T_train_labels[i]]= 1

        X_test = np.array(myFile['X_test'])[:10000,:,:,:]
        
        T_labels = np.array(myFile['T_labels'])[:10000]
        T_labels = T_labels%10
        
        T_test = np.zeros((10000,10))
        for i in range(10000):
            T_test[i,T_labels[i]]= 1
        
        if want_dense:
            X = X.reshape(70000,3072)
            X_test = X_test.reshape(10000,3072)

        return X,T,X_test,T_test,T_train_labels,T_labels


    @staticmethod
    def load_extra_torch():

        myFile = h5py.File(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/SVHN/torch/svhn_extra_rgb_13.h5', 'r')
        # myFile = h5py.File('svhn_old.h5', 'r')
        X = np.array(myFile['X'])
     
        # temp = X[10000:10900,:,:,:]
        # nn.show_images(temp,(30,30)); plt.show()


        T_train_labels = np.array(myFile['T_train_labels'])
        T_train_labels = T_train_labels%10

        # print T_train_labels[100000:1000010]

        print "dataset loaded"
        T = np.zeros((600000,10))
        for i in range(600000):
            # if i%10000==0:
                # print i,T_train_labels[i:i+10]
            T[i,T_train_labels[i]]= 1

        X_test = np.array(myFile['X_test'])[:10000,:,:,:]
        
        T_labels = np.array(myFile['T_labels'])[:10000]
        T_labels = T_labels%10
        
        T_test = np.zeros((10000,10))
        for i in range(10000):
            T_test[i,T_labels[i]]= 1
        
        # if want_bw:
        #     X = X[:,:1,:,:].reshape(70000,1024)
        #     X_test = X_test[:,:1,:,:].reshape(70000,1024)
        #     return X,T,X_test,T_test,T_train_labels,T_labels

        # if want_dense:
        #     X = X.reshape(70000,3072)
        #     X_test = X_test.reshape(10000,3072)

        return X,T,X_test,T_test,T_train_labels,T_labels


    @staticmethod
    def load(want_mean = False,want_dense=False):

        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/train_32x32.mat")
        X = f['X'].astype("float64")
        X = np.swapaxes(X,0,3)
        X = np.swapaxes(X,1,2)
        X = np.swapaxes(X,2,3)
        X = X[:70000,:,:,:]
        
        T_train_labels = f['y'].ravel()[:70000]%10

        T = np.zeros((70000,10))
        for i in range(70000):
            T[i,T_train_labels[i]]= 1

        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/test_32x32.mat")
        X_test = f['X'].astype("float64")
        X_test = np.swapaxes(X_test,0,3)
        X_test = np.swapaxes(X_test,1,2)
        X_test = np.swapaxes(X_test,2,3)
        # print X_test.shape
        X_test = X_test[:20000,:,:,:]

        T_labels = f['y'].ravel()[:20000]%10

        T_test = np.zeros((20000,10))
        for i in range(20000):
            T_test[i,T_labels[i]]= 1   


        if want_mean:
            X_mean= X.mean(0)
            X_std = X.std(0)
            X = (X-X_mean)/X_std
            X_test = (X_test-X_mean)/X_std


        if want_dense:
            X = X.reshape(70000,3072)
            X_test = X_test.reshape(20000,3072)

        return X,T,X_test,T_test,T_train_labels,T_labels

    @staticmethod
    def load_contrast(n=21,k=.01,want_dense=False):

        X,T,X_test,T_test,T_train_labels,T_labels = SVHN.load(want_mean=False,want_dense=True)

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

        X = X.reshape(70000,3,32,32)
        X_test = X_test.reshape(20000,3,32,32)

        filters = nn.gaussian2D(n,k)

        X_cn = nn.zeros(X.shape)
        X_test_cn = nn.zeros(X_test.shape)

        for i in xrange(7):
            X_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X[10000*i:10000*(i+1),:,:,:], filters)
        for i in xrange(2):
            X_test_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X_test[10000*i:10000*(i+1),:,:,:], filters)

        if want_dense:
            X_cn = X_cn.reshape(70000,3072)
            X_test_cn = X_test_cn.reshape(20000,3072)

        return X_cn,T,X_test_cn,T_test,T_train_labels,T_labels



    @staticmethod
    def load_extra(want_mean = False,want_dense=False):
        X_total = np.zeros((600000,3,32,32))
        T_train_labels_total = np.zeros(600000)
        T_total = np.zeros((600000,10))

        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/train_32x32.mat")
        X = f['X'].astype("float64")
        X = np.swapaxes(X,0,3)
        X = np.swapaxes(X,1,2)
        X = np.swapaxes(X,2,3)
        X_total[:70000,:,:,:] = X[:70000,:,:,:]
        
        T_train_labels_total[:70000] = f['y'].ravel()[:70000]%10
        print "Train Loaded."

        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/extra_32x32.mat")
        X = f['X'].astype("float32")
        X = np.swapaxes(X,0,3)
        X = np.swapaxes(X,1,2)
        X = np.swapaxes(X,2,3)
        X_total[70000:600000,:,:,:] = X[:530000,:,:,:]
        
        T_train_labels_total[70000:600000] = f['y'].ravel()[:530000]%10

        print "Extra Loaded."

        T_total = np.zeros((600000,10))
        for i in range(600000):
            T_total[i,T_train_labels_total[i]]= 1




        f = scipy.io.loadmat(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/test_32x32.mat")
        X_test = f['X'].astype("float32")
        X_test = np.swapaxes(X_test,0,3)
        X_test = np.swapaxes(X_test,1,2)
        X_test = np.swapaxes(X_test,2,3)
        X_test = X_test[:20000,:,:,:]

        T_labels = f['y'].ravel()[:20000]%10

        T_test = np.zeros((20000,10))
        for i in range(20000):
            T_test[i,T_labels[i]]= 1   

        print "Test Loaded."

        if want_mean:
            X_mean= X_total.mean(0)
            X_std = X_total.std(0)
            X_total = (X_total-X_mean)/X_std
            X_test = (X_test-X_mean)/X_std

        if want_dense:
            X_total = X_total.reshape(600000,3072)
            X_test = X_test.reshape(20000,3072)

        return X_total,T_total,X_test,T_test,T_train_labels_total,T_labels


    @staticmethod
    def load_contrast_extra(n=21,k=.01):

        X,T,X_test,T_test,T_train_labels,T_labels = SVHN.load_extra(want_mean=False,want_dense=True)
        print "Contrast Normalization."

        X_mean=X.mean(axis=1)
        X_test_mean=X_test.mean(axis=1)

        X_std = X.std(1)
        X_test_std = X_test.std(1)

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]
        X_test = (X_test-X_test_mean[:, np.newaxis])/X_test_std[:, np.newaxis]

        X = X.reshape(600000,3,32,32)
        X_test = X_test.reshape(20000,3,32,32)

        filters = nn.gaussian2D(n,k)

        X_cn = np.zeros(X.shape)
        X_test_cn = np.zeros(X_test.shape)
        
        for i in xrange(60):
            if i%10==0: print i,
            X_cn[i*10000:(i+1)*10000,:,:,:] = nn.SpatialContrast(X[i*10000:(i+1)*10000,:,:,:], filters)        
        for i in xrange(2):
            X_test_cn[10000*i:10000*(i+1),:,:,:] = nn.SpatialContrast(X_test[10000*i:10000*(i+1),:,:,:], filters)


        h5f = h5py.File('/media/nas/PSI-Share-no-backup/Ali/Dataset/SVHN/extra_contrast21.01_new.h5', 'w')
        print "file created."
        h5f.create_dataset('X', data=X_cn)
        h5f.create_dataset('T', data=T)
        h5f.create_dataset('T_train_labels', data=T_train_labels)
        print "train done."
        h5f.create_dataset('X_test', data=X_test_cn)
        h5f.create_dataset('T_test', data=T_test)
        h5f.create_dataset('T_labels', data=T_labels)

        return X_cn,T,X_test_cn,T_test,T_train_labels,T_labels


    # @staticmethod
    # def load_file_svhn_contrast_extra(want_dense=False):

    #     myFile = h5py.File(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/SVHN/extra_contrast21.01_new.h5', 'r')

    #     X = np.array(myFile['X'])
    #     T = np.array(myFile['T'])
    #     X_test = np.array(myFile['X_test'])
    #     T_test = np.array(myFile['T_test'])
    #     T_train_labels = np.array(myFile['T_train_labels'])
    #     T_labels = np.array(myFile['T_labels'])

    #     if want_dense:
    #         X = X_cn.reshape(600000,3072)
    #         X_test = X_test_cn.reshape(10000,3072)

    #     return X,T,X_test,T_test,T_train_labels,T_labels    

    @staticmethod
    def load_file_svhn_contrast_extra():

        myFile = h5py.File(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/SVHN/extra_contrast21.01_new.h5', 'r')

        X = myFile['X']
        T = myFile['T']
        X_test = myFile['X_test']
        T_test = myFile['T_test']
        T_train_labels = myFile['T_train_labels']
        T_labels = myFile['T_labels']

        # if want_dense:
        #     X = X_cn.reshape(600000,3072)
        #     X_test = X_test_cn.reshape(10000,3072)

        return X,T,X_test,T_test,T_train_labels,T_labels    

    @staticmethod
    def MP(bias):

        dp = data_provider_svhn(train_range = [0,100],
                                test_range = [1000,1001],
                                mini_batch = 128)

        X = np.zeros((61440,3,32,32))
        for i in xrange(20):
            x,t_label,id = dp.train()
            print i,x.max()
            X[i*3072:(i+1)*3072,:,:,:]=x

        X = X.reshape(61440,3072)
        X_mean=X.mean(axis=1)
        X_std = (X.var(1)+10)**.5

        X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis]

        def cov(X):
            X_mean = X.mean(axis=0)
            X -= X_mean
            return np.dot(X.T,X)/(1.0*X.shape[0]-1)

        M = X.mean(axis=0)
        X -= M

        sigma = cov(X)
        u,s,v=np.linalg.svd(sigma)
        P = np.dot(np.dot(u,np.diag(np.sqrt(1./(s+bias)))),u.T)
        return M,P    



    @staticmethod
    class data_provider:

    # zerooooooooooooooooooooooooooooooooooooooooooooooooooo

        def __init__(self,train_range=None,test_range=None,mini_batch = None,want_auto = False,bias=None,mean=False):
            self.bias = bias
            self.mean = mean
            if bias:
                # print nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/CIFAR10/MP_"+str(bias)
                try:
                    f = np.load(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/MP_"+str(bias)+".npz")
                    self.M = f['M']; self.P = f['P'];
                except:                
                    print "Started computing M and P for bias=",bias
                    self.M,self.P = svhn_MP(bias = bias)
                    print "M and P saved to NAS for bias=",bias
                    np.savez(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/MP_"+str(bias),M=self.M,P=self.P)                
            elif mean: 
                f = np.load(nn.nas_address()+"/PSI-Share-no-backup/Ali/Dataset/SVHN/mean.npz")            
                self.M = f['svhn_mean']
                self.std = f['svhn_std']


            self.want_auto = want_auto

            self.start = True
            self.switch = 0
            self.train_range = train_range
            self.test_range = test_range
            assert train_range!=None
            self.train_range_id = self.train_range[0]
            self.test_range_id = self.test_range[0]
            self.mini_batch = mini_batch

            shared_array_base_X0 = multiprocessing.Array(ctypes.c_double, 3*32*32*3072)
            shared_array_X0 = np.ctypeslib.as_array(shared_array_base_X0.get_obj())
            self.X0 = shared_array_X0.reshape(3072,3,32,32)

            shared_array_base_X1 = multiprocessing.Array(ctypes.c_double, 3*32*32*3072)
            shared_array_X1 = np.ctypeslib.as_array(shared_array_base_X1.get_obj())
            self.X1 = shared_array_X1.reshape(3072,3,32,32)

            shared_array_base_T0 = multiprocessing.Array(ctypes.c_double, 3072*10)
            shared_array_T0 = np.ctypeslib.as_array(shared_array_base_T0.get_obj())
            self.T0 = shared_array_T0.reshape(3072,10)

            shared_array_base_T1 = multiprocessing.Array(ctypes.c_double, 3072*10)
            shared_array_T1 = np.ctypeslib.as_array(shared_array_base_T1.get_obj())
            self.T1 = shared_array_T1.reshape(3072,10)

            if test_range!=None:
                shared_array_base_X_test = multiprocessing.Array(ctypes.c_double, 3*32*32*3072)
                shared_array_X_test = np.ctypeslib.as_array(shared_array_base_X_test.get_obj())
                self.X_test = shared_array_X_test.reshape(3072,3,32,32)
                assert self.X_test.base.base is shared_array_base_X_test.get_obj()  

                shared_array_base_T_test = multiprocessing.Array(ctypes.c_double, 3072*10)
                shared_array_T_test = np.ctypeslib.as_array(shared_array_base_T_test.get_obj())
                self.T_test = shared_array_T_test.reshape(3072,10)
                assert self.T_test.base.base is shared_array_base_T_test.get_obj()

                shared_array_base_T_labels_test = multiprocessing.Array(ctypes.c_double, 3072)
                shared_array_T_labels_test = np.ctypeslib.as_array(shared_array_base_T_labels_test.get_obj())
                self.T_labels_test = shared_array_T_labels_test.reshape(3072)
                assert self.T_labels_test.base.base is shared_array_base_T_labels_test.get_obj()                                              
         

        def __len__(self): return len(self.train_range) 

        def train(self): 
            if self.start:              
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,self.train_range[0],None))
                self.p_load.start()
                self.p_load.join()
                self.start = False
            else: self.p_load.join()

            if self.switch == 0:
                train_range_id_next = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]  
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X1,self.T1,train_range_id_next,None))
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
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,train_range_id_next,None))
                self.p_load.start()
                self.switch = 0
                temp = self.train_range_id           
                self.train_range_id = train_range_id_next
                if self.want_auto:                                                                               
                    return self.X1,self.X1.copy(),temp   
                else:
                    return self.X1,self.T1,temp               

        def test(self): 
            self.load(self.X_test,self.T_test,self.test_range_id,self.T_labels_test)
            test_range_id_next = self.test_range_id+1 if self.test_range_id != self.test_range[1]-1 else self.test_range[0]
            temp = self.test_range_id           
            self.test_range_id = test_range_id_next
            return self.X_test,self.T_test,self.T_labels_test,temp 

        def load(self,X,T,train_range_id,T_labels=None): 
            T[:]=0    
            fo = np.load(nn.nas_address()+'/PSI-Share-no-backup/Ali/Dataset/SVHN/batches/data_batch_'+str(train_range_id)+'.npz')
           
            if self.bias:
                temp = fo['X'].reshape(3072,32*32*3)
                temp_mean=temp.mean(axis=1)
                temp_std = (temp.var(1)+10)**.5
                temp = (temp-temp_mean[:, np.newaxis])/temp_std[:, np.newaxis]
                temp -= self.M
                temp = np.dot(temp,self.P)
                X[:] = temp.reshape(3072,3,32,32)
            elif self.mean: 
                X[:] = (fo['X']-self.M)/self.std
            else: X[:] = fo['X']

            for i in range(3072):
                T[i,fo['T'][i]%10]= 1
            
            if T_labels!=None: T_labels[:] = fo['T']%10


if __name__ == "__main__":
    nn.set_backend("gnumpy",board=0)
    # print nn.backend
    SVHN.test()