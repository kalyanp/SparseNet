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

class ImageNet():
    @staticmethod
    class data_provider:

        def __init__(self,train_range=None,test_range=None,mini_batch = None,mode="alex",crop_size = 224,want_auto = False, want_whiten = False, p=None):
            self.p = p
            if p:
                f=np.load(nas_address()+"/PSI-Share-no-backup/Ali/Dataset/ImageNet/Other/P_imagenet_"+str(p)+".npz"); 
            # self.M = f['M'] 
                self.P = f['P'] 
            self.want_whiten = want_whiten

            self.mode = mode
            self.num_threads = 3
            assert self.num_threads==3
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
            self.crop_offset = (256-crop_size)

            # self.address = address
            shared_array_base_X0 = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*3072)
            shared_array_X0 = np.ctypeslib.as_array(shared_array_base_X0.get_obj())
            self.X0 = shared_array_X0.reshape(3072,3,self.crop_size,self.crop_size)
            assert self.X0.base.base is shared_array_base_X0.get_obj()

            shared_array_base_X1 = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*3072)
            shared_array_X1 = np.ctypeslib.as_array(shared_array_base_X1.get_obj())
            self.X1 = shared_array_X1.reshape(3072,3,self.crop_size,self.crop_size)
            assert self.X1.base.base is shared_array_base_X1.get_obj()     

            shared_array_base_T0 = multiprocessing.Array(ctypes.c_double, 3072*1000)
            shared_array_T0 = np.ctypeslib.as_array(shared_array_base_T0.get_obj())
            self.T0 = shared_array_T0.reshape(3072,1000)
            assert self.T0.base.base is shared_array_base_T0.get_obj()

            shared_array_base_T1 = multiprocessing.Array(ctypes.c_double, 3072*1000)
            shared_array_T1 = np.ctypeslib.as_array(shared_array_base_T1.get_obj())
            self.T1 = shared_array_T1.reshape(3072,1000)
            assert self.T1.base.base is shared_array_base_T1.get_obj()        

            if test_range!=None:
                shared_array_base_X_test = multiprocessing.Array(ctypes.c_double, 3*self.crop_size*self.crop_size*3072)
                shared_array_X_test = np.ctypeslib.as_array(shared_array_base_X_test.get_obj())
                self.X_test = shared_array_X_test.reshape(3072,3,self.crop_size,self.crop_size)
                assert self.X_test.base.base is shared_array_base_X_test.get_obj()  

                shared_array_base_T_test = multiprocessing.Array(ctypes.c_double, 3072*1000)
                shared_array_T_test = np.ctypeslib.as_array(shared_array_base_T_test.get_obj())
                self.T_test = shared_array_T_test.reshape(3072,1000)
                assert self.T_test.base.base is shared_array_base_T_test.get_obj()

                shared_array_base_T_labels_test = multiprocessing.Array(ctypes.c_double, 3072)
                shared_array_T_labels_test = np.ctypeslib.as_array(shared_array_base_T_labels_test.get_obj())
                self.T_labels_test = shared_array_T_labels_test.reshape(3072)
                assert self.T_labels_test.base.base is shared_array_base_T_labels_test.get_obj()                                              


            fo = open(nas_address()+'/PSI-Share-no-backup/Ali/Dataset/ImageNet/batches/batches.meta', 'rb')
            dict_meta = cPickle.load(fo)              
            fo.close()
            self.data_mean = dict_meta['data_mean'].reshape(3,256,256)

            if mode=="vgg":
                # BGR values should be subtracted: [103.939, 116.779, 123.68].            
                self.data_mean[0,:,:]=123.68
                self.data_mean[1,:,:]=116.779
                self.data_mean[2,:,:]=103.939

            self.meta_alex = scipy.io.loadmat(nas_address()+'/PSI-Share-no-backup/Ali/Dataset/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat') 
            assert mode in ("alex","vgg")
            if mode=="alex": 
                 
                # self.words = lambda k: self.meta_alex['synsets'][k][0][2][0]
                # self.wnid = lambda k: self.meta_alex['synsets'][k][0][1][0]
                self.words = self.words_alex
                self.wnid = self.wnid_alex
                self.map = lambda label: label if label<1000 else None
                self.map_r = lambda label: label if label<1000 else None

            if mode=="vgg":
                fo = open("/home/alireza/Dropbox/work/caffe/data/ilsvrc12/synsets.txt",'r')
                self.dict_vgg = {}
                self.lst_vgg = [None]*1000
                self.map_r = [None]*1000

                for i in xrange(1000):
                    wnid = fo.readline()[:-1]
                    self.dict_vgg[wnid]=i
                    self.lst_vgg[i]=wnid

                self.wnid = lambda label: self.lst_vgg[label]
                self.map = lambda label: self.dict_vgg[self.wnid_alex(label)]


                for i in xrange(1000):
                    self.map_r[self.map(i)]= i
                self.words = lambda label: self.words_alex(self.map_r[label])

                # for i in xrange(1000):
                    # print self.map(i),            

        def __len__(self): return len(self.train_range) 

        def words_alex(self,k):
            return self.meta_alex['synsets'][k][0][2][0]
        
        def wnid_alex(self,k):
            return self.meta_alex['synsets'][k][0][1][0]    

        def train(self): 
            if not self.start:  self.p_load.join()
            if self.start:
                offset_x=np.random.randint(0,self.crop_offset) 
                offset_y=np.random.randint(0,self.crop_offset)                
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,self.train_range[0],None,offset_x,offset_y))
                self.p_load.start()
                self.p_load.join()
                self.start = False

            if self.switch == 0:
                train_range_id_next = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]
                offset_x=np.random.randint(0,self.crop_offset) 
                offset_y=np.random.randint(0,self.crop_offset)    
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X1,self.T1,train_range_id_next,None,offset_x,offset_y))
                self.p_load.start()
                self.switch = 1
                temp = self.train_range_id
                self.train_range_id = train_range_id_next
                if self.want_auto:
                    if self.want_whiten:
                        X = self.X0.reshape(3072,3072)/255.0
                        # X_mean=X.mean(axis=1)
                        # X_std = (X.var(1)+10)**.5
                        # X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis] 
                        # X -= self.M    
                        X=np.dot(X,self.P) 
                        X = X.reshape(3072,3,32,32)                                  
                        return X,X.copy(),temp                       
                    else:
                        return self.X0,self.X0.copy(),temp   
                else:
                    return self.X0,self.T0,temp
            else: 
                train_range_id_next = self.train_range_id+1 if self.train_range_id != self.train_range[1]-1 else self.train_range[0]
                offset_x=np.random.randint(0,self.crop_offset) 
                offset_y=np.random.randint(0,self.crop_offset)            
                self.p_load = multiprocessing.Process(target=self.load, args=(self.X0,self.T0,train_range_id_next,None,offset_x,offset_y))
                self.p_load.start()
                self.switch = 0
                temp = self.train_range_id           
                self.train_range_id = train_range_id_next
                if self.want_auto:
                    if self.want_whiten:
                        X = self.X1.reshape(3072,3072)/255.0                  
                        # X_mean=X.mean(axis=1)
                        # X_std = (X.var(1)+10)**.5
                        # X = (X-X_mean[:, np.newaxis])/X_std[:, np.newaxis] 
                        # X -= self.M    
                        X=np.dot(X,self.P)        
                        X = X.reshape(3072,3,32,32)                                                                                 
                        return X,X.copy(),temp                       
                    else:
                        return self.X1,self.X1.copy(),temp   
                else:
                    return self.X1,self.T1,temp               

        # def synsets(self,address=None):
        #     fo = open("/home/alireza/Dropbox/ipython/caffe/data/ilsvrc12/synsets.txt",'r')
        #     self.map_vgg = [None]*1000
        #     self.map_alex = [None]*1000

        #     dict_alex = {}
        #     dict_vgg = {}

        #     for i in xrange(1000):
        #         wnid = fo.readline()[:-1]
        #         dict_vgg[wnid]=i
        #         dict_alex[self.wnid(i)]=i

        #     for i in xrange(1000):
        #         self.map_vgg[i]=dict_vgg[self.wnid(i)]
        #         self.map_alex[i]=i

        #     self.map = self.map_vgg



        def test(self): 
            self.load(self.X_test,self.T_test,self.test_range_id,self.T_labels_test,offset_x=16,offset_y=16)
            test_range_id_next = self.test_range_id+1 if self.test_range_id != self.test_range[1]-1 else self.test_range[0]
            temp = self.test_range_id           
            self.test_range_id = test_range_id_next
            return self.X_test,self.T_test,self.T_labels_test,temp 

        def load(self,X,T,train_range_id,T_labels=None,offset_x=0,offset_y=0): 
            T[:]=0    
            self.threads_list = [data_provider_imagenet.ThreadImageNetOpen(self,X,T,train_range_id,i,T_labels,offset_x,offset_y) for i in range(self.num_threads)]
            for th in self.threads_list:
                th.start()
            for th in self.threads_list:
                th.join() 

        @staticmethod
        class ThreadImageNetOpen(threading.Thread):
            def __init__(self, class_name, X, T, train_range_id, threadID, T_labels, offset_x, offset_y):
                threading.Thread.__init__(self)
                self.class_name = class_name
                self.offset_x = offset_x
                self.offset_y = offset_y
                self.X = X
                self.T = T
                self.T_labels = T_labels
                self.train_range_id = train_range_id
                self.threadID = threadID        
            def run(self):
                self.class_name.imagenet_open(self.X, self.T, self.train_range_id, self.threadID, self.T_labels, self.offset_x, self.offset_y)

        def imagenet_open(self,X,T,train_range_id,threadID,T_labels,offset_x,offset_y):
            fo = open(nas_address()+'/PSI-Share-no-backup/Ali/Dataset/ImageNet/batches/data_batch_'+str(train_range_id)+'/data_batch_'+str(train_range_id)+'.'+str(threadID), 'rb')
            dict = cPickle.load(fo)   
            for i in xrange(3072/self.num_threads):
                img_code = dict['data'][i]
                img_label = self.map(dict['labels'][i][0])
                # print dict['labels'][i][0],img_label,self.map(0),self.map(1),self.map

                nparr = np.fromstring(img_code, np.uint8)
                temp = (cv2.imdecode(nparr,1)[:,:,::-1].reshape(256**2,3).T.reshape(3,256,256)-self.data_mean)

                if self.mode=="vgg": temp = temp[::-1,:,:]  
                X[i+threadID*3072/self.num_threads,:,:,:] = temp[:,offset_x:self.crop_size+offset_x,offset_y:self.crop_size+offset_y]
                T[threadID*3072/self.num_threads+i,img_label] = 1.0
                if T_labels!=None: 
                    T_labels[threadID*3072/self.num_threads+i]=img_label
            # print "Process ",threadID," ended."                 

