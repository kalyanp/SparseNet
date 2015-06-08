import numpy as np
import gnumpy as gp
from datetime import datetime
import pylab as plt
import gnumpy.cudamat_j as ConvNet
import cPickle
import cv2
import scipy.io
import time
import threading
import multiprocessing
import ctypes
import time
import gnumpy.libcudnn as libcudnn
import os
import Image
from scipy.stats import norm
import h5py


cudamat = gp.cmat

from nn_utils_conv import CudnnBackend, CudaconvnetBackend, NumpyConvBackend, CudaconvnetBackendReversed, NumpyConvBackendReversed




class GnumpyBackend(object):

    @staticmethod
    def hard_reshape(A,*shape):    
        # assert np.prod(A.shape)==np.prod(shape)
        A.shape = shape[0]


    # @staticmethod    
    # def garray(x):
    #     if type(x)==gp.garray: return x
    #     else: 
    #         # print x.dtype
    #         assert x.dtype != "float32"
    #         return gp.garray(x.astype("float32"))



    @staticmethod    
    def k_sparsity_mask(x,k,axis):
        x = x.as_numpy_array()  
        c=np.zeros(x.shape)
        b=np.argsort(x,kind='quicksort',axis=axis)
        if axis==0: loc=b[-k:,:].T.flatten(),np.repeat(np.arange(x.shape[1]),k)
        elif axis==1: loc=np.repeat(np.arange(x.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        return garray(c)


    @staticmethod
    def dropout(A,B,rate,outA,outB):
        #dropout patterns always start the same.
        if outA == None: outA = A
        if outB == None: outB = B
        # dt = datetime.now()
        # gp.seed_rand(dt.microsecond)         
        if B != None:
            cudamat.dropout(A._base_shaped(1),B._base_shaped(1),rate,
                            targetA=outA._base_shaped(1),
                            targetB=outB._base_shaped(1))
        else:
            cudamat.dropout(A._base_shaped(1),None,rate,
                            targetA=outA._base_shaped(1),
                            targetB=None)
        # print A[:5,:5]


    @staticmethod
    def arange(k):
        return garray(np.arange(k))








    # @staticmethod
    # def ConvUp_single(a_gp, f_gp,moduleStride, paddingStart):
    #     assert f_gp.shape[3]==1
    #     f_16 = gp.zeros((f_gp.shape[0],f_gp.shape[1],f_gp.shape[2],16))
    #     f_16[:,:,:,:1]=f_gp
    #     q_gp = ConvNet.convUp(a_gp, f_16 ,moduleStride = moduleStride, paddingStart = paddingStart)
    #     return q_gp[:1,:,:,:]

    @staticmethod
    def argsort(x):
        return gp.garray(np.argsort(x.as_numpy_array()))

    @staticmethod
    def l2_normalize(w):
        l2=gp.sum(w**2,axis=0)**(1./2)
        w[:]=w/l2

    @staticmethod
    def bitwise_or(x,y):
        return x | y
    
    @staticmethod
    def threshold_mask_hard(x,k,mask=None,dropout=None):
        if type(x)==gp.garray: x_ = x.as_numpy_array()
        else: x_ = x
        if dropout!=None:
            dropout_mask =  (np.random.rand(x_.shape[0])>(1-dropout))
        c=np.zeros(x_.shape)
        if k==1: 
            loc=np.arange(x_.shape[0]),x_.argmax(1)
        else: 
            b=np.argsort(x_,kind='quicksort',axis=1)
            loc=np.repeat(np.arange(x_.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if type(x)==gp.garray:
            if dropout!=None: return gp.garray(dropout_mask[:,newaxis]*c)
            else: return gp.garray(c)
        else:
            if dropout!=None: return dropout_mask[:,np.newaxis]*c
            else: return c    

    @staticmethod
    def threshold_mask_hard_groups(x,k,num_groups=None):
        if not num_groups: num_groups = x.shape[1]
        if type(x)==gp.garray: x_ = x.as_numpy_array().T
        else: x_=x.copy().T 
        assert x_.shape[0] % num_groups == 0
        num_rows = x_.shape[0] / num_groups        
        shape = x_.shape       
        k = k*num_rows
        x_ = x_.reshape(num_groups,-1)
        c=np.zeros(x_.shape)
        b=np.argsort(x_,kind='quicksort',axis=1)
        loc=np.repeat(np.arange(x_.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if type(x)==gp.garray: return gp.garray(c.reshape(shape).T)
        else: return c.reshape(shape).T   
   
    @staticmethod
    def threshold_mask_soft(x,k,dropout=None):
        b=k*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        if dropout==None: return (x>std_matrix)
        return (x>std_matrix)*(gp.rand(x.shape)>(1-dropout))
    
    @staticmethod
    def mask(x,dropout=1):
        return (gp.rand(x.shape)>(1-dropout))    

    @staticmethod
    def empty(shape):
        if type(shape)!=tuple: return gp.empty(shape)
        return gp.empty(shape)

    @staticmethod
    def zeros(shape,dtype=None):
        if type(shape)!=tuple: return gp.zeros(shape)
        return gp.zeros(shape)
    
    @staticmethod
    def ones(shape):
        if type(shape)!=tuple: return gp.ones(shape)
        return gp.ones(shape)    
    

    
    @staticmethod
    def sample(x):
        return GnumpyBackend.rand(x.shape)<x    


    @staticmethod
    def rand_binary(shape,dtype):    return gp.rand(*shape)>.5
    
    @staticmethod
    def randn(shape):
        out = gp.empty(shape)
        GnumpyBackend.fill_randn(out)
        return out 

    @staticmethod
    def rand(shape):
        out = empty(shape)
        GnumpyBackend.fill_rand(out)
        return out 
    # @staticmethod
    # def rand(shape):    
    #     return garray(NumpyBackend.rand(shape))

        # if type(shape)!=tuple: return gp.randn(shape)
        # return gp.randn(shape)

    # @staticmethod
    # def randn(shape,dtype):    
    #     out = GnumpyBackend.empty(shape)
    #     out._base.fill_with_randn()
    #     return out

    # @staticmethod
    # def array(A,dtype):  return gp.garray(A)
    

    @staticmethod
    def sigmoid(A,out=None,dout=None):
        if out==None:
            out = empty(A.shape)
        logistic(A,out)  
        if dout!=None: 
            logistic_deriv(out,dout)
        return out
    
    
    @staticmethod
    def relu_prime(x): return gp.garray(x>0)
    
    @staticmethod
    def relu_squared(x): return gp.garray(x>0)*(x**2)

    @staticmethod
    def relu_squared_prime(x): return gp.garray(x>0)*(2*x)

    @staticmethod
    def relu_sigma_1(x): 
        b=2*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        return ((x-std_matrix)>0)*(x-std_matrix)+((x+std_matrix)<0)*(x+std_matrix)
        
    @staticmethod
    def relu_sigma_1_prime(x): 
        b=2*gp.std(x,axis=1)[:,gp.newaxis]
        std_matrix=gp.dot(b,gp.ones((1,x.shape[1])))
        return (x>std_matrix)+(x<-std_matrix)
    
    @staticmethod
    def relu_5(x): return gp.garray(x>.05)*(x-.05)+gp.garray(x<-.05)*(x+.05)
    
    @staticmethod
    def relu_5_prime(x): return gp.garray(x>0.05)+gp.garray(x<-.05)
    
    @staticmethod
    def softmax_old(x):
        y=gp.max(x,axis=1)[:,gp.newaxis]
        logsumexp=y+gp.log(gp.sum((gp.exp(x-y)),axis=1))[:,gp.newaxis]
        return gp.exp(x-logsumexp)
    
    
    @staticmethod    
    def softmax(A,out,dout):
        if out == None: out = GnumpyBackend.empty(A.shape)
        # First pre-allocate enough memory to accumulate denominator of each sample
        maxval = denom = GnumpyBackend.empty((A.shape[0],1))
        # Then compute logsum softmax (subtract off maximum value)
        GnumpyBackend.max(A,axis=1,out=maxval)
        GnumpyBackend.subtract(A,maxval,out=out)
        GnumpyBackend.exp(out,out=out)
        GnumpyBackend.sum(out,axis=1,out=denom)
        GnumpyBackend.reciprocal(denom,out=denom)
        GnumpyBackend.multiply(out,denom,out=out) 
        return out 

    
    @staticmethod
    def linear(A,out,dout): 
        if not (out is A):
            out[:] = A[:]
        if dout != None:
            GnumpyBackend.iassign(dout,1)    
    
    
    @staticmethod
    def KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[gp.where(y<rho_target)]=rho_target*gp.ones(y[gp.where(y<rho_target)].shape)
        return rho_target*gp.log(rho_target/y)+(1-rho_target)*gp.log((1-rho_target)/(1-y))
    
    @staticmethod
    def d_KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[gp.where(y<rho_target)]=rho_target*gp.ones(y[gp.where(y<rho_target)].shape)
        return -rho_target/y+(1-rho_target)/(1-y)
    
    @staticmethod
    def exp_penalty(x,sigma): return x.shape[1]-((gp.exp(-x**2/sigma)).sum())/x.shape[0]
    
    @staticmethod
    def d_exp_penalty(x,sigma): return ((2*(1/sigma)*x*gp.exp(-x**2/sigma)))

    @staticmethod
    def dot(A,B,out):
        if out == None:
            out = gp.empty((A.shape[0],B.shape[1]),dtype='float32')
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        return out

    @staticmethod
    def dot_tn(A,B,out):
        A._base.mat.is_trans = not A._base.mat.is_trans #############ali   
        if out == None:
            out = gp.empty((A.shape[1],B.shape[1]),dtype=A.dtype)
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        A._base.mat.is_trans = not A._base.mat.is_trans ###############ali   
        return out

    @staticmethod
    def dot_nt(A,B,out):
        # Using B._base_as_2d().T does not work; cudamat returns dimensionality error
        B._base.mat.is_trans = not B._base.mat.is_trans 
        if out == None:
            out = gp.empty((A.shape[0],B.shape[0]),dtype='float32')
        cudamat.dot(B._base_as_2d(),A._base_as_2d(),target=out._base_as_2d())
        B._base.mat.is_trans = not B._base.mat.is_trans
        return out


    @staticmethod
    def square(A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        cudamat.square(A._base_as_row(),target=out._base_as_row())
        return out

    @staticmethod
    def _unary(func,A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        func(A._base_as_row(),target=out._base_as_row())
        return out

    @staticmethod
    def logistic(A,out): return GnumpyBackend._unary(cudamat.sigmoid,A,out)

    @staticmethod
    def tanh_func(A,out):     return GnumpyBackend._unary(cudamat.tanh,A,out)

    @staticmethod
    def sqrt(A,out):     return GnumpyBackend._unary(cudamat.sqrt,A,out)

    @staticmethod
    def exp(A,out):      return GnumpyBackend._unary(cudamat.exp,A,out)

    @staticmethod
    def log(A,out):      return GnumpyBackend._unary(cudamat.log,A,out)

    @staticmethod
    def abs(A,out):      return GnumpyBackend._unary(cudamat.abs,A,out)

    @staticmethod
    def sign(A,out):     return GnumpyBackend._unary(cudamat.CUDAMatrix.sign,A,out)

    @staticmethod
    def relu(A,out,dout):
        if out!=None and dout!=None:
            cudamat.relu(A._base_as_row(),
                         target =( out._base_as_row() if  out != None else None),
                         dtarget=(dout._base_as_row() if dout != None else None))
        elif out==None and dout==None:
            return gp.garray(A>0)*A


    @staticmethod
    def logistic_deriv(A,out): return GnumpyBackend._unary(cudamat.sigmoid_deriv,A,out)

    @staticmethod
    def tanh_deriv(A,out): return GnumpyBackend._unary(cudamat.tanh_deriv,A,out)


    @staticmethod
    def max(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            A._base_shaped(1).max(1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.max(A,axis)  # gnumpy has optimized max over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def min(A,axis,out):
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            A._base_shaped(1).min(1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.min(A,axis)  # gnumpy has optimized max over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def sum(A,axis,out):
        if axis == None: 
            temp = GnumpyBackend.sum(A,1,None)
            return GnumpyBackend.sum(temp,0,None)[0,0]
        if A.ndim == 2: 
            if out == None:
                out = gp.empty((A.shape[0],1) if axis == 1 else (1,A.shape[1]),dtype=A.dtype)
            cudamat.sum(A._base_shaped(1),1-axis,target=out._base_shaped(1))
            return out
        else:
            r = gp.sum(A,axis)  # gnumpy has optimized sum over 1D vectors, so use it
            if out != None:
                assert(out.size == 1)
                out[:] = r[:]
            return r

    @staticmethod
    def mean(A,axis,out):
        out = GnumpyBackend.sum(A,axis,out)
        GnumpyBackend.imul(out,1./A.shape[axis])
        return out

    @staticmethod
    def _add(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B): 
            A._base_shaped(1).add(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).add(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).add_col_vec(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).add_row_vec(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def add(A,B,out):
        # turn vec + matrix into matrix + vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._add(A,B,out)

    @staticmethod
    def add_nt(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_shaped(1).add_transpose(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def iadd(A,B):          GnumpyBackend._add(A,B,A)

    @staticmethod
    def iaddmul(A,B,alpha): A._base_shaped(1).add_mult(B._base_shaped(1),alpha)

    @staticmethod
    def iassign(A,B):       A._base_shaped(1).assign(B if np.isscalar(B) else B._base_shaped(1))

    @staticmethod
    def subtract(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B):
            A._base_shaped(1).subtract(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).subtract(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and (A.ndim == 1 or B.size == A.shape[1]):
            A._base_shaped(1).subtract_col_vec(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).subtract_row_vec(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def subtract_nt(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_shaped(1).subtract_transpose(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def isub(A,B):          GnumpyBackend.subtract(A,B,A)

    @staticmethod
    def _multiply(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B): 
            A._base_shaped(1).mult(B,target=out._base_shaped(1))
        elif B.shape == A.shape:
            A._base_shaped(1).mult(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[0] == 1) and B.size == A.shape[1]:
            A._base_shaped(1).mult_by_col(B._base_shaped(1),target=out._base_shaped(1))
        elif (B.ndim == 1 or B.shape[1] == 1) and B.size == A.shape[0]:
            A._base_shaped(1).mult_by_row(B._base_shaped(1),target=out._base_shaped(1))
        else:
            raise Exception("unhandled case")
        return out

    @staticmethod
    def multiply(A,B,out):
        # turn vec * matrix into matrix * vec
        if not np.isscalar(B) and (A.ndim < B.ndim or A.shape[0] < B.shape[0] or A.shape[1] < B.shape[1]):
            A,B = B,A
        return GnumpyBackend._multiply(A,B,out)

    @staticmethod
    def imul(A,B):         GnumpyBackend._multiply(A,B,A)

    @staticmethod
    def divide(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(B):         A._base_shaped(1).divide(B,target=out._base_shaped(1))
        elif A.shape == B.shape:   A._base_shaped(1).divide(B._base_shaped(1),target=out._base_shaped(1))
        else: raise NotImplementedError("broadcasted division not implemented by cudamat")
        return out

    @staticmethod
    def idiv(A,B):          GnumpyBackend.divide(A,B,A)

    @staticmethod
    def reciprocal(A,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        A._base_as_row().reciprocal(out._base_as_row())
        return out

    @staticmethod
    def transpose(A,out):
        if out == None:
            out = gp.empty((A.shape[1],A.shape[0]),dtype=A.dtype)
        A._base_shaped(1).transpose(out._base_shaped(1))
        return out

    @staticmethod
    def maximum(A,B,out):
        if out == None:
            out = gp.empty(A.shape,dtype=A.dtype)
        if np.isscalar(A) and not np.isscalar(B):
            A,B = B,A
        if np.isscalar(B): A._base_shaped(1).maximum(B,target=out._base_shaped(1))
        else:              A._base_shaped(1).maximum(B._base_shaped(1),target=out._base_shaped(1))
        return out

    @staticmethod
    def fill_randn(out): 
        # dt = datetime.now()
        # gp.seed_rand(dt.microsecond)        
        out._base.fill_with_randn()

    @staticmethod
    def fill_rand(out):  
        # dt = datetime.now()
        # gp.seed_rand(dt.microsecond)         
        out._base.fill_with_rand()
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

class NumpyBackend(object):
    

    @staticmethod    
    def k_sparsity_mask(x,k,axis):
        c=np.zeros(x.shape)
        b=np.argsort(x,kind='quicksort',axis=axis)
        if axis==0: loc=b[-k:,:].T.flatten(),np.repeat(np.arange(x.shape[1]),k)
        elif axis==1: loc=np.repeat(np.arange(x.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1.0
        return c

    @staticmethod
    def fill_randn(out): out[:] = randn(out.shape)

    @staticmethod
    def fill_rand(out): out[:] = rand(out.shape)

    @staticmethod
    def hard_reshape(A,shape):    
        A.reshape(shape)
        return A

    @staticmethod
    def relu(A,out,dout): 
        if out!=None:
            result = np.maximum(0,A,out=out)
            if dout != None:
                np.sign(out,out=dout)
            return result
        elif out==None and dout==None:
            return np.array(A>0,dtype=A.dtype)*A

    @staticmethod
    def tanh_deriv(A,out): 
        if out == None:
            out = empty(A.shape(),dtype=A.dtype)
        np.square(A,out=out)
        np.subtract(1,out,out=out)
        return out

    @staticmethod
    def sigmoid(A,out,dout):
        if out==None:
            out = empty(A.shape)
        logistic(A,out)  
        if dout!=None: 
            logistic_deriv(out,dout)       
        return out

        # logistic(A,out)
        # logistic_deriv(out,dout)


    @staticmethod
    def tanh_func(A,out):     return np.tanh(A,out=out)

    @staticmethod
    def logistic(A,out):
        if out == None: out = A.copy()
        else:           out[:] = A[:]
        out *= -1
        np.exp(out,out=out)
        out += 1
        NumpyBackend.reciprocal(out,out=out)
        return out   
    
    @staticmethod
    def logistic_deriv(A,out): return np.subtract(A,np.square(A),out=out)        

    @staticmethod
    def square(A,out):   return np.square(A,out=out)

    @staticmethod
    def dot(A,B,out):    return np.dot(A,B,out=out)

    @staticmethod
    def dot_tn(A,B,out): return np.dot(A.T,B,out=out)

    @staticmethod
    def dot_nt(A,B,out): return np.dot(A,B.T,out=out)


    @staticmethod
    def argsort(x):
        return np.argsort(x)

    @staticmethod
    def l2_normalize(w):
        l2=np.sum(w**2,axis=0)**(1./2)
        w[:]=w/l2

    @staticmethod
    def bitwise_or(x,y):
        return np.float64(np.int64(x) | np.int64(y))

    @staticmethod
    def abs(A,out):      return np.abs(A,out=out)

    @staticmethod
    def sign(A,out):     return np.sign(A,out=out)

    @staticmethod
    def threshold_mask_soft(x,k,mask=None,dropout=None):
        if mask!=None: x *= mask
        b=k*np.std(x,axis=1)[:,np.newaxis]
        std_matrix=np.dot(b,np.ones((1,x.shape[1])))
        if dropout==None: return np.array(x>std_matrix,dtype=x.dtype)
        return np.array(x>std_matrix,dtype=x.dtype)*np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)
    
    @staticmethod
    def mask(x,dropout=1):
        return np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)

    @staticmethod
    def threshold_mask_hard(x,k,mask=None,dropout=None):
        #if dropout!=None: x*=np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)
        if mask!=None: x *= mask
        c=np.zeros(x.shape)
        if dropout==-1: b=np.argsort(np.absolute(x),kind='quicksort',axis=1)
        else:           b=np.argsort(x,kind='quicksort',axis=1)
        loc=np.repeat(np.arange(x.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if (dropout==None or dropout==-1): return c
        if dropout==-2: return 1-c
        return c*np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)

    @staticmethod
    def apple_k_sparsity(A,k):
        #if dropout!=None: x*=np.array((np.random.rand(*x.shape)>(1-dropout)),dtype=x.dtype)
        if mask!=None: x *= mask
        c=np.zeros(x.shape)
        if dropout==-1: b=np.argsort(np.absolute(x),kind='quicksort',axis=1)
        else:           b=np.argsort(x,kind='quicksort',axis=1)
        loc=np.repeat(np.arange(x.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        return c
        
    @staticmethod
    def threshold_mask_hard_groups(x,k,num_groups=None):
        if not num_groups: num_groups = x.shape[1]
        if type(x)==gp.garray: x_ = x.as_numpy_array().T
        else: x_=x.copy().T 
        assert x_.shape[0] % num_groups == 0
        num_rows = x_.shape[0] / num_groups        
        shape = x_.shape       
        k = k*num_rows
        x_ = x_.reshape(num_groups,-1)
        c=np.zeros(x_.shape)
        b=np.argsort(x_,kind='quicksort',axis=1)
        loc=np.repeat(np.arange(x_.shape[0]),k),b[:,-k:].flatten()
        c[loc]=1
        if type(x)==gp.garray: return gp.garray(c.reshape(shape).T)
        else: return c.reshape(shape).T   

    @staticmethod
    def empty(shape): return np.empty(shape)

    @staticmethod
    def zeros(shape,dtype):
        if type(shape)!=tuple: return np.array(np.zeros(shape))
        return np.array(np.zeros(shape))
    
    @staticmethod
    def ones(shape): #print ones((2,2)) doesn't work!!
        if type(shape)!=tuple: return np.array(np.ones(shape))
        return np.ones(shape)
    
    @staticmethod
    def rand(shape):    return np.array(np.random.rand(*shape))

    @staticmethod
    def sample(x):
        return (NumpyBackend.rand(x.shape)<x).astype("float64")

    @staticmethod
    def rand_binary(shape,dtype):    return np.array(np.random.rand(*shape)>.5,dtype)

    @staticmethod
    def randn(shape):    
        if type(shape)!=tuple: return np.array(np.random.randn(shape))
        return np.array(np.random.randn(*shape))
    
    # @staticmethod
    # def array(A,dtype):  return np.array(A,dtype)
    
    # @staticmethod
    # def dot(A,B):    return np.dot(A,B)
    
    @staticmethod
    def exp(A,out):
        if out==None: return np.exp(A)
        return np.exp(A,out=out)
    
    @staticmethod
    def log(A,out):      
        if out==None:
            return np.log(A)
        else:
            return np.log(A,out=out)

    @staticmethod    
    def softmax(A,out,dout):
        if out==None: out = NumpyBackend.empty(A.shape)
        # First pre-allocate enough memory to accumulate denominator of each sample
        maxval = denom = NumpyBackend.empty((A.shape[0],1))
        # print denom.shape
        # Then compute logsum softmax (subtract off maximum value)
        NumpyBackend.max(A,axis=1,out=maxval)
        NumpyBackend.subtract(A,maxval,out=out)
        NumpyBackend.exp(out,out=out)
        NumpyBackend.sum(out,axis=1,out=denom)
        NumpyBackend.reciprocal(denom,out=denom)
        NumpyBackend.multiply(out,denom,out=out)   
        return out

   
    @staticmethod
    def linear(A,out,dout):
        if not (out is A):
            out[:] = A[:]
        if dout != None:
            NumpyBackend.iassign(dout,1)

    @staticmethod
    def KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[y<rho_target]=rho_target
        return rho_target*np.log(rho_target/y)+(1-rho_target)*np.log((1-rho_target)/(1-y))
    
    @staticmethod
    def d_KL(rho,rho_target,KL_flat): 
        y=rho.copy()
        if KL_flat: y[y<rho_target]=rho_target
        return -rho_target/y+(1-rho_target)/(1-y)
    
    @staticmethod
    def exp_penalty(x,sigma): return x.shape[1]-((np.exp(-x**2/sigma)).sum())/x.shape[0]
    
    @staticmethod
    def d_exp_penalty(x,sigma): return ((2*(1/sigma)*x*np.exp(-x**2/sigma)))

    @staticmethod
    def max(A,axis,out): return np.max(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def min(A,axis,out): return np.min(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def sum(A,axis,out): 
        if axis == None: return A.sum()
        if out==None:
            if axis == 0: 
                return np.sum(A,axis=axis)[np.newaxis,:]
            elif axis == 1: 
                return np.sum(A,axis=axis)[:,np.newaxis]              
        else:
            if axis == 0: 
                assert out.shape[0]==1
                out[:1,:] = np.sum(A,axis=axis)[np.newaxis,:]
            elif axis == 1: 
                assert out.shape[1]==1
                out[:,:1] = np.sum(A,axis=axis)[:,np.newaxis]                

    @staticmethod
    def mean(A,axis,out):return np.mean(A,axis=axis,out=out.ravel() if out != None else None)

    @staticmethod
    def add(A,B,out):       return np.add(A,B,out=out)

    @staticmethod
    def add_nt(A,B,out):  return np.add(A,B.transpose(),out=out)

    @staticmethod
    def iadd(A,B):          A += B

    @staticmethod
    def iaddmul(A,B,alpha): B *= alpha; A += B

    @staticmethod
    def iassign(A,B):       A[:] = B

    @staticmethod
    def subtract(A,B,out):  return np.subtract(A,B,out=out)

    @staticmethod
    def subtract_nt(A,B,out):  return np.subtract(A,B.transpose(),out=out)

    @staticmethod
    def isub(A,B):          A -= B

    @staticmethod
    def multiply(A,B,out):  return np.multiply(A,B,out=out)

    @staticmethod
    def imul(A,B):          A *= B

    @staticmethod
    def divide(A,B,out):    return np.divide(A,B,out=out)

    @staticmethod
    def idiv(A,B):          A /= B

    @staticmethod
    def reciprocal(A,out):  return np.divide(1.,A,out=out)

    @staticmethod
    def transpose(A,out):
        AT = A.transpose()
        if out != None:
            out[:] = AT
        return AT

    @staticmethod
    def dropout(A,B,rate,outA,outB):
        assert outA!=None
        # if outA == None: outA = A.copy()
        # if outB == None: outB = B.copy()
        mask = rand(A.shape)>rate
        multiply(A,mask,out=outA)
        if type(B)==np.ndarray: 
            assert outB!=None        
            multiply(B,mask,out=outB)
        # if B != None:
        #     multiply(B,mask,out=outB)    

    #@staticmethod
    #def newaxis(): return np.newaxis
    
    @staticmethod
    def relu_prime(x): return np.array(x>0,dtype=x.dtype)    

    @staticmethod
    def arange(k):
        return np.arange(k)
    
def percent(a):
    perc=zeros((10,10))
    for h in range(10):
        p_temp = percentile(a,10*(10-h),axis=1)
        for t in range(10):
            perc[h,t]= percentile(p_temp,10*(10-t))
    return perc

def max_norm(w,threshold):
    wSqr = nn.empty(w.shape)
    nn.square(w,wSqr)
    l2=nn.sum(wSqr,axis=0)**(1./2)
    index = (l2>threshold).ravel()
    w_ = w[:,index]/l2[:,index]*threshold
    w[:,index] = w_

    
def l2_norm(threshold,j):
    l2=np.sum(weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))**2,axis=0)**(1./2)
    weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l2>threshold]=weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l2>threshold]/l2[l2>threshold]*threshold

def l1_norm(threshold,j):
    l1=np.sum(np.abs(weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))),axis=0)
    weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l1>threshold]=weights[layers_sum[j]-num_w[j]:layers_sum[j]-layers[j+1]].reshape((layers[j],layers[j+1]))[:,l1>threshold]/l1[l1>threshold]*threshold
   
def eye(N,k=0): return backend_array(np.eye(N=N,k=k))
def dot(A,B,out=None):     return backend.dot(A,B,out)
def dot_tn(A,B,out=None):  return backend.dot_tn(A,B,out)
def dot_nt(A,B,out=None):  return backend.dot_nt(A,B,out)
def square(A,out=None):    return backend.square(A,out)   if not np.isscalar(A) else A*A
def logistic(A,out=None):  return backend.logistic(A,out) if not np.isscalar(A) else 1./(1+np.exp(-A))
def sqrt(A,out=None):      return backend.sqrt(A,out)     if not np.isscalar(A) else np.sqrt(A)
def exp(A,out=None):       return backend.exp(A,out)      if not np.isscalar(A) else np.exp(A)
def log(A,out=None):       return backend.log(A,out)      if not np.isscalar(A) else np.log(A)
def abs(A,out=None):       return backend.abs(A,out)      if not np.isscalar(A) else np.abs(A)
def sign(A,out=None):      return backend.sign(A,out)     if not np.isscalar(A) else np.sign(A)
def relu(A,out=None,dout=None): return backend.relu(A,out,dout) if not np.isscalar(A) else max(0,A)
def linear(A,out=None,dout=None):return backend.linear(A,out,dout)
def sigmoid(A,out=None,dout=None):return backend.sigmoid(A,out,dout)
def softmax(A,out=None,dout=None): return backend.softmax(A,out,dout)
def logistic_deriv(A,dout=None): return backend.logistic_deriv(A,dout) if not np.isscalar(A) else A*(1-A)

def tanh_deriv(A,out=None): return backend.tanh_deriv(A,out) if not np.isscalar(A) else 1-A**2
def tanh(A,out=None,dout=None):    
    tanh_func(A,out)
    tanh_deriv(out,dout)    
def tanh_func(A,out=None):      return backend.tanh_func(A,out)     if not np.isscalar(A) else np.tanh(A)

def empty(shape):return backend.empty(shape)
def zeros(shape,dtype="float64"):return backend.zeros(shape,dtype)
def ones(shape):return backend.ones(shape)
def rand(shape):
    assert seed_global
    return backend.rand(shape)
# def rand_binary(shape,dtype):return backend.rand_binary(shape,dtype)
def randn(shape):
    assert seed_global   
    return backend.randn(shape)
# def array(A,dtype):return backend.array(A,dtype)
# def dot(A,B):return backend.dot(A,B)
# def exp(A):return backend.exp(A)
# def log(A):return backend.log(A)
# def max(A,axis):return backend.max(A,axis)
def min(A,axis):return backend.min(A,axis)
# def sum(A,axis=None,out=None):return backend.sum(A,axis,out)
def mean(A,axis):return backend.mean(A,axis)
# def sigmoid(x):return backend.sigmoid(x)
# def sigmoid_prime(x):return backend.sigmoid_prime(x)
# def relu(x):return backend.relu(x)
def relu_prime(x):return backend.relu_prime(x)
# def softmax(x):return backend.softmax(x)
def softmax_grounded(b):return backend.softmax_grounded(b)
def linear_prime(x):return backend.linear_prime(x)
#def relu_truncated(x):return backend.relu_truncated(x)
#def relu_prime_truncated(x):return backend.relu_prime_truncated(x)
def KL(rho,rho_target,KL_flat):return backend.KL(rho,rho_target,KL_flat)
def d_KL(rho,rho_target,KL_flat):return backend.d_KL(rho,rho_target,KL_flat)
def exp_penalty(x,sigma):return backend.exp_penalty(x,sigma)
def d_exp_penalty(x,sigma):return backend.d_exp_penalty(x,sigma)
#def newaxis():return backend.newaxis()
def mask(x,dropout=1):return backend.mask(x,dropout)
def softmax_prime(x):return None
def softmax_old(x):return backend.softmax_old(x)
# def sign(x):return backend.sign(x)
# def abs(x):return backend.abs(x)
def bitwise_or(x,y):return backend.bitwise_or(x,y)
def argsort(x):return backend.argsort(x)

def ConvUp(images, filters, targets=None, moduleStride=1, paddingStart=0):     return conv_backend.ConvUp(images , filters, targets, moduleStride, paddingStart)
def DeConvDown(hidActs, filters, targets=None, moduleStride=1, paddingStart=0):return conv_backend.ConvUp(hidActs, filters, targets, moduleStride, paddingStart)

def ConvDown(hidActs, filters, moduleStride=1, paddingStart=0, imSizeX = "auto"): return conv_backend.ConvDown(hidActs, filters, moduleStride, paddingStart, imSizeX)
def DeConvUp(images , filters, moduleStride=1, paddingStart=0, imSizeX = "auto"): return conv_backend.ConvDown(images , filters, moduleStride, paddingStart, imSizeX)

def ConvOut(images, hidActs, targets = None, moduleStride = 1, paddingStart = 0 ,partial_sum = 0,filterSizeX = None): 
    return conv_backend.ConvOut(images, hidActs, targets, moduleStride, paddingStart, partial_sum,filterSizeX)
def DeConvOut(images, hidActs, targets = None, moduleStride = 1, paddingStart = 0 ,partial_sum = 0,filterSizeX = None): 
    return conv_backend.ConvOut(hidActs, images, targets, moduleStride, paddingStart, partial_sum,filterSizeX)

def MaxPool(images,subsX,startX,strideX,outputsX): return conv_backend.MaxPool(images,subsX,startX,strideX,outputsX)
def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX): return conv_backend.MaxPoolUndo(images,grad,maxes,subsX,startX,strideX)

def AvgPool(images,subsX,startX,strideX,outputsX): return conv_backend.AvgPool(images,subsX,startX,strideX,outputsX)
def AvgPoolUndo(images,grad,subsX,startX,strideX): return conv_backend.AvgPoolUndo(images,grad,subsX,startX,strideX)

def rnorm(images,N,addScale,powScale,blocked,minDiv): return conv_backend.rnorm(images,N,addScale,powScale,blocked,minDiv)


def max(A,axis=None,out=None):return __builtins__['max'](A) if isinstance(A,list) else backend.max(A,axis,out)
def min(A,axis=None,out=None):return __builtins__['min'](A) if isinstance(A,list) else backend.min(A,axis,out)
def sum(A,axis=None,out=None):return __builtins__['sum'](A) if isinstance(A,list) else backend.sum(A,axis,out)
def mean(A,axis=0,out=None):return backend.mean(A,axis,out)
def add(A,B,out=None):     return backend.add(A,B,out)       # A + B
def add_nt(A,B,out=None):return backend.add_nt(A,B,out)      # A + B.transpose()
def iadd(A,B):             return backend.iadd(A,B)          # A += B
def iaddmul(A,B,alpha):    return backend.iaddmul(A,B,alpha) # A += B*alpha (WARNING: value stored in B is undefined after this)
def iassign(A,B):    return backend.iassign(A,B)
def subtract(A,B,out=None):return backend.subtract(A,B,out)  # A - B
def subtract_nt(A,B,out=None):return backend.subtract_nt(A,B,out)  # A - B.transpose()
def isub(A,B):             return backend.isub(A,B)          # A -= B
def multiply(A,B,out=None):return backend.multiply(A,B,out)  # A * B
def imul(A,B):             return backend.imul(A,B)          # A *= B
def divide(A,B,out=None):  return backend.divide(A,B,out)    # A / B
def idiv(A,B):             return backend.idiv(A,B)          # A /= B
def reciprocal(A,out=None):return backend.reciprocal(A,out)  # 1. / A 
def transpose(A,out=None):return backend.transpose(A,out)
def maximum(A,B,out=None): return backend.maximum(A,B,out)
def dropout(A,B,rate,outA=None,outB=None): 
    assert seed_global    
    return backend.dropout(A,B,rate,outA,outB)
def k_sparsity_mask(x,k,axis=0): return backend.k_sparsity_mask(x,k,axis)
def fill_randn(A): 
    assert seed_global
    return backend.fill_randn(A)
def fill_rand(A): 
    assert seed_global    
    return backend.fill_rand(A)
# def garray(x): return backend.garray(x)
def hard_reshape(A,*shape): return backend.hard_reshape(A,*shape)
def arange(k): return backend.arange(k)
def sample(x): return backend.sample(x)
def spatial_dropout(x,size):
    crop_offset = (x.shape[2]-size) 
    mask = ones(x.shape)

    # for i in xrange(x.shape[0]):
    offset_x=np.random.randint(0,crop_offset+1) 
    offset_y=np.random.randint(0,crop_offset+1)         
    mask[:,:,offset_x:size+offset_x,offset_y:size+offset_y]=0
    return mask

def shift(x,shift=0):
    x = array(x)
    # print x

    for shift in range(1,shift+1):
        # print shift
        # print np.hstack((np.zeros((x.shape[0], shift)), x[:,:-shift])) 
        # print np.hstack((x[:,shift:],np.zeros((x.shape[0], shift))))
        x += np.hstack((np.zeros((x.shape[0], shift)), x[:,:-shift]))    
        x += np.hstack((x[:,shift:],np.zeros((x.shape[0], shift))))     
    return backend_array(x) 
def k_sparsity_group_mask(x,k,shift=0):
    x = array(x)
    for shift in range(1,shift+1):
        # print shift
        x += np.hstack((np.zeros((x.shape[0], shift)), x[:,:-shift]))    
        x += np.hstack((x[:,shift:],np.zeros((x.shape[0], shift))))    
    c=np.zeros(x.shape)
    b=np.argsort(x,kind='quicksort',axis=0)
    index = b[-k:,:]
    # print index
    for i in xrange(x.shape[1]):
        # print index[i]-k,index[i]+k+1
        for j in xrange(k):
            begin = i-shift if i-shift>=0 else 0
            end = i+shift+1 if i+shift+1<= x.shape[1] else x.shape[1]
            # print j,i,index[j,i],begin,end
            c[index[j,i],begin:end]=1
    # print c.sum(0)
    return backend_array(c)

def k_sparsity_group_mask_3d(x,k=1,shift=3,filter_type="box"):
    shift_in = shift
    shift_out = shift
    x = array(x)
    mini_batch = x.shape[0]
    width = int(np.sqrt(x.shape[1]))
    x = x.reshape(mini_batch,1,width,width)
    # print x
    # print "---"
    if filter_type=="box":
        filters = np.ones((1,1,shift_in,shift_in))
    elif filter_type=="gaussian":
        filters = array(gaussian2D(shift,k=.01).reshape(1,1,shift,shift))
    # print filters.dtype,x.dtype
    y = NumpyConvBackend.ConvUp(images=x, filters=filters, moduleStride=1 , paddingStart=(shift_in-1)/2)
    # print y
    # print "---"
    y = y.reshape(mini_batch,width**2)
    z = NumpyBackend.k_sparsity_mask(y,k=k,axis=0)
    z = z.reshape(mini_batch,1,width,width)

    filters = np.ones((1,1,shift_out,shift_out))
    # print z.shape
    t = NumpyConvBackend.DeConvUp(images=z, filters=filters, moduleStride=1 , paddingStart=(shift_out-1)/2, imSizeX=width)
    t[t>=1]=1

    t = t.reshape(mini_batch,width**2)
    return backend_array(t)


def create_S(n,width):
    def boundary(x,n):
        if x<0: return 0
        elif x>=n: return n
        else: return x

    def index_k(k,width,n):
        shift = (width-1)/2
        temp = np.zeros((n,n))
        i = k/n
        j = k%n
        # print boundary(i-shift,n),boundary(i+shift+1,n),boundary(j-shift,n),boundary(j+shift+1,n)
        temp[boundary(i-shift,n):boundary(i+shift+1,n),boundary(j-shift,n):boundary(j+shift+1,n)]=1
        return np.nonzero(temp.ravel())[0]
    
    S = np.zeros((n,n))
    for k in xrange(n):
        S[k,index_k(k=k,width=width,n = int(np.sqrt(n)))]=1
    # return S
    return backend_array(S)





#def l2_normalize(x):return backend.l2_normalize(x)

def garray(x):
    if type(x)==gp.garray: 
        # print "WARNING!!!!!"
        return x
    else: 
        # if x.dtype != "float64" and x.dtype != "float32":
        #     # print "WARNING!!!!!", x.dtype           
        #     return gp.garray(x)
        # else:
        return gp.garray(x.astype("float32"))

def array(x):
    if type(x)==gp.garray: 
        # print "WARNING!!!!!"
        return x.as_numpy_array().astype('float64')
    else: 
        return x.astype('float64')

def backend_array(x):
    if backend==NumpyBackend: return array(x)
    elif backend==GnumpyBackend: return garray(x)
    else: raise Exception("No Valid Backend")     



def err_plot(err_list,a,b):
    if type(err_list)==gp.garray: err=err.as_numpy_array()
    plt.grid(True)
    plt.plot(np.arange(a,b),err_list[a:b])
#   plt.show()

def plot_filters(x,img_shape,tile_shape):
    if type(x)==gp.garray: print type(x);show_filters(x.as_numpy_array(),img_shape,tile_shape)
    elif type(x)==np.ndarray: print type(x);show_filters(x,img_shape,tile_shape)
    #plt.show()

def imshow(x):
    if type(x)==gp.garray: print type(x);plt.imshow(x.as_numpy_array(), cmap=plt.cm.gray, interpolation='nearest')
    elif type(x)==np.ndarray: print type(x);plt.imshow(x, cmap=plt.cm.gray, interpolation='nearest')
    #plt.show()

#from __main__ import feedforward,batch_size,want_KL,KL_flat,rho_target,want_exp,sigma,H

# def show_images(imgs,tile_shape=(1,1),scale=None,bar=False,unit=True,bg="black",index="cudnn"):

#     if imgs.ndim==4:
#         if index=="cudnn": 
#             imgs = imgs.reshape(imgs.shape[0],-1).T.reshape(imgs.shape[1],imgs.shape[2],imgs.shape[3],imgs.shape[0])
#         else: assert index=="cudaconvnet"
#     elif imgs.ndim == 2:
#         imgs=imgs.T.reshape(1,int(imgs.shape[1]**.5),int(imgs.shape[1]**.5),imgs.shape[0]) 
#     else: raise Exception("wrong ndim!")


#     if type(imgs) == gp.garray: 
#         # print imgs.shape
#         imgs=imgs.as_numpy_array()
#     else: imgs = imgs.copy()
    
       
#     # print "ali",imgs.shape
#     assert imgs.shape[3] == tile_shape[0]*tile_shape[1]
#     img_shape = imgs.shape

#     # if unit:
#     #     imgs = imgs - imgs.min()  
#     #     imgs = imgs / imgs.max()  

#     if bg=="white" :out=np.ones(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))
#     if bg=="black" :out=np.zeros(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))

#     for i in range(tile_shape[0]):
#         for j in range(tile_shape[1]):
#             k = tile_shape[1]*i+j
#             if unit: x = scale_to_unit_interval(imgs[:,:,:,k])
#             else: x = imgs[:,:,:,k]
#             x = imgs[:,:,:,k]            
#             out[(img_shape[1]+1)*i+1:(img_shape[1]+1)*i+1+img_shape[1],(img_shape[2]+1)*j+1:(img_shape[2]+1)*j+1+img_shape[2],:] = np.rollaxis(x,0,3)

#     if scale!=None: fig=plt.figure(num=None, figsize=(tile_shape[1]*scale, tile_shape[0]*scale), dpi=80, facecolor='w', edgecolor='k')
#     if out.shape[2] == 1: 
#         plt.imshow(out.squeeze(),cmap=plt.cm.gray, interpolation='nearest')
#         return None

#     # print out.max(),out.dtype

#     plt.imshow(out,interpolation='nearest')
#     if bar: plt.colorbar()

def scale_to_225(x):
    x = x.copy()
    # x = x - x.min()
    # x = x / x.std()
    x -= x.min()
    x *= 255.0 / (x.max() + 1e-8)
    return x

def scale_to_unit_interval(x):
    x = x.copy()
    # x = x - x.min()
    # x = x / x.std()
    x -= x.min()
    x *= 1.0 / (x.max() + 1e-8)
    return x

def yuv2rgb(temp):
    Y = temp[:,:,0].copy()
    U = temp[:,:,1].copy()
    V = temp[:,:,2].copy()

    temp[:,:,0] = Y+ 0 * U + 1.13983 * V
    temp[:,:,1] = Y+ -0.39465 * U + -0.58060 * V
    temp[:,:,2] = Y+ 2.03211 * U + 0 * V    
    return temp


def show_images(imgs,tile_shape=(1,1),scale=None,bar=False,unit=1,bg="black",index="cudnn",yuv=False):
    if imgs.ndim==4:
        if index=="cudnn": 
            imgs = imgs.reshape(imgs.shape[0],-1).T.reshape(imgs.shape[1],imgs.shape[2],imgs.shape[3],imgs.shape[0])
        else: assert index=="cudaconvnet"
    elif imgs.ndim == 2:
        # try:
        # print imgs.shape[0],1,int(imgs.shape[1]**.5),int(imgs.shape[1]**.5)
        imgs=imgs.T.reshape(1,int(imgs.shape[1]**.5),int(imgs.shape[1]**.5),imgs.shape[0]) 
        # except:
            # print int(imgs.shape[1]**.5),int(imgs.shape[1]**.5),imgs.shape[0]
    else: raise Exception("wrong ndim!")    
    if type(imgs) == gp.garray: imgs=imgs.as_numpy_array()
    # if imgs.ndim == 2:
        # imgs=imgs.T.reshape(1,imgs.shape[1]**.5,imgs.shape[1]**.5,imgs.shape[0])        

    if unit==2:
        imgs = imgs - imgs.min()  
        imgs = imgs / imgs.max()  

    try:
        assert imgs.shape[3] == tile_shape[0]*tile_shape[1]
    except: 
        print "Image size doesn't match the tile shape.", imgs.shape[3]


    img_shape = imgs.shape


    if imgs.dtype!=np.uint8:
        if bg=="white" :out=np.ones(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))
        if bg=="black" :out=np.zeros(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))
    else:
        assert unit == 0
        if bg=="white" :out=(255*np.ones(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))).astype(np.uint8)
        if bg=="black" :out=(np.zeros(((img_shape[1]+1)*tile_shape[0]+1,(img_shape[2]+1)*tile_shape[1]+1,img_shape[0]))).astype(np.uint8)

    for i in range(tile_shape[0]):
        for j in range(tile_shape[1]):
            k = tile_shape[1]*i+j

            temp = imgs[:,:,:,k]
            temp = np.rollaxis(temp,0,3)
            if yuv: temp = yuv2rgb(temp)           
            if unit==1: temp = scale_to_unit_interval(temp)

            # print temp.shape
            # assert temp.dtype != np.uint8                
            # temp = scale_to_225(temp)
            # temp = temp.astype(np.uint8)   
            # temp = Image.fromarray(temp,mode='YCbCr')
            # temp = temp.convert('RGB')
            # temp = np.array(temp)
            # temp = temp /255.0
            # plt.imshow(temp); plt.show()
            # print 'ali',out.dtype


                # A = np.array([[1.,                 0.,  0.701            ],
                             # [1., -0.886*0.114/0.587, -0.701*0.299/0.587],
                             # [1.,  0.886,                             0.]])

                # shape_ = temp.shape
                # temp = temp.reshape(-1,3)
                # print temp.shape,A.shape
                # temp = np.dot(temp,A)
                # temp = temp.reshape(shape_)
                # assert temp.max()<=1
                # temp *= 255.0
                # temp = temp.astype('uint8')
                # print temp
                # temp = Image.fromarray(temp,mode='YCbCr')
                # temp = temp.convert('RGB')
                # temp = np.array(temp)[:,:,0:3]
                # temp /=255.0
            out[(img_shape[1]+1)*i+1:(img_shape[1]+1)*i+1+img_shape[1],(img_shape[2]+1)*j+1:(img_shape[2]+1)*j+1+img_shape[2],:] = temp
            # print out.dtype
    if scale!=None: fig=plt.figure(num=None, figsize=(tile_shape[1]*scale, tile_shape[0]*scale), dpi=80, facecolor='w', edgecolor='k')
    if out.shape[2] == 1: 
        plt.imshow(out.squeeze(),cmap=plt.cm.gray, interpolation='nearest')
        if bar: plt.colorbar()        
        return None
    plt.imshow(out,interpolation='nearest')
    if bar: plt.colorbar()



def make_activation(typename):
    if   typename == linear:   return linear_prime
    elif typename == sigmoid:  return sigmoid_prime
    elif typename == relu:     return relu_prime
    elif typename == softmax:  return softmax_prime
    elif typename == None:     return None

def find_batch_size(x):
    if x.ndim==4: return x.shape[3]
    elif x.ndim==2: return x.shape[0]

def mask_3d(x,k):
    x_ = np.swapaxes(x.as_numpy_array(),3,1).reshape(x.shape[0]*x.shape[3],-1)
    mask_ = NumpyBackend.k_sparsity_mask(x_,k,axis = 1)
    mask = np.swapaxes(mask_.reshape(x.shape[0],-1,x.shape[2],x.shape[1]),1,3)
    return gp.garray(mask)    

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def set_backend(name,board=None,conv_backend="cudaconvnet",seed=None,silent=False):
    global backend
    if name=="numpy": 
        backend=NumpyBackend
        set_seed(seed)        
        if not silent: print "numpy is running, seed:",seed_global        
    elif name=="gnumpy": 
        assert board!=None
        backend=GnumpyBackend
        gp.set_board(int(board))
        set_seed(seed)
        set_conv_backend(conv_backend)         
        if not silent: print "gnumpy is running on GPU board:",board,"  convolution backend:",conv_backend, "  seed:",seed_global        
    else: 
        raise Exception("No Valid Backend")

def set_seed(seed=None):
    global seed_global
    if seed==None:
        dt = datetime.now()
        seed_global = dt.microsecond
    else: 
        seed_global = seed
    
    if backend==NumpyBackend:
        np.random.seed(seed_global)
    elif backend==GnumpyBackend:
        gp.seed_rand(seed_global)

def get_seed():
    return seed_global

def set_conv_backend(name):
    global conv_backend
    # global backend
    if name=="cudnn": 
        assert backend==GnumpyBackend; 
        CudnnBackend.init()
        conv_backend=CudnnBackend
    
    elif name=="cudaconvnet": 
        assert backend==GnumpyBackend
        conv_backend=CudaconvnetBackend
    
    elif name=="cudaconvnet_reversed": 
        assert backend==GnumpyBackend
        conv_backend=CudaconvnetBackendReversed

    elif name=="numpy_reversed": 
        assert backend==NumpyBackend
        conv_backend=NumpyConvBackendReversed        

    elif name=="numpy": 
        assert backend==NumpyBackend
        conv_backend=NumpyConvBackend
    else: raise Exception("No Valid Conv Backend")    
    








def work_address():
    return os.environ["WORK"]



def nas_address():
    return os.environ["NAS"]

def T_max(X,Y):
    x2 = sum(X**2,1)
    y2 = sum(Y**2,1).T
    xy = dot(X,Y.T)
    dist = x2+y2-2*xy
    index =  dist.argmax(1)
    out = zeros(Y.shape)
    out[:] = Y[index]
    # print dist
    # print index
    # print out
    return Y,index

def T_sort(H):
    # print type(H)
    n = H.shape[0]
    prob = (np.arange(n)+1)/(n+1.0)
    # T = prob
    T = (norm.ppf(prob))

    # print 't',T    
    H_cpu = array(H)
    b=np.argsort(H_cpu,kind='quicksort',axis=0)
    # print b
    c=np.zeros(H_cpu.shape)
    for i in xrange(H.shape[1]):
        c[b[:,i],i] = T
    return backend_array(c)


def gaussian1D(n,k=.1):
    assert n%2==1
    sigma = (n-1)**2/4/(-np.log(k))
    out = np.zeros(n)
    array = range(-(n-1)/2,(n+1)/2)
    print array,sigma
    for i in array:
        out[i+(n-1)/2] = np.exp((-i**2.0)/sigma)
    return backend_array(out)

def gaussian2D(n,k=.1):
    assert n%2==1    
    sigma = (n-1)**2/4/(-np.log(k))
    out = np.zeros((n,n))
    array = range(-(n-1)/2,(n+1)/2)
    for i in array:
        for j in array:
            dist = np.sqrt(i**2+j**2)
            out[i+(n-1)/2,j+(n-1)/2] = np.exp((-dist**2.0)/sigma)
    return backend_array(out) 

def show():
    plt.show()


def SpatialContrastMean(images, filters, minDiv = 1.0):
    assert backend==GnumpyBackend; 
    images = garray(images)

    filter_size = filters.shape[0]
    filters = (filters/filters.sum()).copy()

    if images.shape[1]==3:
        filters_3D = zeros((3,3,filter_size,filter_size))
        for i in xrange(3):
            filters_3D[i,i,:,:] = filters
    elif images.shape[1]==1:
        filters_3D = zeros((1,1,filter_size,filter_size))
        filters_3D[0,0,:,:] = filters        

    padding = (filter_size-1)/2

    mean = ConvUp(images=images,filters=filters_3D,moduleStride=1,paddingStart=padding)
    out = images - mean
    return array(out)

def SpatialContrast(images, filters, minDiv = 1.0):
    assert backend==GnumpyBackend; 
    images = garray(images)

    filter_size = filters.shape[0]
    filters = (filters/filters.sum()).copy()

    d = images.shape[1]    
    filters_3D = zeros((d,d,filter_size,filter_size))
    for i in xrange(d):
        filters_3D[i,i,:,:] = filters

    padding = (filter_size-1)/2

    mean = ConvUp(images=images,filters=filters_3D,moduleStride=1,paddingStart=padding)
    images2 = images**2

    var = ConvUp(images=images2,filters=filters_3D,moduleStride=1,paddingStart=padding)
    std = (minDiv+var)**.5
    # std = 1
    # mean = 0

    out = (images - mean)/std
    return array(out)




class dp_ram: #this returns a "data_batch" of the whole data that already exists in the RAM.
    def __init__(self,X,T=None,X_test=None,T_test=None,T_train_labels=None,T_labels=None,train_range=None,test_range=None,data_batch=10000):
        assert type(X) in (np.ndarray,gp.garray,h5py._hl.dataset.Dataset)  
        # assert type(X) == np.ndarray
        self.data_batch = data_batch

        self.X = X
        self.T = T
        self.X_test = X_test
        self.T_test = T_test
        self.T_train_labels = T_train_labels
        self.T_labels = T_labels
        

        if train_range == None:
            self.train_range = [0,self.X.shape[0]/data_batch]
        else:
            self.train_range = train_range      
        self.train_id = self.train_range[0]    
        self.N_train = self.data_batch*(self.train_range[1]-self.train_range[0])
        assert self.N_train <= self.X.shape[0]
        if self.N_train < self.X.shape[0]: 
            print "*********************************Train is not complete*********************************",self.N_train,self.X.shape[0]

        if type(X_test) in (np.ndarray,gp.garray,h5py._hl.dataset.Dataset):
            if test_range == None:
                self.test_range = [0,self.X_test.shape[0]/data_batch]
            else:
                self.test_range = test_range      
            self.N_test = self.data_batch*(self.test_range[1]-self.test_range[0])
            self.test_id = self.test_range[0]    
            assert self.N_test <= self.X_test.shape[0]
            if self.N_test<self.X_test.shape[0]:
                print "*********************************Test is not complete*********************************",self.N_test,self.X_test.shape[0]

        # if type(X_test) in (np.ndarray,gp.garray):
        #     self.N_test = self.X_test.shape[0]        
        #     self.test_id = test_range[0]
        # else: assert X_test==None


    # def X_id(self,id):
    #     return data_convertor(self.X,id,id+1)
    def test(self):
        test_id_temp = self.test_id
        self.test_id = self.test_id+1 if self.test_id != self.test_range[1]-1 else self.test_range[0]
        return data_convertor(self.X_test,self.data_batch*test_id_temp,self.data_batch*(test_id_temp+1)),\
               data_convertor(self.T_test,self.data_batch*test_id_temp,self.data_batch*(test_id_temp+1)),\
               test_id_temp       
        # test_id = 0 
        # return data_convertor(self.X_test,self.data_batch*test_id,self.data_batch*(test_id+1)),\
        #        data_convertor(self.T_test,self.data_batch*test_id,self.data_batch*(test_id+1)),\
        #        data_convertor(self.T_labels,self.data_batch*test_id,self.data_batch*(test_id+1)),\
        #        test_id

    def train(self):
        train_id_temp = self.train_id
        self.train_id = self.train_id+1 if self.train_id != self.train_range[1]-1 else self.train_range[0]
        return data_convertor(self.X,self.data_batch*train_id_temp,self.data_batch*(train_id_temp+1)),\
               data_convertor(self.T,self.data_batch*train_id_temp,self.data_batch*(train_id_temp+1)),\
               train_id_temp

    # def X_itr(self):
    #     train_id_temp = self.train_id
    #     self.train_id = self.train_id+1 if self.train_id != self.train_range[1]-1 else self.train_range[0]        
    #     return data_convertor(self.X,self.data_batch*train_id_temp,self.data_batch*(train_id_temp+1))


def data_convertor(X,a,b):
    # assert type(X) == np.ndarray
    if type(X) == gp.garray: 
        assert backend == GnumpyBackend
        if len(X.shape) == 4: return X[a:b,:,:,:]
        else : return X[a:b]
    elif type(X) in (np.ndarray,h5py._hl.dataset.Dataset): 
        if backend == GnumpyBackend: 
            if len(X.shape) == 4: return garray(X[a:b,:,:,:])
            else : return garray(X[a:b])
        elif backend == NumpyBackend:
            if len(X.shape) == 4: return X[a:b,:,:,:]
            else : return X[a:b]  
    else: 
        print type(X),backend
        raise Exception("No Valid X for data_convertor") 


def shutdown():
    gp.shutdown()

















# def T_max(X,T):
#     x2 = sum(X**2,1)
#     t2 = sum(T**2,1).T
#     xt = dot(X,T.T)
#     dist = x2+t2-2*xt
#     index =  dist.argmax(0)
#     out = X.copy()
#     out[index] = T[index]
#     # out = Y[index]
#     # print X[0][:100]
#     # print X[1][:100]
#     # print dist
#     # print index
#     # print out
#     return out,index





    #     self.threads_list = [data_provider_imagenet.ThreaCifar10Open(self,X,T,train_id,i,T_labels,offset_x,offset_y) for i in range(self.num_threads)]
    #     for th in self.threads_list:
    #         th.start()
    #     for th in self.threads_list:
    #         th.join() 

    # @staticmethod
    # class ThreaCifar10Open(threading.Thread):
    #     def __init__(self, class_name, X, T, train_id, threadID, T_labels, offset_x, offset_y):
    #         threading.Thread.__init__(self)
    #         self.class_name = class_name
    #         self.offset_x = offset_x
    #         self.offset_y = offset_y
    #         self.X = X
    #         self.T = T
    #         self.T_labels = T_labels
    #         self.train_id = train_id
    #         self.threadID = threadID        
    #     def run(self):
    #         self.class_name.cifar10_open(self.X, self.T, self.train_id, self.threadID, self.T_labels, self.offset_x, self.offset_y)

    # def cifar10_open(self,X,T,train_id,threadID,T_labels,offset_x,offset_y):
    #     fo = open(nas_address()+'/PSI-Share-no-backup/Ali/Dataset/CIFAR10/batches/data_batch_'+str(train_id)+'/data_batch_'+str(train_id)+'.'+str(threadID), 'rb')
    #     dict = cPickle.load(fo)   
    #     for i in xrange(3072/self.num_threads):
    #         img_code = dict['data'][i]
    #         img_label = self.map(dict['labels'][i][0])
    #         # print dict['labels'][i][0],img_label,self.map(0),self.map(1),self.map

    #         nparr = np.fromstring(img_code, np.uint8)
    #         temp = (cv2.imdecode(nparr,1)[:,:,::-1].reshape(256**2,3).T.reshape(3,256,256)-self.data_mean)

    #         if self.mode=="vgg": temp = temp[::-1,:,:]  
    #         X[i+threadID*3072/self.num_threads,:,:,:] = temp[:,offset_x:self.crop_size+offset_x,offset_y:self.crop_size+offset_y]
    #         T[threadID*3072/self.num_threads+i,img_label] = 1.0
    #         if T_labels!=None: 
    #             T_labels[threadID*3072/self.num_threads+i]=img_label


















