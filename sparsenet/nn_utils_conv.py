import numpy as np
import gnumpy as gp
from datetime import datetime
import pylab as plt
import gnumpy.cudamat_j as ConvNet
import time
import gnumpy.libcudnn as libcudnn
import ctypes
cudamat = gp.cmat


class GnumpyBackend(object):
    @staticmethod
    def empty(shape):
        if type(shape)!=tuple: return gp.empty(shape)
        return gp.empty(shape)    



class CudnnBackend(GnumpyBackend):

    @staticmethod
    def init():
        global cudnn_context, tensor_format, data_type, convolution_mode, pool_mode, conv_desc, convolution_fwd_pref
        tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
        data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
        convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
        pool_mode = libcudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']
        # convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']

        conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
        
        cudnn_context = libcudnn.cudnnCreate()

    @staticmethod
    def create_desc(x):
        x_desc = libcudnn.cudnnCreateTensorDescriptor()
        libcudnn.cudnnSetTensor4dDescriptor(x_desc, tensor_format, data_type, x.shape[0], x.shape[1], x.shape[2], x.shape[3])  
        return x_desc      

    @staticmethod
    def create_filter_desc(x):
        x_desc = libcudnn.cudnnCreateFilterDescriptor()
        libcudnn.cudnnSetFilter4dDescriptor(x_desc, data_type, x.shape[0], x.shape[1], x.shape[2], x.shape[3])  
        return x_desc      

    @staticmethod
    def ConvUp(images, filters, targets, moduleStride, paddingStart):
        # t = time.time()

        assert paddingStart>=0
        # Descriptor for input
        images_desc = CudnnBackend.create_desc(images)
        filters_desc = CudnnBackend.create_filter_desc(filters)

        libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h=paddingStart, pad_w=paddingStart, u=moduleStride, v=moduleStride, upscalex=1, upscaley=1, mode=convolution_mode)


        if targets == None:
            _, _, height_output, width_output = libcudnn.cudnnGetConvolution2dForwardOutputDim(conv_desc, images_desc, filters_desc)
            targets = GnumpyBackend.empty((images.shape[0], filters.shape[0], height_output, width_output))

        # print (images.shape[0], filters.shape[0], height_output, width_output)
        targets_desc = CudnnBackend.create_desc(targets)

        alpha = 1.0
        beta = 0
        algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(cudnn_context, images_desc, filters_desc, conv_desc, targets_desc, convolution_fwd_pref, 0)
        libcudnn.cudnnConvolutionForward(cudnn_context, alpha, images_desc, images.data(), filters_desc, filters.data(), conv_desc, algo, None, 0, beta, targets_desc, targets.data())

        # libcudnn.cudnnDestroyTensorDescriptor(images_desc)
        # libcudnn.cudnnDestroyTensorDescriptor(targets_desc)
        # libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
        # libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
        # libcudnn.cudnnDestroy(cudnn_context)
        # gp.sync()
        # print "conv-up:",images.shape,filters.shape,(time.time()-t)*24        
        return targets

    @staticmethod
    def DeConvDown(hidActs, filters, targets, moduleStride, paddingStart):
        return CudnnBackend.ConvUp(hidActs, filters, targets, moduleStride, paddingStart)

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride, paddingStart,imSizeX):
        numImages, numFilters, numModulesX, numModulesX,  = hidActs.shape
        numFilters, numFilterChannels, filterSizeX, filterSizeX = filters.shape
        if imSizeX=="auto":
            imSizeX = (numModulesX - 1) * moduleStride - 2*abs(paddingStart) + filterSizeX
        # print "hi"
        # print numModulesX,imSizeX
        targets = GnumpyBackend.empty(( numImages, numFilterChannels, imSizeX, imSizeX))

        # print targets.shape

        hidActs_desc = CudnnBackend.create_desc(hidActs)
        filters_desc = CudnnBackend.create_filter_desc(filters)
        targets_desc = CudnnBackend.create_desc(targets)
        libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h=paddingStart, pad_w=paddingStart, u=moduleStride, v=moduleStride, upscalex=1, upscaley=1, mode=convolution_mode)        

        alpha = 1.0
        beta = 0
        libcudnn.cudnnConvolutionBackwardData(handle=cudnn_context, alpha=alpha, filterDesc=filters_desc, filterData=filters.data(),
                                            diffDesc=hidActs_desc, diffData=hidActs.data(), convDesc=conv_desc, beta=beta, gradDesc=targets_desc, gradData=targets.data())
           
        # libcudnn.cudnnDestroyTensorDescriptor(hidActs_desc)
        # libcudnn.cudnnDestroyTensorDescriptor(targets_desc)
        # libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
        # libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
        # libcudnn.cudnnDestroy(cudnn_context)
        return targets


    @staticmethod
    def ConvOut(images, hidActs, targets, moduleStride, paddingStart, partial_sum,filterSizeX=None):
        assert filterSizeX != None
        assert paddingStart >= 0
        numImages, numFilters, numModulesX, numModulesX = hidActs.shape
        numImages, numChannels, imSizeX, imSizeX = images.shape    
        numFilterChannels = numChannels

        # filterSizeX = imSizeX - (numModulesX - 1) * moduleStride + 2*abs(paddingStart)


        if targets == None:
            targets = GnumpyBackend.empty((numFilters, numFilterChannels, filterSizeX, filterSizeX))


        hidActs_desc = CudnnBackend.create_desc(hidActs)
        images_desc = CudnnBackend.create_desc(images)
        targets_desc = CudnnBackend.create_filter_desc(targets)
        libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h=paddingStart, pad_w=paddingStart, u=moduleStride, v=moduleStride, upscalex=1, upscaley=1, mode=convolution_mode)        


        alpha = 1.0
        beta = 0

        libcudnn.cudnnConvolutionBackwardFilter(handle=cudnn_context, alpha=alpha, srcDesc=images_desc, srcData=images.data(), diffDesc=hidActs_desc, diffData=hidActs.data(),
                                           convDesc=conv_desc, beta=beta, gradDesc=targets_desc, gradData=targets.data())

        # libcudnn.cudnnDestroyTensorDescriptor(hidActs_desc)
        # libcudnn.cudnnDestroyTensorDescriptor(targets_desc)
        # libcudnn.cudnnDestroyFilterDescriptor(images_desc)
        # libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
        # libcudnn.cudnnDestroy(cudnn_context)
        return targets

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):

        targets = GnumpyBackend.empty(images.shape)
        alpha = 1.0
        beta = 0
            
        images_desc = CudnnBackend.create_desc(images)
        grad_desc = CudnnBackend.create_desc(grad)
        maxes_desc = CudnnBackend.create_desc(maxes)
        targets_desc = CudnnBackend.create_desc(targets)

        pool_desc = libcudnn.cudnnCreatePoolingDescriptor()        
        libcudnn.cudnnSetPooling2dDescriptor(poolingDesc=pool_desc, mode=pool_mode, windowHeight = subsX, windowWidth = subsX,
                                        verticalPadding = -startX, horizontalPadding = -startX, verticalStride = strideX, horizontalStride = strideX)

        libcudnn.cudnnPoolingBackward(handle=cudnn_context, poolingDesc=pool_desc, alpha=alpha, srcDesc=maxes_desc, srcData=maxes.data(), srcDiffDesc=grad_desc,
                 srcDiffData=grad.data(), destDesc=images_desc, destData=images.data(), beta=beta, destDiffDesc=targets_desc, destDiffData=targets.data())
                
        return targets


    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):

        numImages, numChannels, imSizeX, imSizeX = images.shape    
        numImgColors = numChannels
        targets = GnumpyBackend.empty((numImages, numChannels, outputsX, outputsX)) - 1e100

        images_desc = CudnnBackend.create_desc(images)
        targets_desc = CudnnBackend.create_desc(targets)
        
        pool_desc = libcudnn.cudnnCreatePoolingDescriptor()
        #####verticalPadding should be negative!!
        libcudnn.cudnnSetPooling2dDescriptor(poolingDesc=pool_desc, mode=pool_mode, windowHeight = subsX, windowWidth = subsX,
                                            verticalPadding = -startX, horizontalPadding = -startX, verticalStride = strideX, horizontalStride = strideX)

        libcudnn.cudnnPoolingForward(handle=cudnn_context, poolingDesc=pool_desc, alpha = 1.0, srcDesc=images_desc, srcData=images.data(), beta = 0.0, destDesc=targets_desc, destData=targets.data())
        return targets

    @staticmethod
    def rnorm(images,N,addScale,powScale,blocked,minDiv):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])          
        # print "cudnn",images.shape
        out = ConvNet.ResponseNorm(images,N,addScale,powScale,blocked,minDiv)  
        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])


    @staticmethod
    def AvgPoolUndo(images,grad,subsX,startX,strideX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])   

        return ConvNet.AvgPoolUndo(images,grad,subsX,startX,strideX)

    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])   

        targets = ConvNet.AvgPool(images,subsX,startX,strideX,outputsX)        
        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])  

    @staticmethod
    def DeConvUp(images, filters, moduleStride , paddingStart):
        return CudnnBackend.ConvDown(images, filters, moduleStride , paddingStart)            
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class CudaconvnetBackend(GnumpyBackend):
    @staticmethod
    def ConvUp(images, filters, targets, moduleStride, paddingStart):
        assert paddingStart>=0
        paddingStart = -paddingStart
        assert targets==None

        images_shape = images.shape
        filters_shape = filters.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        filters = filters.reshape(filters_shape[0],-1).T.reshape(filters_shape[1],filters_shape[2],filters_shape[3],filters_shape[0])

        if filters.shape[3]==1: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:1]=filters
                out = ConvNet.convUp(images, filters_16, targets, moduleStride = moduleStride, paddingStart = paddingStart)
                out = out[:1,:,:,:]
        elif filters.shape[3]==3: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:3]=filters
                out = ConvNet.convUp(images, filters_16, targets, moduleStride = moduleStride, paddingStart = paddingStart)
                out = out[:3,:,:,:]                
        elif filters.shape[3]%16==0: 
                out = ConvNet.convUp(images, filters, targets, moduleStride, paddingStart)
        else: raise Exception("Filters Mode 16")   

        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])

    @staticmethod
    def DeConvDown(hidActs, filters, targets, moduleStride, paddingStart):
        return CudaconvnetBackend.ConvUp(hidActs, filters, targets, moduleStride, paddingStart)

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride, paddingStart, imSizeX):
        assert paddingStart>=0
        paddingStart = -paddingStart

        hidActs_shape = hidActs.shape
        filters_shape = filters.shape
        hidActs = hidActs.reshape(hidActs_shape[0],-1).T.reshape(hidActs_shape[1],hidActs_shape[2],hidActs_shape[3],hidActs_shape[0])
        filters = filters.reshape(filters_shape[0],-1).T.reshape(filters_shape[1],filters_shape[2],filters_shape[3],filters_shape[0])

        if filters.shape[3]==1 and hidActs.shape[0]==1: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:1] = filters
            out = ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart, imSizeX = imSizeX)
        elif filters.shape[3]==3 and hidActs.shape[0]==3: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:3] = filters
            out = ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart, imSizeX = imSizeX)            
        elif filters.shape[3]%16==0 and hidActs.shape[0]%16==0:
            out = ConvNet.convDown(hidActs, filters, moduleStride, paddingStart, imSizeX)
        else: raise Exception("Hidden or Filters Mode 16")

        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])        

    @staticmethod
    def ConvOut(images, hidActs, targets, moduleStride, paddingStart, partial_sum,filterSizeX):
        assert filterSizeX != None
        assert paddingStart>=0
        paddingStart = -paddingStart

        images_shape = images.shape
        hidActs_shape = hidActs.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        hidActs = hidActs.reshape(hidActs_shape[0],-1).T.reshape(hidActs_shape[1],hidActs_shape[2],hidActs_shape[3],hidActs_shape[0])

        if hidActs.shape[0]==1:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX=filterSizeX)
            out = out[:,:,:,:1]
        elif hidActs.shape[0]==3:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX=filterSizeX)
            out = out[:,:,:,:3]            
        elif hidActs.shape[0]%16 == 0:
            # t = time.time()
            temp = ConvNet.convOutp(images = images, hidActs=hidActs, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX=filterSizeX)
            # print "nn_utils",1000*(time.time()-t)
            out = temp
        else: raise Exception("Hidden Mode 16")  

        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])        
            
    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        # print "ccc",images.dtype        
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        out = ConvNet.MaxPool(images,subsX,startX,strideX,outputsX)                      
        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])         

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        grad_shape = grad.shape
        grad = grad.reshape(grad_shape[0],-1).T.reshape(grad_shape[1],grad_shape[2],grad_shape[3],grad_shape[0])
        maxes_shape = maxes.shape
        maxes = maxes.reshape(maxes_shape[0],-1).T.reshape(maxes_shape[1],maxes_shape[2],maxes_shape[3],maxes_shape[0])                

        out = ConvNet.MaxPoolUndo(images,grad,maxes,subsX,startX,strideX)  
        out_shape = out.shape        
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])         

    @staticmethod
    def rnorm(images,N,addScale,powScale,blocked):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])                
        out = ConvNet.ResponseNorm(images,N,addScale,powScale,blocked,minDiv)  
        out_shape = out.shape
        return out.reshape(-1,out_shape[3]).T.reshape(out_shape[3],out_shape[0],out_shape[1],out_shape[2])        


    @staticmethod
    def AvgPoolUndo(images,grad,subsX,startX,strideX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])   

        return ConvNet.AvgPoolUndo(images,grad,subsX,startX,strideX)

    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])   

        targets = ConvNet.AvgPool(images,subsX,startX,strideX,outputsX)        
        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])         

    @staticmethod
    def DeConvUp(images, filters, moduleStride , paddingStart):
        return CudaconvnetBackend.ConvDown(images, filters, moduleStride , paddingStart)
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class NumpyConvBackend(object):
    @staticmethod
    def ConvUp(images, filters, moduleStride, paddingStart):
        assert paddingStart>=0
        paddingStart = -paddingStart

        images_shape = images.shape
        filters_shape = filters.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        filters = filters.reshape(filters_shape[0],-1).T.reshape(filters_shape[1],filters_shape[2],filters_shape[3],filters_shape[0])

        global images2
        assert paddingStart <= 0
        numChannels, imSizeX, imSizeX, numImages = images.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        # assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        # assert        (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride+1
        targets = np.zeros((numFilters, numModulesX, numModulesX, numImages))
        images2 = np.zeros((numChannels, 
                            imSizeX+2*abs(paddingStart), 
                            imSizeX+2*abs(paddingStart), 
                            numImages))  
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1).T
                targets_patch = np.dot(images_patch,filters_2d)
                targets[:,i/moduleStride,j/moduleStride,:]=targets_patch.T
                
        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])                
        
    @staticmethod
    def DeConvDown(hidActs, filters, targets, moduleStride, paddingStart):
        return NumpyConvBackend.ConvUp(hidActs, filters, targets, moduleStride, paddingStart)

    # @staticmethod
    # def AvgPoolUndo(images,grad,subsX,startX,strideX): 
    #     numChannels, imSizeX_, imSizeX, numImages = images.shape    
    #     assert imSizeX_ == imSizeX
    #     numImgColors = numChannels
    #     numChannels, outputsX_, outputsX, numImages = grad.shape
    #     assert outputsX_ == outputsX
    #     targets = np.zeros(images.shape)
    #     for i in range(numImages):
    #         for c in range(numChannels):
    #             o1 = 0
    #             for s1 in range(startX, imSizeX, strideX):
    #                 if s1<0:
    #                     continue
    #                 o2 = 0
    #                 for s2 in range(startX, imSizeX, strideX):
    #                     if s2<0:
    #                         continue
    #                     for u1 in range(subsX):
    #                         for u2 in range(subsX):
    #                             try:
    #                                 #if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
    #                                 targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]

    #                             except IndexError:
    #                                 pass #??  I don't fucking get it.
    #                     o2 += 1
    #                 o1 += 1
    #     return targets/subsX**2


    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):       
        images_shape = images.shape

        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])

        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages))
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] += images[c,s1+u1,s2+u2,i]                           
                                except IndexError:
                                    pass #?
                        o2 += 1
                    o1 += 1
        targets = 1.0*targets/subsX**2
        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])          

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        grad_shape = grad.shape
        grad = grad.reshape(grad_shape[0],-1).T.reshape(grad_shape[1],grad_shape[2],grad_shape[3],grad_shape[0])
        maxes_shape = maxes.shape
        maxes = maxes.reshape(maxes_shape[0],-1).T.reshape(maxes_shape[1],maxes_shape[2],maxes_shape[3],maxes_shape[0])  

        numChannels, imSizeX_, imSizeX, numImages = images.shape    
        assert imSizeX_ == imSizeX
        numImgColors = numChannels
        numChannels, outputsX_, outputsX, numImages = maxes.shape
        assert outputsX_ == outputsX    
        # print maxes.shape, grad.shape   
        targets = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX): ######why
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                        targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]
                                        break
                                except IndexError:
                                    pass #??  I don't fucking get it.                           
                            else: continue
                            break
                        o2 += 1
                    o1 += 1
        
        targets_shape = targets.shape        
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])          

    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        images_shape = images.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])

        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages)) - 1e100
        def max(a,b):
            return a if a>b else b
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] = max(images[c,s1+u1,s2+u2,i],targets[c,o1,o2,i])
                                except IndexError:
                                    pass #?                           
                        o2 += 1
                    o1 += 1
        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])    

    @staticmethod
    def ConvOut(images, hidActs, targets = None, moduleStride=1, paddingStart = 0, partial_sum = 0,filterSizeX=None):
        # print "hiiiii", filterSizeX
        assert filterSizeX != None
        assert paddingStart>=0
        paddingStart = -paddingStart


        images_shape = images.shape
        hidActs_shape = hidActs.shape
        images = images.reshape(images_shape[0],-1).T.reshape(images_shape[1],images_shape[2],images_shape[3],images_shape[0])
        hidActs = hidActs.reshape(hidActs_shape[0],-1).T.reshape(hidActs_shape[1],hidActs_shape[2],hidActs_shape[3],hidActs_shape[0])

        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numFilterChannels = numChannels
        ####filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)
        # print numFilterChannels, filterSizeX, filterSizeX, numFilters
        # filterSizeX = imSizeX - (numModulesX - 1) * moduleStride + 2*abs(paddingStart)

        if targets == None:
            targets = np.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))
        else: 
            assert targets.shape == (numFilterChannels, filterSizeX, filterSizeX, numFilters)
        numImgColors = numChannels
        images2 = np.zeros((numChannels,imSizeX+2*abs(paddingStart),imSizeX+2*abs(paddingStart),numImages))
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = targets.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)              
            
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1)
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:].T
                filters_2d[:] += np.dot(images_patch,hidden_patch)

        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])     


    @staticmethod
    def ConvDown(hidActs, filters, moduleStride , paddingStart, imSizeX):
        assert paddingStart>=0
        paddingStart = -paddingStart

        hidActs_shape = hidActs.shape
        filters_shape = filters.shape
        hidActs = hidActs.reshape(hidActs_shape[0],-1).T.reshape(hidActs_shape[1],hidActs_shape[2],hidActs_shape[3],hidActs_shape[0])
        filters = filters.reshape(filters_shape[0],-1).T.reshape(filters_shape[1],filters_shape[2],filters_shape[3],filters_shape[0])

        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        
        if imSizeX == "auto":
            imSizeX = moduleStride*(numModulesX - 1) + filterSizeX
        else: 
            imSizeX = imSizeX + 2*abs(paddingStart)

        targets2 = np.zeros((numFilterChannels, imSizeX, imSizeX, numImages))
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX-filterSizeX+1,moduleStride):
                targets2_patch = targets2[:,i:i+filterSizeX,j:j+filterSizeX,:]
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:]
                targets2_patch[:] += np.dot(hidden_patch.T,filters_2d.T).T.reshape(numFilterChannels,filterSizeX,filterSizeX,-1)
                
        targets = targets2[:,
                abs(paddingStart):imSizeX-abs(paddingStart),
                abs(paddingStart):imSizeX-abs(paddingStart),
                :] 

        targets_shape = targets.shape
        return targets.reshape(-1,targets_shape[3]).T.reshape(targets_shape[3],targets_shape[0],targets_shape[1],targets_shape[2])  


    @staticmethod
    def DeConvUp(images, filters, moduleStride , paddingStart, imSizeX):
        return NumpyConvBackend.ConvDown(images, filters, moduleStride , paddingStart, imSizeX)

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class CudaconvnetBackendReversed(GnumpyBackend):
    @staticmethod
    def ConvUp(images, filters, targets, moduleStride, paddingStart):
        assert paddingStart>=0
        paddingStart = -paddingStart

        if filters.shape[3]==1: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:1]=filters
                out = ConvNet.convUp(images, filters_16, targets, moduleStride = moduleStride, paddingStart = paddingStart)
                return out[:1,:,:,:]
        elif filters.shape[3]==3: 
                filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
                filters_16[:,:,:,:3]=filters
                out = ConvNet.convUp(images, filters_16, targets, moduleStride = moduleStride, paddingStart = paddingStart)
                return out[:3,:,:,:]                
        elif filters.shape[3]%16==0: 
                return ConvNet.convUp(images, filters, targets, moduleStride, paddingStart)
        else: raise Exception("Filters Mode 16")   

    @staticmethod
    def DeConvDown(hidActs, filters, targets, moduleStride, paddingStart):
        return CudaconvnetBackendReversed.ConvUp(hidActs, filters, targets, moduleStride, paddingStart)

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride, paddingStart, imSizeX):
        assert paddingStart>=0
        paddingStart = -paddingStart

        if filters.shape[3]==1 and hidActs.shape[0]==1: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:1] = filters
            return ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart, imSizeX = imSizeX)
        elif filters.shape[3]==3 and hidActs.shape[0]==3: 
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            filters_16 = gp.zeros((filters.shape[0],filters.shape[1],filters.shape[2],16))
            filters_16[:,:,:,:3] = filters
            return ConvNet.convDown(hidActs_16, filters_16 , moduleStride=moduleStride , paddingStart = paddingStart, imSizeX = imSizeX)            
        elif filters.shape[3]%16==0 and hidActs.shape[0]%16==0:
            return ConvNet.convDown(hidActs, filters, moduleStride, paddingStart,imSizeX)
        else: raise Exception("Hidden or Filters Mode 16")

    @staticmethod
    def ConvOut(images, hidActs, targets, moduleStride, paddingStart, partial_sum,filterSizeX=None):
        assert filterSizeX !=None
        assert paddingStart>=0
        paddingStart = -paddingStart

        if hidActs.shape[0]==1:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:1,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX = filterSizeX)
            return out[:,:,:,:1]
        elif hidActs.shape[0]==3:
            hidActs_16 = gp.zeros((16,hidActs.shape[1],hidActs.shape[2],hidActs.shape[3]))
            hidActs_16[:3,:,:,:] = hidActs
            out = ConvNet.convOutp(images, hidActs_16, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX = filterSizeX)
            return out[:,:,:,:3]            
        elif hidActs.shape[0]%16 == 0:
            # t = time.time()
            temp = ConvNet.convOutp(images = images, hidActs=hidActs, targets = targets, moduleStride = moduleStride, paddingStart = paddingStart, partial_sum = partial_sum,filterSizeX = filterSizeX)
            # print "nn_utils",1000*(time.time()-t)
            return temp
        else: raise Exception("Hidden Mode 16")  
            
    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        return ConvNet.MaxPool(images,subsX,startX,strideX,outputsX)                      

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        return ConvNet.MaxPoolUndo(images,grad,maxes,subsX,startX,strideX)  

    @staticmethod
    def rnorm(images,N,addScale,powScale,blocked):
        return ConvNet.ResponseNorm(images,N,addScale,powScale,blocked,minDiv)

    @staticmethod
    def DeConvUp(images, filters, moduleStride , paddingStart):
        return CudaconvnetBackendReversed.ConvDown(images, filters, moduleStride , paddingStart)        
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

class NumpyConvBackendReversed(object):
    @staticmethod
    def ConvUp(images, filters, targets, moduleStride, paddingStart):
        assert paddingStart>=0
        paddingStart = -paddingStart

        global images2
        assert paddingStart <= 0
        numChannels, imSizeX, imSizeX, numImages = images.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        # assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        # assert        (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride+1
        targets = np.zeros((numFilters, numModulesX, numModulesX, numImages))
        images2 = np.zeros((numChannels, 
                            imSizeX+2*abs(paddingStart), 
                            imSizeX+2*abs(paddingStart), 
                            numImages))  
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1).T
                targets_patch = np.dot(images_patch,filters_2d)
                targets[:,i/moduleStride,j/moduleStride,:]=targets_patch.T
                
        return targets        

    @staticmethod
    def AvgPoolUndo(images,grad,subsX,startX,strideX): 
        assert paddingStart>=0
        paddingStart = -paddingStart

        numChannels, imSizeX_, imSizeX, numImages = images.shape    
        assert imSizeX_ == imSizeX
        numImgColors = numChannels
        numChannels, outputsX_, outputsX, numImages = grad.shape
        assert outputsX_ == outputsX
        targets = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    #if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                    targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]

                                except IndexError:
                                    pass #??  I don't fucking get it.
                        o2 += 1
                    o1 += 1
        return targets/subsX**2


    @staticmethod
    def AvgPool(images,subsX,startX,strideX,outputsX):        
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages))
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] += images[c,s1+u1,s2+u2,i]                           
                                except IndexError:
                                    pass #?
                        o2 += 1
                    o1 += 1
        return 1.0*targets/subsX**2

    @staticmethod
    def MaxPoolUndo(images,grad,maxes,subsX,startX,strideX):
        numChannels, imSizeX_, imSizeX, numImages = images.shape    
        assert imSizeX_ == imSizeX
        numImgColors = numChannels
        numChannels, outputsX_, outputsX, numImages = maxes.shape
        assert outputsX_ == outputsX 
        assert maxes.shape == grad.shape
        targets = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX): ######why
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    if maxes[c,o1,o2,i]==images[c,s1+u1,s2+u2,i]:
                                        targets[c,s1+u1,s2+u2,i]+=grad[c,o1,o2,i]
                                        break
                                except IndexError:
                                    pass #??  I don't fucking get it.                           
                            else: continue
                            break
                        o2 += 1
                    o1 += 1
        return targets

    @staticmethod
    def MaxPool(images,subsX,startX,strideX,outputsX):
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numImgColors = numChannels
        targets = np.zeros((numChannels, outputsX, outputsX, numImages)) - 1e100
        def max(a,b):
            return a if a>b else b
        for i in range(numImages):
            for c in range(numChannels):
                o1 = 0
                for s1 in range(startX, imSizeX, strideX):
                    if s1<0:
                        continue
                    o2 = 0
                    for s2 in range(startX, imSizeX, strideX):
                        if s2<0:
                            continue
                        for u1 in range(subsX):
                            for u2 in range(subsX):
                                try:
                                    targets[c,o1,o2,i] = max(images[c,s1+u1,s2+u2,i],targets[c,o1,o2,i])
                                except IndexError:
                                    pass #?                           
                        o2 += 1
                    o1 += 1
        return targets

    @staticmethod
    def ConvOut(images, hidActs, targets = None, moduleStride=1, paddingStart = 0, partial_sum = 0,filterSizeX=None):
        assert filterSizeX!=None
        assert paddingStart>=0
        paddingStart = -paddingStart        
        assert filterSizeX!=None
        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        numFilterChannels = numChannels
        ####filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)
        # print numFilterChannels, filterSizeX, filterSizeX, numFilters
        # filterSizeX = imSizeX - (numModulesX - 1) * moduleStride + 2*abs(paddingStart)

        if targets == None:
            targets = np.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))
        else: 
            assert targets.shape == (numFilterChannels, filterSizeX, filterSizeX, numFilters)
        numImgColors = numChannels
        images2 = np.zeros((numChannels,imSizeX+2*abs(paddingStart),imSizeX+2*abs(paddingStart),numImages))
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images

        numChannels, imSizeX2, imSizeX2, numImages = images2.shape
        filters_2d = targets.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)              
            
        for i in range(0,imSizeX2-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX2-filterSizeX+1,moduleStride):
                images_patch = images2[:,i:i+filterSizeX,j:j+filterSizeX,:].reshape(numChannels*filterSizeX**2,-1)
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:].T
                filters_2d[:] += np.dot(images_patch,hidden_patch)

        return targets

    @staticmethod
    def ConvDown(hidActs, filters, moduleStride , paddingStart):
        assert paddingStart>=0
        paddingStart = -paddingStart

        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        imSizeX = moduleStride*(numModulesX - 1) + filterSizeX
        targets2 = np.zeros((numFilterChannels, imSizeX, imSizeX, numImages))
        filters_2d = filters.reshape(numFilterChannels*filterSizeX*filterSizeX,numFilters)
        
        for i in range(0,imSizeX-filterSizeX+1,moduleStride):
            for j in range(0,imSizeX-filterSizeX+1,moduleStride):
                targets2_patch = targets2[:,i:i+filterSizeX,j:j+filterSizeX,:]
                hidden_patch = hidActs[:,i/moduleStride,j/moduleStride,:]
                targets2_patch[:] += np.dot(hidden_patch.T,filters_2d.T).T.reshape(numFilterChannels,filterSizeX,filterSizeX,-1)
                
        targets = targets2[:,
                abs(paddingStart):imSizeX-abs(paddingStart),
                abs(paddingStart):imSizeX-abs(paddingStart),
                :] 

        return targets

    @staticmethod
    def ConvUp_old(images, filters, moduleStride, paddingStart):
        global images2
        assert paddingStart <= 0
        numChannels, imSizeX, imSizeX, numImages = images.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape
        assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        assert        (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
        numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride+1
        numModules = numModulesX**2 
        numGroups = 1
        targets = np.zeros((numFilters, numModulesX, numModulesX, numImages))
        images2 = np.zeros((numChannels, 
                            imSizeX+2*abs(paddingStart), 
                            imSizeX+2*abs(paddingStart), 
                            numImages))  
        if paddingStart != 0:
            images2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] = images
        else:
            images2 = images
        for i in range(numImages):
            for f in range(numFilters):
                for c in range(numChannels):
                    for y1 in range(numModulesX):
                        for y2 in range(numModulesX):
                            for u1 in range(filterSizeX):
                                for u2 in range(filterSizeX):
                                    x1 = y1*moduleStride + u1 
                                    x2 = y2*moduleStride + u2
                                    targets[f, y1, y2, i] += filters[c ,u1,u2,f] * images2[c,x1,x2,i]
        return targets


    @staticmethod
    def ConvDown_old(hidActs, filters, moduleStride , paddingStart):
        numGroups = 1
        assert paddingStart <= 0
        numFilters, numModulesX, numModulesX, numImages = hidActs.shape
        numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

        imSizeX = moduleStride*(numModulesX - 1) - 2*abs(paddingStart) + filterSizeX
        imSizeX2 = moduleStride*(numModulesX - 1) + filterSizeX

        numChannels = numFilterChannels * numGroups
        numModules = numModulesX**2 
        
        targets = np.zeros((numChannels, imSizeX, imSizeX, numImages))
        targets2 = np.zeros((numChannels, imSizeX2, imSizeX2, numImages))
        
        for i in range(numImages):
            for f in range(numFilters):
                for c in range(numChannels):
                    for y1 in range(numModulesX):
                        for y2 in range(numModulesX):
                            for u1 in range(filterSizeX):
                                for u2 in range(filterSizeX):
                                    x1 = y1*moduleStride + u1 
                                    x2 = y2*moduleStride + u2
                                    targets2[c,x1,x2,i] += filters[c ,u1,u2,f] * hidActs[f, y1, y2, i]
        if paddingStart != 0:
            targets[:] = targets2[:, 
                abs(paddingStart):-abs(paddingStart),
                abs(paddingStart):-abs(paddingStart),
                :] 
        else:
            targets = targets2
        return targets

    @staticmethod
    def ConvOut_old(images, hidActs, moduleStride=1,paddingStart = 0):
            numGroups = 1
            assert paddingStart <= 0
            numFilters, numModulesX, numModulesX, numImages = hidActs.shape
            numChannels, imSizeX, imSizeX, numImages = images.shape    
            numFilterChannels = numChannels / numGroups

            filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)
            targets = np.zeros((numFilterChannels, filterSizeX, filterSizeX, numFilters))
            numImgColors = numChannels
            images2 = np.zeros((numChannels,imSizeX+2*abs(paddingStart),imSizeX+2*abs(paddingStart),numImages))
            if paddingStart != 0:
                images2[:, 
                    abs(paddingStart):-abs(paddingStart),
                    abs(paddingStart):-abs(paddingStart),
                    :] = images
            else:
                images2 = images
            for i in range(numImages):
                for f in range(numFilters):
                    for c in range(numChannels):
                        for y1 in range(numModulesX):
                            for y2 in range(numModulesX):
                                for u1 in range(filterSizeX):
                                    for u2 in range(filterSizeX):
                                        x1 = y1*moduleStride + u1 
                                        x2 = y2*moduleStride + u2
                                        targets[c ,u1,u2,f] += hidActs[f, y1, y2, i] * images2[c,x1,x2,i]
            return targets

    @staticmethod
    def rnorm(images,N,addScale,powScale,blocked,minDiv):
        numChannels, imSizeX, imSizeX, numImages = images.shape    
        targets = np.zeros(images.shape)
        denom = np.zeros(images.shape)
        for i in range(numImages):
            for c in range(numChannels):
                for x in range(0, imSizeX):
                    for y in range(0, imSizeX):
                        if block==0:
                            start_ = -int(N/2) + c
                            start = max(0,start_)
                            end = min(numChannels,start+N)  
                        elif block==1:
                            start_ = int(c / N) * N
                            start =  start_
                            end = min(numChannels,start+N)  
                        for j in range(start,end):
                            denom[c,x,y,i] += images[j,x,y,i]**2
                        denom[c,x,y,i] = (minDiv+addScale*denom[c,x,y,i])
                        targets[c,x,y,i] = images[c,x,y,i]*denom[c,x,y,i]**(-powScale)
        return targets,denom              