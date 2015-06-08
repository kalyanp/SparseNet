### goal: write cudamat and gnumpy functions that 


import ctypes as ct

try:
    _ConvNet = ct.cdll.LoadLibrary('../gnumpy/libcudamat_conv.so')
except OSError:
    try:
        _ConvNet = ct.cdll.LoadLibrary('./gnumpy/libcudamat_conv.so')
    except OSError:
        _ConvNet = ct.cdll.LoadLibrary('/home/alireza/Dropbox/work/gnumpy/libcudamat_conv.so')


import numpy as np
import gnumpy as g
import time
from numpy import prod



def convUp(images, filters, targets, moduleStride = 1, paddingStart = 0):
    assert paddingStart <= 0
    #numImages , imSizeX, imSizeX, numChannels = images.shape
    #numFilters, filterSizeX, filterSizeX, numFilterChannels = filters.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    assert (numChannels == numFilterChannels)
    # assert (2*abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0

    numModulesX = (2*abs(paddingStart) + imSizeX - filterSizeX)/moduleStride + 1
    numModules = numModulesX**2 

    numGroups = 1
    #moduleStride = 1  

    #targets = g.zeros((numImages, numModulesX, numModulesX, numFilters))
    if targets==None:
        targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))

    numImgColors = numChannels






    imagesCu = images._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


    imagesCu_orig, filtersCu_orig, targetsCu_orig = [tuple(x.contents.size) for x in (imagesCu, filtersCu, targetsCu)]


    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilters
    filtersCu.contents.size[1] = numFilterChannels * filterSizeX**2
    #print filtersTotSize,prod(filtersCu.contents.size),filtersCu,filtersCu.contents.size[0],filtersCu.contents.size[1]
    # assert filtersTotSize == prod(filtersCu.contents.size)

    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numImgColors * imSizeX**2
    #assert imagesTotSize == prod(imagesCu.contents.size)


    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numFilters * numModulesX**2
    #assert targetsTotSize == prod(targetsCu.contents.size) 


    _ConvNet.convUp(imagesCu,
                   filtersCu,
                   targetsCu,

                   numModulesX,
                   paddingStart,
                   moduleStride,
                   numImgColors,  

                   numGroups,       
                   )



    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        imagesCu.contents.size[i] = imagesCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets





def convDown(hidActs, filters, moduleStride, paddingStart, imSizeX):
    numGroups = 1

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    #assert (numChannels == numFilterChannels)

    numModules = numModulesX**2 
    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    if imSizeX=="auto":    
        imSizeX = (numModulesX - 1) * moduleStride - 2*abs(paddingStart) + filterSizeX

    #assert (abs(paddingStart) + imSizeX - filterSizeX) % moduleStride == 0
    numChannels = numFilterChannels * numGroups

    targets = g.zeros((numChannels, imSizeX, imSizeX, numImages))

    numImgColors = numChannels

    hidActsCu = hidActs._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


# * hidActs:     (numFilters, numModules, numImages)
# * filters:     (numFilterColors, filterPixels, numFilters)               if conv
# *              (numModules, numFilterColors, filterPixels, numFilters)   otherwise
# * targets:     (numImageColors, imgPixels, numImages)

    
    hidActsCu_orig, filtersCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (hidActsCu, filtersCu, targetsCu)]

    # filters are as before    
    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilters 
    filtersCu.contents.size[1] = numFilterChannels * filterSizeX**2
    #assert filtersTotSize == prod(filtersCu.contents.size)
    
    # hidActs are like the targets of the past:
    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numImages
    hidActsCu.contents.size[1] = numFilters * numModulesX**2
    #assert hidActsTotSize == prod(hidActsCu.contents.size) 

    # targets are like images:
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numImgColors * imSizeX**2
    #assert targetsTotSize == prod(targetsCu.contents.size)


    _ConvNet.convDown(
		       hidActsCu,
		       filtersCu,
		       targetsCu,

		       imSizeX,
		       paddingStart,
		       moduleStride,
		       numImgColors,
		       numGroups)

    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets


def convOutp(images, hidActs, targets = None, moduleStride = 1, paddingStart = 0, partial_sum = 0, filterSizeX = None):
    assert filterSizeX != None
    numGroups = 1
    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups
    if partial_sum == 0: partial_sum = numModulesX**2
    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    #filterSizeX = (imSizeX - numModulesX + abs(paddingStart))/moduleStride + 1
    ###### filterSizeX = imSizeX - moduleStride*(numModulesX - 1) + 2*abs(paddingStart)


    # assert partialSum is None
    # partialSum = numModulesX**2 
    # t = time.time()
    # print targets
    # print "hi"
    # print imSizeX,numModulesX,moduleStride,paddingStart
    # filterSizeX = imSizeX - (numModulesX - 1) * moduleStride + 2*abs(paddingStart)
    # print filterSizeX

    targets_temp = g.zeros(( (numModulesX**2)/partial_sum * numFilterChannels * filterSizeX * filterSizeX, numFilters))


    # print "jimmy_zeros",1000*(time.time()-t)
    # print targets_temp.shape
    numImgColors = numChannels

    hidActsCu = hidActs._base.p_mat
    imagesCu = images._base.p_mat
    targetsCu = targets_temp._base.p_mat

    imagesCu = images._base.p_mat
    imagesCu_orig, hidActsCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (imagesCu, hidActsCu, targetsCu)]

    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numImgColors * imSizeX**2
    #assert imagesTotSize == prod(imagesCu.contents.size)

    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numImages
    hidActsCu.contents.size[1] = numFilters * numModulesX**2
    #assert hidActsTotSize == prod(hidActsCu.contents.size) 


    targetsTotSize = targets_temp.size
    targetsCu.contents.size[0] = numFilters
    targetsCu.contents.size[1] = numFilterChannels * filterSizeX**2 * (numModulesX**2)/partial_sum
    #assert targetsTotSize == prod(targetsCu.contents.size)
    # print targetsCu.contents.size[0],targetsCu.contents.size[1]
    # t = time.time()
    _ConvNet.convOutp(
        imagesCu,
        hidActsCu,
        targetsCu,

        numModulesX,
        filterSizeX,
        paddingStart,
        moduleStride,
        numImgColors,
        numGroups,
        partial_sum
        )
    # print "jimmy",1000*(time.time()-t)
    for i in range(2):
        imagesCu.contents.size[i] = imagesCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    targets_temp = targets_temp.reshape((numModulesX**2)/partial_sum,numFilterChannels*filterSizeX*filterSizeX*numFilters)
    # print numModulesX**2
    # print "ali",targets_temp.shape
    targets_temp = targets_temp.sum(0)
    # print targets_temp.shape
    # print numFilterChannels,filterSizeX,filterSizeX,numFilters

    if targets == None:
        return targets_temp.reshape(numFilterChannels,filterSizeX,filterSizeX,numFilters)
    else: 
        targets[:] = targets_temp.reshape(numFilterChannels,filterSizeX,filterSizeX,numFilters)
        return targets



def MaxPool(images, 
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numImgColors = numChannels

    targets = g.zeros((numChannels, outputsX, outputsX, numImages))

    
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    from pylab import prod
    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numImgColors * imSizeX**2
    #assert imagesTotSize == prod(imagesCu.contents.size)

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numImgColors * outputsX**2
    #assert targetsTotSize == prod(targetsCu.contents.size)

    numFilters = numImgColors

    _ConvNet.MaxPool(imagesCu,
                  targetsCu,
                  numFilters,
                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]

    return targets












def MaxPoolUndo(images, 
                grad,
                maxes,


                subsX,
                startX,
                strideX,
       ):
    
    numChannels, imSizeX_, imSizeX, numImages = images.shape    
    assert imSizeX_ == imSizeX
    numChannels = numChannels

    numChannels, outputsX_, outputsX, numImages = maxes.shape
    assert outputsX_ == outputsX
    
    assert maxes.shape == grad.shape 
    targets = g.zeros(images.shape)

    #Alireza
    assert numChannels % 16 == 0
    

    imagesCu = images._base.p_mat
    maxesCu = maxes._base.p_mat
    gradCu = grad._base.p_mat
    targetsCu = targets._base.p_mat


    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numChannels * imSizeX**2
    # assert imagesTotSize == prod(imagesCu.contents.size)

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numChannels * imSizeX**2
    #assert targetsTotSize == prod(targetsCu.contents.size)



    maxesCu_orig = tuple(maxesCu.contents.size)
    maxesTotSize = maxes.size
    maxesCu.contents.size[0] = numImages
    maxesCu.contents.size[1] = numChannels * outputsX**2
    #assert maxesTotSize == prod(maxesCu.contents.size)

    gradCu_orig = tuple(gradCu.contents.size)
    gradTotSize = grad.size
    gradCu.contents.size[0] = numImages
    gradCu.contents.size[1] = numChannels * outputsX**2
    #assert gradTotSize == prod(gradCu.contents.size)


    _ConvNet.MaxPoolUndo(imagesCu,
                     gradCu,
                     maxesCu,
                     targetsCu,

                  subsX,
                  startX,
                  strideX,
                  outputsX
                  )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]
        gradCu.contents.size[i]=gradCu_orig[i]
        maxesCu.contents.size[i]=maxesCu_orig[i]


    return targets




## dosen't work for some reason.  Investigate the reason.
def AvgPool(images, 
            subsX,
            startX,
            strideX,
            outputsX
       ):
    
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numImgColors = numChannels

    targets = g.zeros((numChannels, outputsX, outputsX, numImages))

    
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numImgColors * imSizeX**2
    #assert imagesTotSize == prod(imagesCu.contents.size)

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numImgColors * outputsX**2
    #assert targetsTotSize == prod(targetsCu.contents.size)

    numFilters = numImgColors

    _ConvNet.AvgPool(imagesCu,
                     targetsCu,
                     numFilters,
                     subsX,
                     startX,
                     strideX,
                     outputsX,
                     #subsX**2,
                     )

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]

    return targets


def AvgPoolUndo(images,grad,subsX,startX,strideX):
    
    numChannels, imSizeX_, imSizeX, numImages = images.shape    
    assert imSizeX_ == imSizeX
    numChannels = numChannels

    numChannels, outputsX_, outputsX, numImages = grad.shape
    assert outputsX_ == outputsX
    
    targets = g.zeros(images.shape)
    imgSize = imSizeX

    assert numChannels % 16 == 0
    
    gradCu = grad._base.p_mat
    targetsCu = targets._base.p_mat

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numChannels * imSizeX**2
    # assert targetsTotSize == prod(targetsCu.contents.size)

    gradCu_orig = tuple(gradCu.contents.size)
    gradTotSize = grad.size
    gradCu.contents.size[0] = numImages
    gradCu.contents.size[1] = numChannels * outputsX**2
    # assert gradTotSize == prod(gradCu.contents.size)

    _ConvNet.AvgPoolUndo(gradCu,targetsCu,subsX,startX,strideX,outputsX,imgSize)

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        gradCu.contents.size[i]=gradCu_orig[i]


    return targets


def ResponseNorm(images, N, addScale, powScale, blocked, minDiv):
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    targets = g.zeros(images.shape)   
    denoms = g.zeros(images.shape)


    imagesCu = images._base.p_mat
    denomsCu = denoms._base.p_mat
    targetsCu = targets._base.p_mat

    imagesCu_orig = tuple(imagesCu.contents.size)
    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImages
    imagesCu.contents.size[1] = numChannels * imSizeX**2

    denomsCu_orig = tuple(denomsCu.contents.size)
    denomsTotSize = denoms.size
    denomsCu.contents.size[0] = numImages
    denomsCu.contents.size[1] = numChannels * imSizeX**2

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numChannels * imSizeX**2    

    # blocked = 1
    # num_images = images.shape[0]
    # numpixels = images.shape[1] / numChannels
    # imgsize = int(math.sqrt(numpixels))
    #assert images.shape[1] == numChannels * numpixels
    #assert imgsize * imgsize == numpixels
    #pdb.setrace()
    # _ConvNet.ResponseNorm(imagesCu, denomsCu, targetsCu, numChannels, sizeX, ct.c_float(addScale), ct.c_float(powScale))
    # print "ali",targets[0,:4,:4,0] 
    # print "ali",imagesCu.contents.size[0]
    _ConvNet.convResponseNormCrossMap(imagesCu, denomsCu, targetsCu, ct.c_int(numChannels), ct.c_int(N), ct.c_float(addScale), ct.c_float(powScale), ct.c_float(minDiv), ct.c_bool(blocked));

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        imagesCu.contents.size[i]=imagesCu_orig[i]
        denomsCu.contents.size[i]=denomsCu_orig[i]
    # print "ali",targets[0,:4,:4,0] 


    return targets



def ResponseNormUndo(images_in, images_out, denoms, N, addScale, powScale):
    numChannels, imSizeX, imSizeX, numImages = images_in.shape    

    targets = g.zeros(images_out.shape)
    acts = images_out.copy()
    assert targets.shape == images_in.shape
    assert targets.shape == denoms.shape
    assert targets.shape == images_out.shape    

    images_outCu = images_out._base.p_mat
    images_inCu = images_in._base.p_mat
    denomsCu = denoms._base.p_mat
    targetsCu = targets._base.p_mat
    actsCu = acts._base.p_mat

    images_outCu_orig = tuple(images_outCu.contents.size)
    images_outTotSize = images_out.size
    images_outCu.contents.size[0] = numImages
    images_outCu.contents.size[1] = numChannels * imSizeX**2

    images_inCu_orig = tuple(images_inCu.contents.size)
    images_inTotSize = images_in.size
    images_inCu.contents.size[0] = numImages
    images_inCu.contents.size[1] = numChannels * imSizeX**2

    actsCu_orig = tuple(actsCu.contents.size)
    actsTotSize = acts.size
    actsCu.contents.size[0] = numImages
    actsCu.contents.size[1] = numChannels * imSizeX**2

    denomsCu_orig = tuple(denomsCu.contents.size)
    denomsTotSize = denoms.size
    denomsCu.contents.size[0] = numImages
    denomsCu.contents.size[1] = numChannels * imSizeX**2

    targetsCu_orig = tuple(targetsCu.contents.size)
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImages
    targetsCu.contents.size[1] = numChannels * imSizeX**2    


    _ConvNet.ResponseNormUndo(images_outCu, denomsCu, images_inCu,
               actsCu, targetsCu, numChannels, N,
               ct.c_float(addScale), ct.c_float(powScale))

    for i in range(2):
        targetsCu.contents.size[i]=targetsCu_orig[i]
        images_outCu.contents.size[i]=images_outCu_orig[i]
        images_inCu.contents.size[i]=images_inCu_orig[i]
        denomsCu.contents.size[i]=denomsCu_orig[i]
        actsCu.contents.size[i]=actsCu_orig[i]

    return targets,acts


################################################################


def localUp(images, filters):

    numChannels, imSizeX, imSizeX, numImages = images.shape

    ## this is a hell of a filter-matrix. 
    numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape


    assert numModulesX <= imSizeX


    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)

    paddingStart = -(numModulesX - imSizeX + filterSizeX - 1)
    assert paddingStart <= 0

    numModules = numModulesX**2 



    numGroups = 1
    moduleStride = 1  

    targets = g.zeros((numFilters, numModulesX, numModulesX, numImages))

    numImgColors = numChannels






    imagesCu = images._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


    imagesCu_orig, filtersCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (imagesCu, filtersCu, targetsCu)]

    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    filtersCu.contents.size[1] = numFilters
    assert filtersTotSize == prod(filtersCu.contents.size)

    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    # assert imagesTotSize == prod(imagesCu.contents.size)


    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numFilters * numModulesX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size) 

    _ConvNet.localUp(imagesCu,
                   filtersCu,
                   targetsCu,

                   numModulesX,
                   paddingStart,
                   moduleStride,
                   numImgColors,  

                   numGroups,       
                   )

    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        imagesCu.contents.size[i] = imagesCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets





def localDown(hidActs, filters, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  

    assert paddingStart <= 0
    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numModulesX_, numModulesX_, numFilterChannels, filterSizeX, filterSizeX, numFilters = filters.shape

    ##### I DONT SUPPORT THE FUCKING STRIDE. SHIT.
    assert numModulesX_ == numModulesX
    #numModulesX = (abs(paddingStart) + imSizeX - filterSizeX + 1)
    imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1

    numChannels = numFilterChannels * numGroups

    #paddingStart = -(numModulesX - imSizeX + filterSizeX + 1)

    numModules = numModulesX**2 

    targets = g.zeros((numChannels, imSizeX, imSizeX, numImages))

    numImgColors = numChannels




    hidActsCu = hidActs._base.p_mat
    filtersCu = filters._base.p_mat
    targetsCu = targets._base.p_mat


# * hidActs:     (numFilters, numModules, numImages)
# * filters:     (numFilterColors, filterPixels, numFilters)               if conv
# *              (numModules, numFilterColors, filterPixels, numFilters)   otherwise
# * targets:     (numImageColors, imgPixels, numImages)

    
    hidActsCu_orig, filtersCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (hidActsCu, filtersCu, targetsCu)]

    # filters are as before    
    filtersTotSize = filters.size
    filtersCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    filtersCu.contents.size[1] = numFilters
    assert filtersTotSize == prod(filtersCu.contents.size)
    
    # hidActs are like the targets of the past:
    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numFilters * numModulesX**2
    hidActsCu.contents.size[1] = numImages
    assert hidActsTotSize == prod(hidActsCu.contents.size) 

    # targets are like images:
    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numImgColors * imSizeX**2
    targetsCu.contents.size[1] = numImages
    assert targetsTotSize == prod(targetsCu.contents.size)


    _ConvNet.localDown(
		       hidActsCu,
		       filtersCu,
		       targetsCu,

		       imSizeX,
		       paddingStart,
		       moduleStride,
		       numImgColors,
		       numGroups)

    for i in range(2):
        filtersCu.contents.size[i] = filtersCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets














def localOutp(images, hidActs, paddingStart = 0):
    numGroups = 1
    moduleStride = 1  



    numFilters, numModulesX, numModulesX, numImages = hidActs.shape
    numChannels, imSizeX, imSizeX, numImages = images.shape    
    numFilterChannels = numChannels / numGroups

    #imSizeX = numModulesX - abs(paddingStart) + filterSizeX - 1
    assert paddingStart <= 0
    filterSizeX = imSizeX - numModulesX + abs(paddingStart) + 1

    #assert partialSum is None
    #partialSum = numModulesX**2 

    targets = g.zeros((numModulesX, numModulesX, numFilterChannels, filterSizeX, filterSizeX, numFilters))


    numImgColors = numChannels



    hidActsCu = hidActs._base.p_mat
    imagesCu = images._base.p_mat
    targetsCu = targets._base.p_mat

    imagesCu = images._base.p_mat
    imagesCu_orig, hidActsCu_orig, targetsCu_orig = \
        [tuple(x.contents.size) for x in 
         (imagesCu, hidActsCu, targetsCu)]

    imagesTotSize = images.size
    imagesCu.contents.size[0] = numImgColors * imSizeX**2
    imagesCu.contents.size[1] = numImages
    # assert imagesTotSize == prod(imagesCu.contents.size)

    hidActsTotSize = hidActs.size
    hidActsCu.contents.size[0] = numFilters * numModulesX**2
    hidActsCu.contents.size[1] = numImages
    assert hidActsTotSize == prod(hidActsCu.contents.size) 


    targetsTotSize = targets.size
    targetsCu.contents.size[0] = numFilterChannels * filterSizeX**2 * numModulesX**2
    targetsCu.contents.size[1] = numFilters
    assert targetsTotSize == prod(targetsCu.contents.size)


    _ConvNet.localOutp(
        imagesCu,
        hidActsCu,
        targetsCu,

        numModulesX,
        filterSizeX,
        paddingStart,
        moduleStride,
        numImgColors,
        numGroups,
        )

    for i in range(2):
        imagesCu.contents.size[i] = imagesCu_orig[i]
        hidActsCu.contents.size[i] = hidActsCu_orig[i]
        targetsCu.contents.size[i] = targetsCu_orig[i]

    return targets