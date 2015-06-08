import nn_utils as nn
import numpy as np
import pickle


class WeightSet:

    @staticmethod
    def prod(lst):
        return np.prod(np.array(lst))

    def __init__(self,shape, tied_list=None, tied_type = "first"):
        self.shape = shape
        self.index = []

        tied_dict = {}
        if tied_list:
            for pairs in tied_list:
                i = pairs[0]
                j = pairs[1]
                if tied_type == "all":
                    assert shape[i] == shape[j]
                    for k in xrange(len(shape[i])):
                        tied_dict[(j,k)]=(i,k)
                elif tied_type == "first":
                    print shape[i][0]
                    print shape[j][0]                    
                    assert shape[i][0] == shape[j][0]


                    tied_dict[(j,0)]=(i,0)

        i = 0
        for layer_index in xrange(len(self.shape)):
            layer = self.shape[layer_index]
            if layer in [None,[None],[None,None]]:
                self.index.append(None)
                continue
            self.index.append([None]*len(layer))           
            for w_index in xrange(len(layer)):
                w = layer[w_index]

                tied = tied_dict.get((layer_index,w_index))
                if tied == None:
                    self.index[layer_index][w_index] = (i,i+WeightSet.prod(w))
                    i+=WeightSet.prod(w)                    
                else:
                    self.index[layer_index][w_index] = self.index[tied[0]][tied[1]]
                
        # print "shape",self.shape
        # print "index",self.index

        self.num_weights = self.index[-1][-1][-1]
        self.mem = nn.zeros(self.num_weights) 

        self.weights = [];
        for layer_index in xrange(len(self.shape)):
            self.weights.append(self.get_weights(layer_index))


        # print self.index
        # print self.shape


    def __getitem__(self,i):
        return self.weights[i]

    def get_weights(self,layer_index):
        layer = self.shape[layer_index]
        if layer in [None,[None],[None,None]]:
            return layer
        index = self.index[layer_index]
        lst = []
        for i in xrange(len(layer)):
            w_shape = layer[i]
            w_index = index[i]
            # print w_shape,w_index
            w = self.mem[w_index[0]:w_index[1]].reshape(w_shape)
            lst.append(w)
        if len(lst)==1: return lst[0]
        else: return tuple(lst)   

    
    @classmethod
    def clone_shape(cls,weight_set):
        weigth_set_new = cls(weight_set.shape)
        return weigth_set_new

    @classmethod
    def from_file(cls,address):
        fileObject = open(address,'r')         
        mem = pickle.load(fileObject)
        shape = pickle.load(fileObject)
        weigth_set_new = cls(shape)
        weigth_set_new.mem[:] = nn.array(mem)
        return weigth_set_new

    def load(self,address):
        fileObject = open(address,'r')         
        mem = nn.backend_array(pickle.load(fileObject))
        shape = pickle.load(fileObject)
        assert self.shape == shape
        self.mem[:] = mem

    def save(self,address):
        fileObject = open(address,'wb')   
        pickle.dump(nn.array(self.mem),fileObject)   
        pickle.dump(self.shape,fileObject) 
        fileObject.close()         



    def randn(self,k=1.0):
        nn.fill_randn(self.mem)
        self.mem *= k

    def rand(self,a=0.0,b=1.0):
        nn.fill_rand(self.mem)
        self.mem *= (b-a)
        self.mem += a        

    def make_tied(self,i,j):
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        
        if wk0.ndim==2:
            wk = wk0 + wk1.T
            wk0[:]  = wk
            wk1[:]  = wk.T
        else:
            wk = wk0 + wk1
            wk0[:]  = wk
            wk1[:]  = wk            
            # w=nn.array(wk1)
            # for ch in xrange(wk1.shape[0]):
            #     for f in range(wk1.shape[1]):
            #         x = w[ch,f,:,:]
            #         x_flip = np.flipud(np.fliplr(x))
            #         wk0[f,ch,:,:] += nn.garray(x_flip)
            #         x_flip = np.flipud(np.fliplr(wk0[f,ch,:,:].as_numpy_array()))
            #         wk1[ch,f,:,:] = x_flip
            #show_filters(w2,(4,4))
            # if type(wk1) == gp.garray: w2=gp.garray(w2)
            # return w2
      
    def make_tied_copy(self,i,j):       
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        if wk0.ndim==2:
            wk1[:] = wk0.T
            # wk1[:] = wk0.T
        else:
            wk1[:] = wk0          
            # w=wk1.as_numpy_array()
            # for ch in xrange(wk1.shape[0]):
            #     for f in range(wk1.shape[1]):
            #         x = w[ch,f,:,:]
            #         x_flip = np.flipud(np.fliplr(x))
            #         wk0[f,ch,:,:] = nn.garray(x_flip)


    def __iadd__(self, other):
        if isinstance(other, WeightSet): nn.iadd(self.mem,other.mem); return self
        else: nn.iadd(self.mem,other); return self
        
    def __isub__(self, other):
        if isinstance(other, WeightSet): nn.isub(self.mem,other.mem); return self
        else: nn.isub(self.mem,other); return self
    
    def __imul__(self, other):
        if isinstance(other, WeightSet): nn.imul(self.mem,other.mem); return self
        else: nn.imul(self.mem,other); return self
    
    def __add__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem + other.mem)
        else:return WeightSet(self.cfg,self.mem + other)
    
    __radd__=__add__
    
    def __sub__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem - other.mem)
        else:return WeightSet(self.cfg,self.mem - other)
    
    __rsub__=__sub__
    
    def __mul__(self, other):
        if isinstance(other, WeightSet):return WeightSet(self.cfg,self.mem * other.mem)
        else: return WeightSet(self.cfg,self.mem * other)
    
    __rmul__=__mul__
