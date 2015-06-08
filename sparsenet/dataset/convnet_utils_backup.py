import nn_utils as nn
import numpy as np
from collections import defaultdict
import time
import os
import gnumpy as gp


class NeuralNetLayerCfg(object):
    def __init__(self, dropout, k_sparsity, j_sparsity, l2, want_train = True):
        self.type = None
        self.activation = None
        # self.activation_prime = None
        self.num_filters = None
        self.filter_shape = None
        self.filter_size = None  
        self.num_weights = None
        self.shape = None #to be determined
        self.size = None #to be determined
        self.initW = None
        self.initB = None
        self.padding = None
        self.stride = None        
        self.k_sparsity = k_sparsity
        self.j_sparsity = j_sparsity
        self.l2 = l2         
        self.dropout = dropout
        self.want_train = want_train
        self.spatial_dropout = None

class NeuralNetLayerPoolingCfg(NeuralNetLayerCfg):
    def __init__(self, mode, pooling_width, stride, padding, dropout, l2):
        NeuralNetLayerCfg.__init__(self,dropout, None, None, l2)
        self.type="pooling"
        self.mode = mode
        assert mode == "avg" or mode == "max"
        self.pooling_width = pooling_width
        self.stride = stride
        self.padding = padding
        self.num_weights = 0

    def fprop(self, H):
        assert self.padding==0
        if self.mode == "max": return nn.MaxPool(H, self.pooling_width, self.padding, self.stride, self.shape[1])
        elif self.mode == "avg": return nn.AvgPool(H, self.pooling_width, self.padding, self.stride, self.shape[1])
    def applyPoolingUndo(self, H, delta, H_next):
        assert self.padding==0
        if self.mode == "max": return nn.MaxPoolUndo(H, delta, H_next, self.pooling_width, self.padding, self.stride)
        elif self.mode == "avg": return nn.AvgPoolUndo(H, delta, self.pooling_width, self.padding, self.stride)


class NeuralNetLayerDenseCfg(NeuralNetLayerCfg):
    def __init__(self,num_filters,activation, initW, initB, dropout, k_sparsity, j_sparsity, l2):
        NeuralNetLayerCfg.__init__(self,dropout, k_sparsity, j_sparsity, l2)
        self.type = "dense"
        self.num_filters = num_filters
        self.activation = activation
        # self.activation_prime = nn.make_activation(activation)        
        self.size = num_filters 
        self.shape = num_filters
        self.initW = initW
        self.initB = initB

    def fprop(self, H_prev, w, b, H, dH):
        if H_prev.ndim==2:
            H_temp = H_prev                    
        else: 
            H_temp = H_prev.reshape(H_prev.shape[0], -1) 
        # print 'hey',H_temp.shape,w.shape,H.shape
        nn.dot(H_temp,w,H)
        nn.iadd(H,b)
        # print H.shape,H
        self.activation(H,H,dH)  
        return H,dH               


class NeuralNetLayerReparameterize(NeuralNetLayerCfg):
    def __init__(self,mode,want_reg):
        NeuralNetLayerCfg.__init__(self,None, None, None, None)
        self.type = "reparam"
        self.num_weights = 0
        self.mode = mode
        self.want_reg = want_reg

    def fprop(self, H_prev, H, dH):
        dH = None
        self.mu = H_prev[:,:self.size]
        self.log_sigma = H_prev[:,self.size:]

        nn.fill_randn(self.eps)
        # self.eps= nn.ones(self.eps.shape)

        nn.exp(self.log_sigma,self.sigma)            

        H[:] = self.mu + self.sigma*self.eps

        return H,dH,self.sigma,self.mu               

    def bprop(self,delta,dH_prev,delta_prev):
        num_z = self.shape

        delta_prev[:,:num_z] = delta
        delta_prev[:,num_z:] = self.sigma*self.eps*delta 

        # else:
        #     delta_prev[:,num_z:] = self.eps*delta 

        if self.want_reg:
            delta_prev[:,:num_z] += self.want_reg*self.mu            
            delta_prev[:,num_z:] += self.want_reg*(self.sigma**2 - 1)

        delta_prev *= dH_prev

class NeuralNetLayerConvolutionCfg(NeuralNetLayerCfg):
    def __init__(self, num_filters, activation, filter_width, stride, padding, initW, initB, dropout, l2, k_sparsity, j_sparsity, partial_sum  = 1):
        NeuralNetLayerCfg.__init__(self,dropout, k_sparsity, j_sparsity, l2)
        self.type = "convolution"
        self.partial_sum = partial_sum
        self.num_filters = num_filters
        self.activation = activation
        # self.activation_prime = nn.make_activation(activation)        
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.initW = initW
        self.initB = initB

    def fprop(self, H_prev, w, b, H, dH):
            H = nn.ConvUp(H_prev, w, moduleStride = self.stride, paddingStart = self.padding)        
            H += b.reshape(1,-1,1,1)   
            self.activation(H,H,dH) 
            return H,dH

    def applyConvDown(self, delta, w):
        return nn.ConvDown(delta, w, moduleStride = self.stride, paddingStart = self.padding)
    def applyConvOut(self, H, delta):
        return nn.ConvOut(H, delta, moduleStride = self.stride, paddingStart = self.padding, partial_sum = self.partial_sum, filterSizeX = self.filter_shape[2])     

class NeuralNetLayerResponseNormCfg(NeuralNetLayerCfg):
    def __init__(self, N, addScale, powScale, blocked = 1, minDiv = 1):
        NeuralNetLayerCfg.__init__(self, dropout=None, k_sparsity=None, j_sparsity=None, l2=None)
        self.type = "reparam"
        self.N = N
        self.addScale = addScale
        self.powScale = powScale
        self.blocked = blocked
        self.minDiv = minDiv
        self.num_weights = 0

    def fprop(self, H):
        return nn.rnorm(images=H, N=self.N, addScale=self.addScale, powScale=self.powScale, blocked = self.blocked, minDiv=self.minDiv)



#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#double check l2_norm

class NeuralNetCfg(object):
    def __init__(self, want_dropout=False,want_k_sparsity = False,want_spatial_dropout = False,
                 want_tied=False, tied_list=[(1,2)],num_groups = None, want_layerwise_train = False):
        self._layers = [None,None]
        self.index_layer = 0
        self.index_convolution = []
        self.index_pooling = []
        self.index_dense = []
        self.index_rnorm = []
        self.index_reparam = []
        self.want_dropout = want_dropout
        self.want_tied = want_tied
        self.tied_list = tied_list
        self.want_k_sparsity = want_k_sparsity
        self.want_spatial_dropout = want_spatial_dropout
        self.num_groups = num_groups
        self.want_layerwise_train = want_layerwise_train
            
    def input_conv(self,shape, dropout=None, spatial_dropout = None):
        layer = NeuralNetLayerConvolutionCfg(num_filters=None, activation=None, filter_width=None, stride=None, padding=None, initW=None, initB=None, dropout=dropout, l2=None, k_sparsity = None, j_sparsity = None)
        layer.shape = shape
        layer.spatial_dropout = spatial_dropout        
        layer.size = shape[0]*shape[1]*shape[2]
        self._layers[0] = layer
        self.index_convolution.append(self.index_layer); self.index_layer+=1

    def input_dense(self,shape, dropout=None):
        layer = NeuralNetLayerDenseCfg(num_filters=None, activation=None, initW=None, initB=None, dropout=dropout, k_sparsity=None, j_sparsity=None, l2=None)
        layer.shape = shape
        layer.size = shape
        self._layers[0] = layer
        self.index_dense.append(self.index_layer); self.index_layer+=1        
    
    def convolution(self, num_filters, activation, filter_width=None, stride=1, padding=0, initW = None, initB = None, dropout = None, k_sparsity=None, j_sparsity=None, l2=None, partial_sum = 1, want_train = True, spatial_dropout = None):
        layer = NeuralNetLayerConvolutionCfg(num_filters=num_filters, activation=activation, filter_width=filter_width, stride=stride, padding=padding, initW=initW, initB=initB, dropout=dropout, k_sparsity=k_sparsity, j_sparsity=j_sparsity, l2=l2, partial_sum = partial_sum)      
        layer.spatial_dropout = spatial_dropout        
        layer.want_train = want_train        
        self._layers.insert(-1,layer)
        self.index_convolution.append(self.index_layer); self.index_layer+=1
   
    def pooling(self, mode, pooling_width, stride, padding = 0, dropout = None, l2=None, spatial_dropout = None):
        assert padding == 0 #implement padding for pooling layer
        layer = NeuralNetLayerPoolingCfg(mode=mode, pooling_width=pooling_width, stride=stride, padding=padding, dropout=dropout, l2=l2)    
        layer.spatial_dropout = spatial_dropout                
        self._layers.insert(-1,layer)
        self.index_pooling.append(self.index_layer); self.index_layer+=1

    def rnorm(self, N, scale, power, blocked = 1, minDiv = 1):
        layer = NeuralNetLayerResponseNormCfg(N=N, addScale=scale, powScale=power , blocked = blocked, minDiv=minDiv)   
        self._layers.insert(-1,layer)
        self.index_rnorm.append(self.index_layer); self.index_layer+=1        
        
    def dense(self, num_filters, activation, initW = None, initB = None, dropout = None, k_sparsity = None, j_sparsity=None, l2 = None, want_train=True):
        layer = NeuralNetLayerDenseCfg(num_filters=num_filters, activation=activation, initW=initW, initB=initB, dropout=dropout, k_sparsity=k_sparsity, j_sparsity=j_sparsity, l2=l2)
        layer.want_train = want_train        
        self._layers.insert(-1,layer)
        self.index_dense.append(self.index_layer); self.index_layer+=1

    def reparam(self, mode="exp",want_reg=True):
        layer = NeuralNetLayerReparameterize(mode,want_reg)
        self._layers.insert(-1,layer)
        self.index_reparam.append(self.index_layer); self.index_layer+=1     
        
    def output_dense(self, num_filters, activation, initW = None, initB = None, dropout = None, l2 = None, want_train=True):
        layer = NeuralNetLayerDenseCfg(num_filters=num_filters, activation=activation, initW=initW, initB=initB, dropout=dropout, k_sparsity = None, j_sparsity = None, l2=l2)
        layer.want_train = want_train                
        self._layers[-1] = layer
        self.index_dense.append(self.index_layer); self.index_layer+=1

    def output_conv(self, num_filters, activation, filter_width=None, stride=1, padding=0, initW = None, initB = None, l2=None, partial_sum = 1):
        layer = NeuralNetLayerConvolutionCfg(num_filters=num_filters, activation=activation, filter_width=filter_width, stride=stride, padding=padding, initW=initW, initB=initB, dropout=None, k_sparsity=None, j_sparsity=None, l2=l2, partial_sum = partial_sum)
        self._layers[-1] = layer
        self.index_convolution.append(self.index_layer); self.index_layer+=1
        self.finalize()    

    def save_location(self,name):
        self.name = name       
        self.directory = nn.work_address()+"/save/"+name+"/"    
        if not os.path.exists(self.directory): 
            os.makedirs(self.directory)

    def params(self,arch,learning,dataset,dataset_type=None,test_only = False):
        self.arch = arch
        self.learning = learning
        self.dataset = dataset
        self.test_only = test_only
        self.dataset_type = dataset_type
    
    def cost(self,cost):
        self.cost = cost
        self.finalize()

    def finalize(self):
        for k in range(1,len(self)):
            layer_previous = self[k-1]
            layer = self[k]
            
            if k in self.index_convolution:
                try:
                    pass
                    # assert (2*abs(layer.padding) + layer_previous.shape[1] - layer.filter_width) % layer.stride == 0
                    # assert (2*abs(layer.padding) + layer_previous.shape[2] - layer.filter_width) % layer.stride == 0
                except AssertionError:
                    print("\x1b[31m\"Error in layer: \"\x1b[0m"),k
                    print [(self[i].shape) for i in range(k)]
                layer.shape = [layer.num_filters,(layer_previous.shape[1]+2*layer.padding-layer.filter_width)/layer.stride+1, (layer_previous.shape[2]+2*layer.padding-layer.filter_width)/layer.stride+1]
                layer.size = layer.shape[1] * layer.shape[2] * layer.num_filters                
                layer.filter_shape = [layer.shape[0],layer_previous.shape[0],layer.filter_width,layer.filter_width]
                layer.weights_shape = [[layer.shape[0],layer_previous.shape[0],layer.filter_width,layer.filter_width],[1,layer.num_filters]]
                layer.filter_size = layer.filter_width** 2 * layer_previous.shape[0] *layer.shape[0]
                layer.num_weights = layer.filter_size + layer.num_filters
       
            if k in self.index_dense:
                layer.weights_shape = [[layer_previous.size,layer.num_filters],[1,layer.num_filters]]         
                layer.filter_shape = layer_previous.size
                layer.filter_size = layer_previous.size
                layer.num_weights = (layer_previous.size+1)*layer.num_filters

            if k in self.index_pooling:
                try:
                    pass
                    # assert layer_previous.shape[1] % layer.stride == 0
                    # assert layer_previous.shape[2] % layer.stride == 0
                except AssertionError:
                    print("\x1b[31m\"Error in layer: \"\x1b[0m"),k
                    print [(self[i].shape) for i in range(k)]
                layer.shape = [layer_previous.shape[0],layer_previous.shape[1]/layer.stride, layer_previous.shape[2]/layer.stride]
                layer.size = layer.shape[1] * layer.shape[2] * layer.shape[0]
                layer.weights_shape = [None,None]

            if k in self.index_rnorm:
                layer.shape = layer_previous.shape
                layer.size = layer_previous.size     
                layer.weights_shape = [None,None]

            if k in self.index_reparam:
                layer.shape = layer_previous.shape/2
                layer.size = layer_previous.size/2   
                self.want_reg = self[k].want_reg                       
                layer.weights_shape = [None,None]


                
        self.k_sparsity=[(self[k].k_sparsity) for k in range(len(self))]
        self.layer_shape=[(self[k].shape) for k in range(len(self))]
        self.padding=[(self[k].padding) for k in range(len(self))]
        self.stride=[(self[k].stride) for k in range(len(self))]
        self.layer_size= [self[k].size for k in range(len(self))]
        self.activation=[(self[k].activation) for k in range(len(self))]
        # self.activation_prime=[(self[k].activation_prime) for k in range(len(self))]
        self.filter_shape=[(self[k].filter_shape) for k in range(1,len(self))]
        self.weights_shape=[[None,None]]+[(self[k].weights_shape) for k in range(1,len(self))]
        self.filter_size=[(self[k].filter_size) for k in range(1,len(self))]
        self.num_weights=[(self[k].num_weights) for k in range(1,len(self))]
        self.num_weights_sum = [sum(self.num_weights[:i+1]) for i in range(len(self)-1)]
        self.num_parameters=sum(self.num_weights)

        self.dic= defaultdict(int)
        self.dic['k_sparsity']  = self.k_sparsity
        self.dic['index convolution'] = self.index_convolution
        self.dic['index dense'] = self.index_dense
        self.dic['index pooling'] = self.index_pooling
        self.dic['activations'] = self.activation
        self.dic['layer shape'] = self.layer_shape
        self.dic['layer size'] = self.layer_size
        self.dic['filter shape'] = self.filter_shape
        self.dic['filter size'] = self.filter_size
        self.dic['number of weights'] = self.num_weights
        self.dic['number of weights sum'] = self.num_weights_sum
        self.dic['padding'] = self.padding
        self.dic['stride'] = self.stride

        # print self.weights_shape

    def info(self):
        for dic_k,dic_v in self.dic.items():
            if dic_k=="layer shape":
                print dic_k,"   :   ",dic_v        

    def __getitem__(self,i): return self._layers[i]
    def __len__(self):       return len(self._layers)
    def __iter__(self):      return self._layers.__iter__()

#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

class WeightSet:
    def __init__(self,cfg,initial_weights=None):
        self.cfg=cfg 
        self.size=len(cfg.num_weights)           
        if initial_weights == None: self.mem = nn.zeros(self.cfg.num_parameters) #it has to be initialized because of wk,bk = self.weights[k] in the self.init_weights(initial_weights)
        # if initial_weights == None: pass
        else: self.mem = initial_weights

        self.weights = [(None,None)];
        for k in range(1,len(cfg)):
            self.weights.append(self.get_weights(k))
  
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

    def full(self,i): 
        pass #to be implemented
    
    def make_tied(self,i,j):
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        
        if wk0.ndim==2:
            wk0    += wk1.T
            wk1[:]  = wk0.T
        else:
            w=wk1.as_numpy_array()
            for ch in xrange(wk1.shape[0]):
                for f in range(wk1.shape[1]):
                    x = w[ch,f,:,:]
                    x_flip = np.flipud(np.fliplr(x))
                    wk0[f,ch,:,:] += nn.garray(x_flip)
                    x_flip = np.flipud(np.fliplr(wk0[f,ch,:,:].as_numpy_array()))
                    wk1[ch,f,:,:] = x_flip
            #show_filters(w2,(4,4))
            # if type(wk1) == gp.garray: w2=gp.garray(w2)
            # return w2


       
    def make_tied_copy(self,i,j):
        wk0,bk0 = self[i]
        wk1,bk1 = self[j]
        if wk0.ndim==2:
            wk0    = wk1.T
            wk1[:] = wk0.T
        else:
            w=wk1.as_numpy_array()
            for ch in xrange(wk1.shape[0]):
                for f in range(wk1.shape[1]):
                    x = w[ch,f,:,:]
                    x_flip = np.flipud(np.fliplr(x))
                    wk0[f,ch,:,:] = nn.garray(x_flip)


    def get_weights(self,i):
        if i in self.cfg.index_pooling: return None, None
        if i in self.cfg.index_reparam: return None, None        
        if i in self.cfg.index_rnorm: return None, None
        w=self.mem[(self.cfg.num_weights_sum[i-1]-self.cfg.num_weights[i-1]):self.cfg.num_weights_sum[i-1]-self.cfg[i].num_filters]
        b=self.mem[self.cfg.num_weights_sum[i-1]-self.cfg[i].num_filters:self.cfg.num_weights_sum[i-1]]
        if i in self.cfg.index_convolution: return w.reshape(tuple(self.cfg[i].filter_shape)),b[np.newaxis,:]
        elif i in self.cfg.index_dense: return w.reshape(self.cfg[i-1].size,self.cfg[i].size),b[np.newaxis,:]
        else: raise Exception("Wrong index!")

    def __getitem__(self,i):
        return self.weights[i]




    def w1_w2(self,w1):
        if type(w1) == gp.garray: w=w1.as_numpy_array()
        else: w=w1
        w2 = np.zeros((16, self.filter_width, self.filter_width, 1))
        for i in range(16):
            x = w[0,:,:,i]
            x_flip = np.flipud(np.fliplr(x))
            w2[i,:,:,0] = x_flip
        #show_filters(w2,(4,4))
        if type(w1) == gp.garray: w2=gp.garray(w2)
        return w2



















        # if hyper:
        #     tic = time.time()
        #     for epoch in range(1,num_epochs+1):                          
        #         for l in range(num_batch):
        #             self.compute_grad(X,T,l,batch_size)
        #             # self.dweights*=-learning_rate
        #             # v*=momentum
        #             # v+=self.dweights
        #             v = momentum*v - learning_rate*self.dweights
        #             self.weights+=v #v=momentum*v-learning_rate*dw; w=w+v
        #         self.err_train[epoch-1] = self.compute_cost(X,T,1,batch_size)
        #         if report: self.err_test[epoch-1] = self.test(X_test,T_labels)  
        #     return self.err_test[epoch-1], self.err_train[epoch-1], round(time.time()-tic,2)

        
    # def test(self,X_test,T_labels):
    #     self.feedforward(X_test)
    #     if self.cfg[-1].activation==nn.softmax: 
    #         if nn.backend==nn.GnumpyBackend: return (np.argmax(self.H[-1].as_numpy_array(),axis=1)!=T_labels.as_numpy_array()).sum()
    #         return (np.argmax(self.H[-1],axis=1)!=T_labels).sum()
    # def hyper_search_layers(self, X, T, X_test, T_labels, num_epochs = 1):
    #     dataset_size = X.shape[3]
    #     batch_size = 100
    #     size = 384
    #     counter = 0
    #     for rnd1W in [0.1, 0.01, 0.001, 0.0001]:
    #         for rnd1B in [0.1, 0.01, 0.001, 0.0001, "one"]:
    #             for rnd2 in [0.1, 0.01, 0.001, 0.0001]:        
    #                 for learning_rate in [0.001, 0.0001]:
    #                     for momentum in [0, 0.3, 0.6]:
    #                         counter += 1
    #                         self.cfg[1].initW = rnd1W; self.cfg[1].initB = rnd1B;
    #                         self.cfg[3].initW = rnd2; self.cfg[3].initB = rnd2;
    #                         self.init_weights()                        
    #                         initial_weights = "layers"
    #                         result = self.train(X = X, T = T, X_test = X_test, T_labels = T_labels, momentum = float(momentum), learning_rate = float(learning_rate), batch_size = 100,
    #                         dataset_size = dataset_size, initial_weights = initial_weights, visual = False, report = True, num_epochs = num_epochs, hyper = True)
    #                         print counter,'/', size, result, 'rnd1W =', rnd1W, 'rnd1B =', rnd1B, 'rnd2 =', rnd2, 'learn =', learning_rate,  'momentum =', momentum
    #                         plt.sys.stdout.flush()

    # def hyper_search(self, X, T, X_test, T_labels, num_epochs = 1, speed = 'quick'):
    #     if speed == "quick": rnd_step, learning_rate_step, momentum_step = 1.0, 1.0, .3
    #     elif speed == "slow": rnd_step, learning_rate_step, momentum_step = .5, .5, .1
    #     else: raise Exception("Wrong!") 
    #     dataset_size = X.shape[3]
    #     batch_size = 100
    #     size = np.arange(1, 5, rnd_step).shape[0] * np.arange(1, 5, learning_rate_step).shape[0] * np.arange(0, 1, momentum_step).shape[0]
    #     counter = 0
    #     for rnd in 10**-np.arange(1, 5, rnd_step):
    #         for learning_rate in 10**-np.arange(1, 5, learning_rate_step):
    #             for momentum in np.arange(0, 1, momentum_step):
    #                 counter += 1
    #                 initial_weights = float(rnd)*nn.randn(self.cfg.num_parameters)
    #                 result = self.train(X = X, T = T, X_test = X_test, T_labels = T_labels, momentum = float(momentum), learning_rate = float(learning_rate), batch_size = 100,
    #                 dataset_size = dataset_size, initial_weights = initial_weights, visual = False, report = True, num_epochs = num_epochs, hyper = True)
    #                 print counter,'/', size, result, 'rnd =', rnd, 'learn =', learning_rate,  'momentum =', momentum

    # def hyper_search_deep(self, X, T, X_test, T_labels, num_epochs = 1):
    #     dataset_size = X.shape[3]
    #     batch_size = 100
    #     size = 384
    #     counter = 0
    #     for rnd1 in [0.1, 0.01, 0.001, 0.0001]:
    #         for rnd3 in [0.1, 0.01, 0.001, 0.0001]:
    #             for rnd5 in [0.1, 0.01, 0.001, 0.0001]:        
    #                 for learning_rate in [0.001, 0.0001]:
    #                     for momentum in [0, 0.3, 0.6]:
    #                         counter += 1
    #                         self.cfg[1].initW = rnd1; self.cfg[1].initB = rnd1;
    #                         self.cfg[3].initW = rnd3; self.cfg[3].initB = rnd3;
    #                         self.cfg[5].initW = rnd5; self.cfg[5].initB = rnd5;
    #                         self.init_weights()                        
    #                         initial_weights = "layers"
    #                         result = self.train(X = X, T = T, X_test = X_test, T_labels = T_labels, momentum = float(momentum), learning_rate = float(learning_rate), batch_size = 100,
    #                         dataset_size = dataset_size, initial_weights = initial_weights, visual = False, report = True, num_epochs = num_epochs, hyper = True)
    #                         print counter,'/', size, result, 'rnd1 =', rnd1, 'rnd3 =', rnd3, 'rnd5 =', rnd5, 'learn =', learning_rate,  'momentum =', momentum