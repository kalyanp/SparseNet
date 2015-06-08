from IPython.display import clear_output
import numpy as np
import scipy.ndimage
import gnumpy as gp
import nn_utils as nn
import pylab as plt
import time
import dataset as dataset
import convnet_utils as cn
import os
import scipy.misc
from nn_class import NeuralNetPredict,NeuralNetVisual,NeuralNetOther
import convnet_utils as cn
from weights import WeightSet



class NeuralNet(NeuralNetPredict,NeuralNetVisual,NeuralNetOther):
    def __init__(self,cfg):
        self.cfg = cfg
        self.size = len(cfg)
        self.weights = WeightSet(self.cfg.weights_shape)
        self.dweights = WeightSet(self.cfg.weights_shape)
        self.mini_batch = None
        # self.init_weights()
        self.test_mode = False
        if cfg.cost == "cross-entropy": 
            assert self.cfg[-1].activation == nn.softmax
            self.compute_cost = self.compute_cost_log
        elif cfg.cost == "euclidean":
            self.compute_cost = self.compute_cost_euclidean
            assert self.cfg[-1].activation == nn.linear           
        elif cfg.cost == "hinge":           
            self.compute_cost = self.compute_cost_hinge
            assert self.cfg[-1].activation == nn.linear 
        elif cfg.cost == "bce": 
            assert self.cfg[-1].activation == nn.sigmoid
            self.compute_cost = self.compute_cost_bce                      
        else: raise ValueError("Wrong Cost!")
        
    def finalize(self,mini_batch):
        self.mini_batch = mini_batch 
        self.H = [None] * self.size
        # self.H_ = [None] * self.size
        self.dH = [None] * self.size
        self.delta = [None] * self.size        
        self.mask_matrix = [None] * self.size

        for k in range(0,self.size):
            if self.cfg[k].type == "dense":
                self.H[k] = nn.empty((self.mini_batch,self.cfg[k].size))
                self.delta[k] = nn.empty((self.mini_batch,self.cfg[k].size))
                self.dH[k] = nn.empty((self.mini_batch,self.cfg[k].size))
            elif self.cfg[k].type == "reparam":
                self.H[k] = nn.empty((self.mini_batch,self.cfg[k].size))
                self.cfg[k].eps = nn.empty((self.mini_batch,self.cfg[k].size))                
                self.cfg[k].mu = nn.empty((self.mini_batch,self.cfg[k].size))                
                self.cfg[k].sigma = nn.empty((self.mini_batch,self.cfg[k].size))                
                self.cfg[k].log_sigma = nn.empty((self.mini_batch,self.cfg[k].size))                
            else: 
                self.H[k] = nn.empty((self.mini_batch,self.cfg[k].shape[0],self.cfg[k].shape[1],self.cfg[k].shape[2]))
                self.delta[k] = nn.empty((self.mini_batch,self.cfg[k].shape[0],self.cfg[k].shape[1],self.cfg[k].shape[2]))
                self.dH[k] = nn.empty((self.mini_batch,self.cfg[k].shape[0],self.cfg[k].shape[1],self.cfg[k].shape[2]))              
        # if self.size-1 in self.cfg.index_dense:
        self.H_cost = nn.empty(self.H[-1].shape)
        # elif self.size-1 in self.cfg.index_conv:


    def set_cost(self,cost):
        self.compute_cost = cost
        
    def feedforward(self,x,t=None):
        if self.mini_batch != x.shape[0]: self.finalize(x.shape[0])
        if t!=None: assert x.shape[0] == t.shape[0]

        if (self.cfg.want_dropout and self.cfg[0].dropout != None): 
            if self.test_mode: self.H[0]=x.copy(); nn.imul(self.H[0], 1-self.cfg[0].dropout)
            else:              
                self.H[0]=x.copy(); 
                nn.dropout(A=self.H[0],B=None,rate=self.cfg[0].dropout,outA=self.H[0],outB=None)
        else: self.H[0]=x
        
        for k in range(1,self.size):
            wk,bk = self.weights[k]
            ###############################################################################################################    
            if self.cfg[k].type=="convolution":
                self.H[k],self.dH[k] = self.cfg[k].fprop(self.H[k-1], wk, bk, self.H[k], self.dH[k])              
                

                if ((not self.test_mode) and self.cfg.want_spatial_sparsity and self.cfg[k].spatial_sparsity != None):
                    assert self.H[k].shape[2]%self.cfg[k].spatial_sparsity == 0 
                    pool_width = self.H[k].shape[2]/self.cfg[k].spatial_sparsity
                    # print self.H[k].shape[2],pool_width
                    H_=nn.MaxPool(self.H[k],subsX=pool_width,startX=0,strideX=pool_width,outputsX=self.cfg[k].spatial_sparsity)
                    # print H_.shape
                    if self.cfg[k].lifetime_sparsity and self.mini_batch>1:
                        shape = H_.shape
                        H_ = H_.reshape(self.mini_batch,-1)
                        mask = nn.k_sparsity_mask(H_,self.cfg[k].lifetime_sparsity,axis=0) 
                        nn.multiply(H_,mask,H_)
                        H_ = H_.reshape(shape)

                    self.H[k]=nn.MaxPoolUndo(self.H[k],H_,H_,subsX=pool_width,startX=0,strideX=pool_width)
                    mask = (self.H[k] > 0)
                    self.dH[k] *= mask         
            ###############################################################################################################    
            if self.cfg[k].type=="deconvolution":
                self.H[k],self.dH[k] = self.cfg[k].fprop(self.H[k-1], wk, bk, self.H[k], self.dH[k])                                   
            ###############################################################################################################               
            elif self.cfg[k].type=="pooling":
                self.H[k] = self.cfg[k].fprop(self.H[k-1])
            ###############################################################################################################    
            elif self.cfg[k].type=="rnorm":
                self.H[k] = self.cfg[k].fprop(self.H[k-1])                
            ###############################################################################################################    
            elif self.cfg[k].type=="reparam":

                if not self.test_mode:
                    self.H[k],self.dH[k],self.sigma,self.mu = self.cfg[k].fprop(self.H[k-1], self.H[k], self.dH[k])                           
                else:
                    nn.fill_randn(self.H[k])
            ###############################################################################################################    
            elif self.cfg[k].type=="dense":
                self.H[k],self.dH[k] = self.cfg[k].fprop(self.H[k-1], wk, bk, self.H[k], self.dH[k])      
                # print k

                if ((not self.test_mode) and self.cfg.want_lifetime_sparsity and self.cfg[k].lifetime_sparsity != None):
   
                    # print self.mask_matrix[:,0]            
                    if self.cfg[k].lifetime_sparsity>0:
                        self.mask_matrix = nn.k_sparsity_mask(self.H[k],self.cfg[k].lifetime_sparsity,axis=0) 
                        nn.imul(self.H[k],self.mask_matrix)
                        nn.imul(self.dH[k],self.mask_matrix)
                    else: 
                        self.mask_matrix = nn.k_sparsity_mask(self.H[k],-self.cfg[k].lifetime_sparsity,axis=1) 
                        nn.imul(self.H[k],self.mask_matrix)
                        nn.imul(self.dH[k],self.mask_matrix)       

                if self.cfg.dataset_extra == "generate" and self.cfg[k].l2_activity != None:
                    if self.test_mode:
                        self.test_rand = .5*nn.randn(self.H[k].shape)
                        self.H[k][:] = self.test_rand
                        # nn.fill_rand()
                    else: 
                        pass
                        # self.H[k] = nn.T_sort(self.H[k])
            ###############################################################################################################    

            if (self.cfg.want_dropout and self.cfg[k].dropout != None): 
                if self.test_mode: 
                    nn.imul(self.H[k],1-self.cfg[k].dropout)
                else: 
                    nn.dropout(A=self.H[k],B=self.dH[k],rate=self.cfg[k].dropout,outA=self.H[k],outB=self.dH[k]) 

        if t!=None:
            return self.compute_cost(x,t)

    def compute_cost_bce(self,x,t):
        H_temp = self.H[-1]*t+(1-self.H[-1])*(1-t)
        H_temp = nn.log(H_temp)
        out = (-1.0/self.mini_batch) * nn.sum(H_temp)

        # for k in self.cfg.index_dense:
        #     if self.cfg[k].l2_activity!=None: index = k 
        # out += .5 * (1.0/self.mini_batch) * self.cfg[index].l2_activity * ((self.H[index])**2).sum()


        # for k in self.cfg.index_dense:
        #     if self.cfg[k].l2_activity!=None: index = k 
        # out += (1.0/self.mini_batch) * self.cfg[index].l2_activity * ((nn.sum(self.H[index],0))**2).sum()


        # if self.cfg.want_reg: 
        #     temp = self.sigma**2
        #     kl = -.5 * (1.0/self.mini_batch) * nn.sum(-temp-self.mu**2+1+nn.log(temp))
        #     out += self.cfg.want_reg*kl
        return out

    def compute_cost_log(self,x,t):
        H_temp = nn.sum(self.H[-1]*t,1)
        H_temp = nn.log(H_temp)
        out = (-1.0/self.mini_batch) * nn.sum(H_temp)
        for k in range(1,len(self.cfg)):
            if self.cfg[k].l2!=None:
                wk,bk = self.weights[k]
                sqrW = nn.empty(wk.shape)
                nn.square(wk,sqrW)
                out  += self.cfg[k].l2*.5*nn.sum(sqrW)
        return out

    def compute_cost_euclidean(self,x,t):
        nn.subtract(t,self.H[-1],self.H_cost)
        nn.square(self.H_cost,self.H_cost)
        nn.imul(self.H_cost,.5/self.mini_batch)
        return self.H_cost.sum()

    def compute_cost_hinge(self,x,t):
        out = 1-(self.H[-1]*t)
        out *= (out>0) 
        return 1.0/self.mini_batch * nn.sum(out)     

    
    def compute_grad(self,x,t):

        cost = self.feedforward(x,t)

        wk,bk = self.weights[self.size-1]
        dwk,dbk = self.dweights[self.size-1]

        if self.cfg.cost == "euclidean" or self.cfg.cost == "cross-entropy" or self.cfg.cost == "bce":
            nn.subtract(self.H[-1],t,self.delta[self.size-1])

        elif self.cfg.cost == "hinge":
            self.delta[self.size-1] = -1*((self.H[-1]*t)<1)*t


        for k in range(1,self.size)[::-1]:
            #We update the weights of the k-th layer and compute delta[k-1] of the previous layer.
            if self.cfg.want_layerwise_train:           
                if not self.cfg[k].want_train: break
                if self.cfg[k].type=="pooling" and not self.cfg[k-1].want_train: break

            ###############################################################################################################    
            if self.cfg[k].type=="dense":

                wk,bk = self.weights[k]
                dwk,dbk = self.dweights[k]
                if self.cfg[k-1].type in ("dense","reparam"):
                    H_temp = self.H[k-1]
                else: 
                    H_temp = self.H[k-1].reshape(self.mini_batch,self.cfg[k-1].shape[0] * self.cfg[k-1].shape[1] * self.cfg[k-1].shape[2])        

                nn.dot_tn(H_temp,self.delta[k],out=dwk)
                nn.imul(dwk,1.0/self.mini_batch)
                nn.sum(self.delta[k],axis=0,out=dbk)
                nn.imul(dbk,1.0/self.mini_batch)   

                self.delta[k-1] = nn.dot(self.delta[k], wk.T)

                if self.cfg[k-1].type=="pooling": 
                    self.delta[k-1]  = self.delta[k-1].reshape(self.mini_batch, self.cfg[k-1].shape[0], self.cfg[k-1].shape[1], self.cfg[k-1].shape[2])
                elif (self.cfg[k-1].type == "convolution" and k!=1):
                    self.delta[k-1]  = self.delta[k-1].reshape(self.mini_batch, self.cfg[k-1].shape[0], self.cfg[k-1].shape[1], self.cfg[k-1].shape[2])
                    nn.imul(self.delta[k-1],self.dH[k-1])
                elif (self.cfg[k-1].type == "dense" and k!=1):
                    # if self.cfg[k-1].l2_activity!=None:
                    #     self.T_sort = nn.T_sort(self.H[k-1])
                    #     self.delta[k-1] += self.cfg[k-1].l2_activity * (self.H[k-1]-self.T_sort)
                        # print self.T_sort
                    ##########################
                    # if self.cfg[k-1].l2_activity!=None: 
                        # self.delta[k-1] += self.cfg[k-1].l2_activity * ((self.H[k-1]))

                    # if self.cfg[k-1].l2_activity!=None: 
                        # z = self.H[k-1]
                        # temp = z*nn.sum(z,0)-z**2
                        # print temp.shape
                        # self.delta[k-1] = temp
                        # self.delta[k-1] += self.cfg[k-1].l2_activity * ((self.H[k-1]))

                    nn.imul(self.delta[k-1],self.dH[k-1])
            ###############################################################################################################    
            if self.cfg[k].type=="reparam":
                self.cfg[k].bprop(self.delta[k],self.dH[k-1],self.delta[k-1])
            ###############################################################################################################    
            elif self.cfg[k].type=="pooling":
                self.delta[k-1]  = self.cfg[k].applyPoolingUndo(self.H[k-1],self.delta[k],self.H[k])
                self.delta[k-1] *= self.dH[k-1]
            ###############################################################################################################    
            elif self.cfg[k].type=="convolution":
                wk,bk = self.weights[k]
                dwk,dbk = self.dweights[k]    

                dwk[:] = (1.0/self.mini_batch)*self.cfg[k].applyConvOut(self.H[k-1], self.delta[k])

                delta_ = self.delta[k].reshape(self.mini_batch*self.cfg[k].shape[0] , self.cfg[k].shape[1]*self.cfg[k].shape[2]).T.reshape(self.cfg[k].shape[1]*self.cfg[k].shape[2]*self.mini_batch,self.cfg[k].shape[0])
                dbk[:] = (1.0/self.mini_batch)*delta_.sum(0)

                if k!=1: self.delta[k-1] = self.cfg[k].applyConvDown(self.delta[k], wk, self.cfg[k-1].shape[1]) #convdown is unnecessary if k==1
                if (self.cfg[k-1].type=="convolution" and k!=1): 
                    nn.imul(self.delta[k-1],self.dH[k-1])
            ###############################################################################################################    '
            elif self.cfg[k].type=="deconvolution":
                wk,bk = self.weights[k]
                dwk,dbk = self.dweights[k]    

                dwk[:] = (1.0/self.mini_batch)*self.cfg[k].applyDeConvOut(self.H[k-1], self.delta[k])

                delta_ = self.delta[k].reshape(self.mini_batch*self.cfg[k].shape[0] , self.cfg[k].shape[1]*self.cfg[k].shape[2]).T.reshape(self.cfg[k].shape[1]*self.cfg[k].shape[2]*self.mini_batch,self.cfg[k].shape[0])
                dbk[:] = (1.0/self.mini_batch)*delta_.sum(0)

                if k!=1: self.delta[k-1] = self.cfg[k].applyDeConvDown(self.delta[k], wk) #convdown is unnecessary if k==1
                if (self.cfg[k-1].type in ("convolution","deconvolution") and k!=1): 
                    nn.imul(self.delta[k-1],self.dH[k-1])
            ###############################################################################################################              
        
        #tied weights
        if self.cfg.want_tied:
            for hidden_pairs in self.cfg.tied_list: self.dweights.make_tied(*hidden_pairs)  
                
        for k in range(1,len(self.cfg)):
            if self.cfg[k].l2!=None:
                wk,bk   = self.weights[k]
                dwk,dbk = self.dweights[k]
                nn.iadd(dwk,self.cfg[k].l2*wk)


        if self.cfg.want_layerwise_train:           
            for k in range(1,len(self.cfg)):
                if not self.cfg[k].want_train:
                    dwk,dbk = self.dweights[k]
                    dwk[:]=0
                    dbk[:]=0
        
        return cost
        
##########################################################################################################

    def train(self,dp,
              momentum,
              mini_batch,
              learning_rate,learn_params=None,
              initial_weights = None,
              want_visual=False,visual_params=None,
              want_test=False,test_params=None,
              want_log=False,log_params=None,
              want_weights=False,weights_params={'interval':100,'want_last':True},
              num_epochs=10000,
              hyper=False,silent_mode = False):

        # if silent_mode: assert not(visual_params['save'] or want_log or want_weights)
        if (not silent_mode and (visual_params['save'] or want_log or want_weights)): assert os.path.exists(self.cfg.directory)
        # dp.train_range = train_range; dp.train_id = train_range[0]
        self.visual_params = visual_params
        # print "Backend: ",nn.backend
        assert self.cfg.cost != "hinge"
        # if self.cfg.cost == "hinge": 
        #     T = ((T-.5)*2).copy()
        
        self.init_weights(initial_weights,silent_mode)
        self.finalize(mini_batch)   

        v = WeightSet(self.cfg.weights_shape)        
        tic = time.time()
        
        self.err_train = []; self.err_train_epoch = []
        self.err_test = []; self.err_test_epoch = []
        # self.err_train = np.zeros((num_epochs)*(train_range[1]-train_range[0]),'float32')
        # self.err_test = np.zeros((num_epochs)*(train_range[1]-train_range[0]),'int32')


            # if self.cfg.learning == "auto": plt.figure(2)

        if want_log and not silent_mode: 
            f = open(self.cfg.directory+self.cfg.name+"_training_log.txt", 'w')
            for dic_k,dic_v in self.cfg.dic.items():
                f.write(str(dic_k)+"   :   "+str(dic_v)+"\n")
            f.write('-----------------------------------\n')
            f.close()

        id = 0
        for epoch in range(1,num_epochs+1):       
            self.epoch = epoch

            if learn_params['epoch_1']:
                if epoch >= learn_params['epoch_1'][0]: learning_rate = learn_params['epoch_1'][1]
            if learn_params['epoch_2']:
                if epoch >= learn_params['epoch_2'][0]: learning_rate = learn_params['epoch_2'][1]



            
            for _ in range(dp.train_range[0],dp.train_range[1]):
                tic = time.time()
                X,T,train_id = dp.train()   
                # w,b = self.weights[1]
                # print w[0,:3,:3,0]  

                # nn.show_images(X[:25].reshape(25,3,32,32),(5,5),unit=2)
                # print T[:5]
                # # nn.show_images(X[:25,:,:,:],(5,5),unit=1)
                # plt.draw(); plt.pause(5)  

                data_batch_waiting_time = time.time()-tic
                id = (dp.train_range[1]-dp.train_range[0])*(epoch-1)+train_id+1
                tic = time.time()

                # if learn_params['epoch_1']:
                #     if id >= learn_params['epoch_1'][0]: learning_rate = learn_params['epoch_1'][1]

                # if learn_params['epoch_2']:
                #     if id >= learn_params['epoch_2'][0]: learning_rate = learn_params['epoch_2'][1]


                # if dp.data_batch == None:
                #     if self.cfg.dataset in ('imagenet','svhn'): dp.data_batch = 3072
                #     elif self.cfg.dataset == 'frey': dp.data_batch = 1965
                #     else: dp.data_batch = 10000
                assert not(self.cfg.dataset in ('imagenet','svhn','frey')) 
                assert dp.data_batch % mini_batch == 0 
                num_batch = dp.data_batch/mini_batch


                self.cost_data_batch = 0
                for l in range(num_batch):

                    x = nn.data_convertor(X,mini_batch*l,mini_batch*(l+1))
                    t = nn.data_convertor(T,mini_batch*l,mini_batch*(l+1))
                    cost = self.compute_grad(x,t)
                    self.cost_data_batch += cost

                    self.dweights*=-learning_rate
                    v*=momentum
                    v+=self.dweights #v = momentum*v - learning_rate*self.dweights
                    self.weights+=v #v=m*v-l*dw; w=w+v

                # print self.sigma[0]
                # print self.mu[0]


                self.err_train.append(self.cost_data_batch/num_batch); self.err_train_epoch.append(epoch+1.0*train_id/(dp.train_range[1]-dp.train_range[0])) #dataset size or batch size greater than the actual still works!!!!!!
                train_str = "Epoch:"+str(epoch)+"."+str(train_id)+"   Train Error:"+str(self.err_train[-1])+"   Time:"+str(round(time.time()-tic,2))+"   Learning Rate:"+str(learning_rate)+"   Databatch Waiting:"+str(data_batch_waiting_time)
                if (not silent_mode): print train_str; plt.sys.stdout.flush()    

                if want_log and not silent_mode: 
                    f = open(self.cfg.directory+self.cfg.name+"_training_log.txt", 'a')
                    f.write(train_str+"\n")
                    f.close()

                if want_weights and id%weights_params['interval']==0 and not silent_mode:
                    self.save(self.cfg.directory+self.cfg.name+"_last")
                    print "----Saved Weights."
                    # self.save(self.cfg.directory+self.cfg.name+"_epoch_"+str(self.epoch)+"_id_"+str(id))

                if want_visual and id%visual_params['interval']==0 and not silent_mode: self.visualize(dp,visual_params)


                if want_test and id%test_params['interval']==0 and not silent_mode:
                    if self.cfg.dataset=="imagenet":
                        test_rate,test_rate_5,logprob_test = self.test(dp)
                    else:
                        test_rate,logprob_test = self.test(dp)
                    if (self.err_test==[] and want_weights) or (self.err_test and test_rate < min(self.err_test) and want_weights):
                        # print "Minimum achived at epoch", epoch
                        self.save(self.cfg.directory+self.cfg.name+"_best")
                    self.err_test.append(test_rate); self.err_test_epoch.append(epoch)             
                    
                    if self.cfg.dataset=="imagenet":
                        test_str =  "---------------\nTop-1 Error:"+str(self.err_test[-1])+"   Top-5 Error:"+str(test_rate_5)+"   Logprob Test:"+str(logprob_test)+"   ---Best:"+str(min(self.err_test))+"   ---Name:"+self.cfg.name+"\n---------------"
                    else:
                        test_str =  "---------------\nTest Error:"+str(self.err_test[-1])+"/"+str(dp.N_test)+"="+ \
                        str(1.0*self.err_test[-1]/dp.N_test)+"   Logprob Test:"+str(logprob_test)+"   ---Best:"+str(min(self.err_test))+ \
                        "/"+str(dp.N_test)+"="+str(1.0*min(self.err_test)/dp.N_test)+"   ---Name:"+self.cfg.name+"\n---------------"

                    print test_str; plt.sys.stdout.flush();
                    if want_log: 
                        f = open(self.cfg.directory+self.cfg.name+"_training_log.txt", 'a')
                        f.write(test_str+"\n")
                        f.close()
                # if not silent_mode and epoch!=num_epochs and want_test and id%test_params['interval']==test_params['interval']-1: clear_output()

            # if not silent_mode and epoch!=num_epochs and visual and self.epoch%visual_params['interval']==visual_params['interval']-1: clear_output()

        # if want_test or silent_mode : return self.test(dp)        
        return self.test(dp)
        # if want_visual and self.epoch==num_epochs and not visual_params['save']: plt.show()

 

    
    def test(self,dp):
        # print self.cfg.dataset

        if self.cfg.learning == "auto" and self.cfg.arch == "conv":
            if self.cfg.dataset == "mnist":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                H_train = np.zeros((60000,self.H[index_sparsity].shape[1], 10, 10))
                for i in range(300):
                    # if i%100==0: print i
                    self.feedforward(nn.garray(dp.X[i*200:(i+1)*200,:,:,:]))
                    H_train[i*200:(i+1)*200,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=3,outputsX=10).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 10, 10))
                for i in range(50):
                    # if i%50==0: print i
                    self.feedforward(nn.garray(dp.X_test[i*200:(i+1)*200,:,:,:]))
                    H_test[i*200:(i+1)*200,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=3,outputsX=10).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],10,10])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='mnist')
                LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,_,T_labels=dataset.MNIST.load()

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,6],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.1,learn_params={'epoch_1':[10,.01],'epoch_2':[20,.001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})                      

            elif self.cfg.dataset == "mnist-second":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                ############################# 
                H_train = np.zeros((60000,self.H[index_sparsity].shape[1], 5, 5))
                for i in range(600*4):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*25,(i+1)*25))
                    H_train[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=5).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 5, 5))
                for i in range(100*4):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*25,(i+1)*25))
                    H_test[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=5).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],5,5])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,_,T_labels=dataset.MNIST.load()

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,6],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 40,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.1,learn_params={'epoch_1':[20,.01],'epoch_2':[40,.001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})  

            if self.cfg.dataset == "mnist-semi":
                N = self.cfg.dataset_extra
                X,T,X_test,T_test,T_train_labels,T_labels = dataset.MNIST.semi(N=N,want_dense = False)
                dp = nn.dp_ram(X=X,T=T,X_test=X_test,T_test=T_test,T_train_labels=None,T_labels=None,
                                     train_range = [0,1],
                                     test_range = [0,10000/N],
                                     data_batch = N)

                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                H_train = np.zeros((N,self.H[index_sparsity].shape[1], 5, 5))
                for i in range(N/100):
                    # if i%100==0: print i
                    self.feedforward(nn.garray(dp.X[i*100:(i+1)*100,:,:,:]))
                    H_train[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=11,startX=0,strideX=6,outputsX=5).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 5, 5))
                for i in range(50):
                    # if i%50==0: print i
                    self.feedforward(nn.garray(dp.X_test[i*200:(i+1)*200,:,:,:]))
                    H_test[i*200:(i+1)*200,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=11,startX=0,strideX=6,outputsX=5).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],5,5])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='mnist')
                LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                dp = nn.dp_ram(X=H_train,T=dp.T,X_test=H_test,T_test=dp.T_test,T_train_labels=None,T_labels=None,
                                     train_range = [0,1],
                                     test_range = [0,10000/N],
                                     data_batch = N)

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch = 100,
                         num_epochs = 5000,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.1,learn_params={'epoch_1':[],'epoch_2':[]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':50},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})      

            elif self.cfg.dataset == "svhn-ram":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k   
                # print index_sparsity        
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((70000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(700*4):
                    # if i%100==0: print i
                    self.feedforward(nn.data_convertor(dp.X,i*25,(i+1)*25))
                    H_train[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=6,startX=0,strideX=4,outputsX=8).as_numpy_array()

                H_test = np.zeros((20000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(200*4):
                    # if i%25==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*25,(i+1)*25))
                    H_test[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=6,startX=0,strideX=4,outputsX=8).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],8,8])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='svhn-ram')
                # LR_cfg.save_location(__file__[:-3])
                # LR_cfg.info()
                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,_,T_labels = dataset.SVHN.load()

                dp_ = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,7],
                                          test_range = [0,1])


                return LR.train(dp_,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 25,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.001,learn_params={'epoch_1':[10,.0001],'epoch_2':[20,.00001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})     

            elif self.cfg.dataset == "svhn-second":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k   
                print "index_sparsity",index_sparsity        

                self.test_mode = True          
                outputsX = 4
                #############################                
                H_train = np.zeros((70000,self.H[index_sparsity].shape[1], outputsX, outputsX))
                for i in range(700):
                    if i%100==0: print i
                    self.feedforward(nn.data_convertor(dp.X,i*100,(i+1)*100))
                    H_train[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=outputsX).as_numpy_array()
                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], outputsX, outputsX))
                for i in range(100):
                    if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*100,(i+1)*100))
                    H_test[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=outputsX).as_numpy_array()
                #############################
                self.test_mode = False
                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],outputsX,outputsX])
                # LR_cfg.dense(num_filters=1000,activation=nn.relu,spatial_sparsity=None,dropout =None)
                ##################################################################################l222222

                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='svhn-ram')
                # LR_cfg.save_location(__file__[:-3])
                # LR_cfg.info()
                LR=NeuralNet(LR_cfg)
                _,T,_,T_test,T_train_labels,T_labels = dataset.SVHN.load()

                dp_ = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,7],
                                          test_range = [0,1])

                return LR.train(dp_,
                         silent_mode = False,
                         mini_batch=100,
                         num_epochs = 40,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=1e-2,learn_params={'epoch_1':[10,1e-3],'epoch_2':[20,1e-4]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})  

            elif self.cfg.dataset == "cifar10":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(1000):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*50,(i+1)*50))
                    H_train[i*50:(i+1)*50,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=4,outputsX=8).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(200):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*50,(i+1)*50))
                    H_test[i*50:(i+1)*50,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=4,outputsX=8).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],8,8])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.01,learn_params={'epoch_1':[10,.001],'epoch_2':[20,.0001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})     

            elif self.cfg.dataset == "cifar10-stride":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(1000):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*50,(i+1)*50))
                    H_train[i*50:(i+1)*50,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=8).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(200):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*50,(i+1)*50))
                    H_test[i*50:(i+1)*50,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=8).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],8,8])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.01,learn_params={'epoch_1':[10,.001],'epoch_2':[20,.0001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})    

            elif self.cfg.dataset == "tiny":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(500):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X_cifar,i*100,(i+1)*100))
                    H_train[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=4,outputsX=8).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 8, 8))
                for i in range(100):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test_cifar,i*100,(i+1)*100))
                    H_test[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=5,startX=0,strideX=4,outputsX=8).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],8,8])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                LR_dp = nn.dp_ram(X=H_train,T=dp.T,X_test=H_test,T_test=dp.T_test,T_train_labels=None,T_labels=dp.T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(LR_dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.01,learn_params={'epoch_1':[10,.001],'epoch_2':[20,.0001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})                                                   

            
            elif self.cfg.dataset == "cifar10-second":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                ############################# 
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 4, 4))
                for i in range(500*4):
                    if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*25,(i+1)*25))
                    H_train[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=4).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 4, 4))
                for i in range(100*4):
                    if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*25,(i+1)*25))
                    H_test[i*25:(i+1)*25,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=4).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],4,4])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = False,
                         mini_batch=100,
                         num_epochs = 40,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.01,learn_params={'epoch_1':[10,.001],'epoch_2':[20,.0001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=True,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})  

            elif self.cfg.dataset == "cifar10-third":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 4, 4))
                for i in range(500):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*100,(i+1)*100))
                    H_train[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=4).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 4, 4))
                for i in range(100):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*100,(i+1)*100))
                    H_test[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=4).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],4,4])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.001,learn_params={'epoch_1':[10,.0001],'epoch_2':[20,.00001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})                  

            elif self.cfg.dataset == "cifar10-fourth":
                for k in range(0,self.size):
                    if self.cfg[k].type == "convolution" and self.cfg[k].spatial_sparsity: index_sparsity = k           
                
                self.test_mode = True            
                #############################
                
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1], 2, 2))
                for i in range(500):
                    # if i%100==0: print i
                    # print dp.X[i*200:(i+1)*200,:,:,:].shape
                    self.feedforward(nn.data_convertor(dp.X,i*100,(i+1)*100))
                    H_train[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=2).as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1], 2, 2))
                for i in range(100):
                    # if i%50==0: print i
                    self.feedforward(nn.data_convertor(dp.X_test,i*100,(i+1)*100))
                    H_test[i*100:(i+1)*100,:,:,:] = nn.MaxPool(self.H[index_sparsity],subsX=3,startX=0,strideX=2,outputsX=2).as_numpy_array()
                #############################
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[self.H[index_sparsity].shape[1],2,2])
                ##################################################################################l222222
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='cifar10')
                # LR_cfg.save_location(__file__[:-3])

                LR=NeuralNet(LR_cfg)

                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)

                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.001,learn_params={'epoch_1':[10,.0001],'epoch_2':[20,.00001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})                    

            elif self.cfg.dataset == "cifar10-fourth-fc":
                # print "hiii"
                for k in range(0,self.size):
                    if self.cfg[k].type == "dense" and self.cfg[k].spatial_sparsity: index_sparsity = k 
            
                self.test_mode = True
                #############################
                H_train = np.zeros((50000,self.H[index_sparsity].shape[1]))
                for i in range(250):
                    # if i%100==0: print i
                    self.feedforward(nn.garray(dp.X[i*200:(i+1)*200]))
                    H_train[i*200:(i+1)*200] = self.H[index_sparsity].as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1]))
                for i in range(50):
                    # if i%50==0: print i
                    self.feedforward(nn.garray(dp.X_test[i*200:(i+1)*200]))
                    H_test[i*200:(i+1)*200] = self.H[index_sparsity].as_numpy_array()
                #############################
                # H_train_mean = H_train.mean(axis=1)
                # H_train_std = np.sqrt(H_train.var(axis=0)+10)
                # H_train = H_train - H_train.mean(axis=1)[:, np.newaxis]
                # H_test = H_test - H_test.mean(axis=1)[:, np.newaxis]
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_dense(shape=self.H[index_sparsity].shape[1])
                #cfg.dense(num_filters=1000,activation=nn.relu,initW=.01,initB=.01)
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='dense',learning='disc',dataset='cifar')
                LR=NeuralNet(LR_cfg)

                # print H_train.shape
                # w,_ = LR.weights[1]
                # print w.shape


                _,T,_,T_test,T_train_labels,T_labels = dataset.CIFAR10.load(want_mean=False)


                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,5],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 50,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.001,learn_params={'epoch_1':[15,.0001],'epoch_2':[50,.00001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})

        
        if self.cfg.learning == "auto" and self.cfg.arch == "dense":

            if self.cfg.dataset=="mnist":                
                for k in range(0,self.size):
                    if self.cfg[k].type == "dense" and self.cfg[k].lifetime_sparsity: index_sparsity = k 
            
                
                self.test_mode = True
                #############################
                H_train = np.zeros((60000,self.H[index_sparsity].shape[1]))
                for i in range(300):
                    # if i%100==0: print i
                    self.feedforward(nn.garray(dp.X[i*200:(i+1)*200]))
                    H_train[i*200:(i+1)*200] = self.H[index_sparsity].as_numpy_array()

                H_test = np.zeros((10000,self.H[index_sparsity].shape[1]))
                for i in range(50):
                    # if i%50==0: print i
                    self.feedforward(nn.garray(dp.X_test[i*200:(i+1)*200]))
                    H_test[i*200:(i+1)*200] = self.H[index_sparsity].as_numpy_array()
                #############################
                # H_train_mean = H_train.mean(axis=1)
                # H_train_std = np.sqrt(H_train.var(axis=0)+10)
                # H_train = H_train - H_train.mean(axis=1)[:, np.newaxis]
                # H_test = H_test - H_test.mean(axis=1)[:, np.newaxis]
                self.test_mode = False

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_dense(shape=self.H[index_sparsity].shape[1])
                #cfg.dense(num_filters=1000,activation=nn.relu,initW=.01,initB=.01)
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='dense',learning='disc',dataset='mnist')
                LR=NeuralNet(LR_cfg)
                
                _,T,_,T_test,_,T_labels=dataset.MNIST.load("numpy",want_dense=False)


                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                          train_range = [0,6],
                                          test_range = [0,1])

                return LR.train(dp,
                         silent_mode = True,
                         mini_batch=100,
                         num_epochs = 30,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=.1,learn_params={'epoch_1':[15,.01],'epoch_2':[50,.001]},
                         want_visual=False,visual_params={'interval':3,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100})



            elif self.cfg.dataset=="cifar10-patch":  
                # pass
                f=np.load(nn.work_address()+"/Dataset/CIFAR10/cifar_bias_.1_new.npz")
                X=f['X'];T=f['T'];X_test=f['X_test'];T_test=f['T_test'];T_train_labels=f['T_train_labels'];T_labels=f['T_labels']
                X=nn.garray(X);T=nn.garray(T);X_test=nn.garray(X_test);T_test=nn.garray(T_test);T_train_labels=nn.garray(T_train_labels);T_labels=nn.garray(T_labels)

                for k in range(0,self.size):
                    if self.cfg[k].type == "dense" and self.cfg[k].lifetime_sparsity: index_sparsity = k 

                num_filters = self.cfg[index_sparsity].shape
                size = int(np.sqrt(self.cfg[0].shape/3))

                w1,b1 = self.weights[1]
                # print wk.shape, bk.shape
                w = w1.T.reshape(num_filters,3,size,size)
                b = b1.reshape(1,-1,1,1).as_numpy_array()
                # print w.shape,b.shape
                #############################
                H_train = np.zeros((50000,num_filters, 8, 8))
                H_test = np.zeros((10000,num_filters, 8, 8))


                for i in range(50):
                    # if i%50==0: print i
                    x = X[i*100:(i+1)*100,:,:,:]
                    H = nn.ConvUp(x, w, moduleStride = 1, paddingStart = int(size-1)/2)
                    H = nn.relu(H)    
                    H = nn.MaxPool(H,subsX=6,startX=0,strideX=4,outputsX=8).as_numpy_array()
                    H_train[i*100:(i+1)*100,:,:,:] = H + b

                for i in range(100):
                    # if i%50==0: print i
                    x_test = X_test[i*10:(i+1)*10,:,:,:]
                    H = nn.ConvUp(x_test, w, moduleStride = 1, paddingStart = int(size-1)/2)
                    H = nn.relu(H)      
                    H = nn.MaxPool(H,subsX=6,startX=0,strideX=4,outputsX=8).as_numpy_array() 
                    H_test[i*10:(i+1)*10,:,:,:] = H + b


                dp = nn.dp_ram(X=H_train,T=T,X_test=H_test,T_test=T_test,T_train_labels=None,T_labels=T_labels,
                                     train_range = [0,5],
                                     test_range = [0,1],
                                     mini_batch = 100)

                LR_cfg = cn.NeuralNetCfg()
                LR_cfg.input_conv(shape=[num_filters,8,8])
                LR_cfg.output_dense(num_filters=10,activation=nn.softmax)
                LR_cfg.cost("cross-entropy")
                LR_cfg.params(arch='conv',learning='disc',dataset='')
                # LR_cfg.save_location(__file__[:-3])
                LR=NeuralNet(LR_cfg)
                return LR.train(dp,
                         silent_mode = True,                    
                         mini_batch=100,
                         num_epochs = 40,
                         initial_weights=.01,
                         momentum=.9,
                         learning_rate=1e-4,learn_params={'epoch_1':[20,1e-5],'epoch_2':[]},
                         want_visual=False,visual_params={'interval':1,'save':''},
                         want_test=False,test_params={'interval':10},
                         want_log=False,log_params={'interval':1},
                         want_weights=False,weights_params={'interval':100,'want_last':True})                         


        elif self.cfg.learning == "disc":

            if self.cfg.dataset=="svhn":
                logprob_test_total = 0.0                
                num_errors_total = 0.0    
                test_size_total = 0.0

                for _ in range(dp.test_range[0],dp.test_range[1]):

                    X_test,T_test,T_labels,test_id = dp.test()    

                    test_size =  X_test.shape[0]
                    test_size_total += test_size

                    assert test_size % self.mini_batch == 0
                    # assert nn.backend == nn.GnumpyBackend
                    num_batch = int(test_size/self.mini_batch)
                    logprob_test = 0                
                    num_errors = 0.0

                    for l in range(num_batch):
                        x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
                        t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))
                        t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
                        
                        self.test_mode = True
                        logprob_test += self.feedforward(x,t)
                        self.test_mode = False

                        H_cpu = self.H[-1].as_numpy_array()
                        t_cpu = t.as_numpy_array()
                        mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,1,1)

                        num_errors += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                    
                    num_errors_total += num_errors
                    logprob_test_total += logprob_test

                    # print test_id,num_errors/test_size,logprob_test/num_batch 
                # print test_size_total
                # print num_errors_total
                return num_errors_total/test_size_total,logprob_test_total/num_batch/(dp.test_range[1]-dp.test_range[0])

            # if self.cfg.dataset=="svhn":
            #     X_test,T_test,T_labels,_ = dp.test()
            #     # print 'a',T_test.shape


            #     test_size =  X_test.shape[0]
            #     assert test_size % self.mini_batch == 0
            #     # assert nn.backend == nn.GnumpyBackend
            #     num_batch = int(test_size/self.mini_batch)
            #     logprob_test = 0                
            #     num_errors = 0.0

            #     for l in range(num_batch):
            #         x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
            #         t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))

            #         t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
                    
            #         self.test_mode = True
            #         logprob_test += self.feedforward(x,t)
            #         self.test_mode = False

            #         H_cpu = self.H[-1].as_numpy_array()
            #         t_cpu = t.as_numpy_array()
            #         mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,1,1)
            #         num_errors += self.mini_batch - np.logical_and(mask,t_cpu).sum()
            #     return num_errors/test_size,logprob_test/num_batch 

            elif self.cfg.dataset=="imagenet" and not self.cfg.test_only:
                X_test,T_test,T_labels,_ = dp.test()

                test_size =  X_test.shape[0]
                assert test_size % self.mini_batch == 0
                # assert nn.backend == nn.GnumpyBackend
                num_batch = int(test_size/self.mini_batch)
                logprob_test = 0                
                num_errors = 0.0
                num_errors_5 = 0.0                

                for l in range(num_batch):
                    x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
                    t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))
                    t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
                    
                    self.test_mode = True
                    logprob_test += self.feedforward(x,t)
                    self.test_mode = False

                    H_cpu = self.H[-1].as_numpy_array()
                    t_cpu = t.as_numpy_array()
                    mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,5,1)
                    num_errors_5 += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                    mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,1,1)
                    num_errors += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                return num_errors/test_size,num_errors_5/test_size,logprob_test/num_batch                

            elif self.cfg.dataset=="imagenet" and self.cfg.test_only:
                for _ in range(dp.test_range[0],dp.test_range[1]):

                    X_test,T_test,T_labels,test_id = dp.test()    

                    test_size =  X_test.shape[0]
                    assert test_size % self.mini_batch == 0
                    # assert nn.backend == nn.GnumpyBackend
                    num_batch = int(test_size/self.mini_batch)
                    logprob_test = 0                
                    num_errors = 0.0
                    num_errors_5 = 0.0  

                    for l in range(num_batch):
                        x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
                        t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))
                        t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
                        
                        self.test_mode = True
                        logprob_test += self.feedforward(x,t)
                        self.test_mode = False

                        H_cpu = self.H[-1].as_numpy_array()
                        t_cpu = t.as_numpy_array()
                        mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,5,1)
                        num_errors_5 += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                        mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,1,1)
                        num_errors += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                    print test_id,num_errors/test_size,num_errors_5/test_size,logprob_test/num_batch 
            
            else:
                logprob_test = 0                
                num_errors = 0                
                
                for _ in range(dp.test_range[0],dp.test_range[1]):

                    X_test,T_test,test_id = dp.test()    

                    test_size =  X_test.shape[0]
                    assert test_size % self.mini_batch == 0
                    # assert nn.backend == nn.GnumpyBackend
                    num_batch = int(test_size/self.mini_batch)


                    for l in range(num_batch):
                        x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
                        t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))
                        # t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
                        
                        self.test_mode = True
                        logprob_test += self.feedforward(x,t)
                        self.test_mode = False

                        H_cpu = nn.array(self.H[-1])
                        t_cpu = nn.array(t)
                        # mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,5,1)
                        # num_errors_5 += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                        mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,1,1)
                        num_errors += self.mini_batch - np.logical_and(mask,t_cpu).sum()
                return num_errors,logprob_test/num_batch/(dp.test_range[1]-dp.test_range[0])
            # else:
            #     X_test,T_test,T_labels,_ = dp.test()
            #     test_size =  X_test.shape[0]
            #     assert test_size % self.mini_batch == 0
            #     # assert nn.backend == nn.GnumpyBackend
            #     num_batch = int(test_size/self.mini_batch)
            #     logprob_test = 0     

            #     num_errors = 0.0                
            #     for l in range(num_batch):
            #         x = nn.data_convertor(X_test,self.mini_batch*l,self.mini_batch*(l+1))
            #         t = nn.data_convertor(T_test,self.mini_batch*l,self.mini_batch*(l+1))
            #         t_labels = nn.data_convertor(T_labels,self.mini_batch*l,self.mini_batch*(l+1))         
            #         # x = X_test[:,:,:,batch_test_size*l:batch_test_size*(l+1)]
            #         # t = T_labels[batch_test_size*l:batch_test_size*(l+1)]
            #         self.test_mode = True
            #         logprob_test += self.feedforward(x,t)
            #         self.test_mode = False
            #         # print np.argmax(self.H[-1].as_numpy_array(),axis=1)
            #         # print t.as_numpy_array()

            #         if nn.backend==nn.GnumpyBackend: num_errors += (np.argmax(self.H[-1].as_numpy_array(),axis=1) != t_labels.as_numpy_array()).sum()
            #         if nn.backend==nn.NumpyBackend: 
            #             num_errors += np.array((np.argmax(self.H[-1],axis=1) != t_labels)).sum()                
            #     return int(num_errors),logprob_test/num_batch
               
