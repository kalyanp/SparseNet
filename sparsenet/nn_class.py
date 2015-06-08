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
import convnet_utils as cn
from pylab import figure, show, rand
from matplotlib.patches import Ellipse

class NeuralNetVisual:   
    def visualize(self,dp,visual_params):

        w1,b1 = self.weights[1]
        w2,b2 = self.weights[-1]

        if visual_params['save']:
            if self.cfg.learning == "disc":
                num_filters = w1.shape[0]
                nn.show_images(w1,(4,num_filters/4))
            if self.cfg.arch == "dense" and self.cfg.learning == "auto":
                # plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')    
                size=(10,20)        
                if self.cfg.dataset == "mnist":
                    # print size, w2.shape
                    nn.show_images(w2[:size[0]*size[1]],(size[0],size[1]),unit=1,scale=2)            
                elif self.cfg.dataset in ("cifar10","svhn-ram"): 
                    nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],3,32,32),(size[0],size[1]),unit=1)
                elif self.cfg.dataset == "faces": 
                    nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],1,32,32),(size[0],size[1]),unit=1)                       
                elif self.cfg.dataset == "faces48": 
                    nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],1,48,48),(size[0],size[1]),unit=1)   
                elif self.cfg.dataset == "frey": 
                    nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],1,20,20),(size[0],size[1]),unit=1)                                                        
                elif self.cfg.dataset == "cifar10-patch": #CIFAR10 dense patches
                    size = int(np.sqrt(self.cfg[0].shape/3))
                    nn.show_images(w2[:256].reshape(256,3,size,size),(16,16),unit=1,scale=2)                                      
                else: #CIFAR10 dense patches
                    size = int(np.sqrt(self.cfg[0].shape))
                    # print size
                    nn.show_images(w2[:200].reshape(200,1,size,size),(10,20),unit=1,scale=2)                                      
                    # nn.show_images(w2[:64].reshape(64,1,size,size),(8,8),unit=1,scale=2)                                      

            if self.cfg.learning == "auto" and self.cfg.arch == "conv":            
           
                # print w2.as_numpy_array()[:num_filters,:,:,:].shape
                num_filters = w2.shape[0]   
                # print w2.shape             
                nn.show_images(w2.as_numpy_array()[:num_filters,:,:,:],(4,num_filters/4),unit=1,scale=2)

                # plt.subplot(212)     
                # num_filters = w1.shape[0]                           
                # nn.show_images(w1.as_numpy_array()[:num_filters,:,:,:],(4,num_filters/4),unit=1)

                # print w1.shape
                # nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:num_filters,:,:,:],(4,num_filters/4),unit=1)
                # plt.show()
            plt.savefig(self.cfg.directory+self.cfg.name+".png",format="png")
            plt.close()
        else:
            if not nn.is_interactive(): 
                if self.cfg.learning == "auto" and not(self.cfg.dataset in ("cifar10-second","svhn-second","mnist-second")):
                    plt.figure(num=1, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')
                else: 
                    plt.figure(num=1, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')            
        # X = dp.X_id(0)
        # x = nn.data_convertor(X,0,1)
            w1,b1=self.weights[1]
            w2,b2=self.weights[-1]

            if self.cfg.arch == "dense" and self.cfg.learning == "disc": #dense

                if nn.is_interactive(): plt.figure(num=1, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')    
                plt.figure(1)    
                plt.subplot(131); self.plot_train()
                plt.subplot(132); self.plot_test()
                plt.subplot(133)        
                
                if self.cfg.dataset == "mnist":
                    if w1.shape[1]>25: nn.show_images(w1[:,:25].T,(5,5)) #MNIST dense
                    else: nn.show_images(w1[:,:].T,(5,w1.shape[1]/5))  #MNIST softmax          
                elif self.cfg.dataset in ("cifar10","svhn-ram"): 
                    # print w1.shape
                    if w1.shape[1]>25: nn.show_images(w1.T[:25,:].reshape(25,3,32,32),(5,5)) #CIFAR10 dense
                    else: nn.show_images(w1[:,:].reshape(3,32,32,10),(5,2))  #CIFAR10 softmax
                elif self.cfg.dataset in ("svhn-torch"): 
                    # print w1.shape
                    if w1.shape[1]>25: nn.show_images(w1.T[:25,:].reshape(25,3,32,32),(5,5),yuv=True) #CIFAR10 dense
                    else: nn.show_images(w1[:,:].reshape(3,32,32,10),(5,2),yuv=True)  #CIFAR10 softmax                    
                elif self.cfg.dataset == "cifar10-patches": #CIFAR10 dense patches
                    if u==None: nn.show_images(w1[:,:25].reshape(3,8,8,25),(5,5)) 
                    else: nn.show_images(whiten_undo(w1[:,:25].T.as_numpy_array(),u,s).T.reshape(3,8,8,25),(5,5),unit=True)
                elif self.cfg.dataset == "mnist-patches": #MNIST dense patches
                    nn.show_images(w1[:,:25].T.as_numpy_array().T.reshape(1,8,8,16),(4,4),unit=True)
                else: 
                    channel = self.H[0].shape[1]
                    size = self.H[0].shape[2]
                    if w1.shape[1]>25: nn.show_images(w1.T[:25,:].reshape(25,channel,size,size),(5,5)) 
                    else: nn.show_images(w1[:,:].reshape(10,channel,size,size),(5,2))   

            if self.cfg.arch == "dense" and self.cfg.learning == "auto" and not self.cfg.dataset_extra: #dense
                if nn.is_interactive(): plt.figure(num=1, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')    
                plt.figure(1)    
                plt.subplot2grid((2,3), (0,0), colspan=1); self.plot_train()    
                
                plt.subplot2grid((2,3), (0,1), colspan=2)       

                if self.cfg.dataset == "mnist": 
                    nn.show_images(w2[:50],(5,10))         
                    # nn.show_images(w2[:25],(5,5))         
                elif self.cfg.dataset in ("cifar10","svhn-ram"): 
                    # print w2.shape
                    nn.show_images(w2[:50].reshape(50,3,32,32),(5,10))
                elif self.cfg.dataset == "svhn-torch": #CIFAR10 dense patches
                    nn.show_images(w2[:50].reshape(50,3,32,32),(5,10),yuv=True)                    
                elif self.cfg.dataset == "cifar10-patch": #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]/3))
                    # print size
                    nn.show_images(w2[:50].reshape(50,3,size,size),(5,10))                              
                    # if u==None: nn.show_images(w1[:,:25].reshape(3,8,8,25),(5,5)) 
                    # else: nn.show_images(whiten_undo(w1[:,:25].T.as_numpy_array(),u,s).T.reshape(3,8,8,25),(5,5),unit=True)
                elif self.cfg.dataset == "mnist-patches": #MNIST dense patches
                    nn.show_images(w1[:,:25].T.as_numpy_array().T.reshape(1,8,8,16),(4,4),unit=True)
                else: #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]))
                    # print w2[:50].shape,size
                    nn.show_images(w2[:50].reshape(50,1,size,size),(5,10))                     

                plt.subplot2grid((2,3), (1,0), colspan=1); 
                if self.cfg.dataset in ("natural","mnist"): 
                    nn.show_images(w1[:,:25].T,(5,5))  

                    # w1,b1 = self.weights[1] 
                    # w2,b2 = self.weights[2] 
                    # print w1[:5,:5]
                    # print w2[:5,:5].T
                    # print "------"



                plt.subplot2grid((2,3), (1,1), colspan=1); 
                if self.cfg.dataset == "mnist": 
                    nn.show_images(self.H[0][0].reshape(1,1,28,28),(1,1))        
                elif self.cfg.dataset in ("cifar10","svhn-ram"): 
                    nn.show_images(self.H[0][0].reshape(1,3,32,32),(1,1))
                elif self.cfg.dataset == "svhn-torch": 
                    nn.show_images(self.H[0][0].reshape(1,3,32,32),(1,1),yuv=True)                    
                elif self.cfg.dataset == "cifar10-patch": #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]/3))            
                    nn.show_images(self.H[0][0].reshape(1,3,size,size),(1,1))   
                else: #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]))            
                    nn.show_images(self.H[0][0].reshape(1,1,size,size),(1,1))                               

                plt.subplot2grid((2,3), (1,2), colspan=1); 
                if self.cfg.dataset == "mnist": 
                    nn.show_images(self.H[-1][0].reshape(1,1,28,28),(1,1))        
                elif self.cfg.dataset in ("cifar10","svhn-ram"): 
                    nn.show_images(self.H[-1][0].reshape(1,3,32,32),(1,1))
                elif self.cfg.dataset == "svhn-torch": 
                    nn.show_images(self.H[-1][0].reshape(1,3,32,32),(1,1),yuv=True)
                elif self.cfg.dataset == "cifar10-patch": #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]/3))            
                    nn.show_images(self.H[-1][0].reshape(1,3,size,size),(1,1))                   
                else: #CIFAR10 dense patches
                    size = int(np.sqrt(self.H[0].shape[1]))            
                    nn.show_images(self.H[-1][0].reshape(1,1,size,size),(1,1))  







            if self.cfg.arch == "conv" and self.cfg.learning == "disc":

                if nn.is_interactive(): plt.figure(num=1, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')    
                plt.figure(1)    
                plt.subplot(131); self.plot_train()
                plt.subplot(132); self.plot_test()
                plt.subplot(133)        
                
                nn.show_images(w1[:16,:,:,:],(4,4))

            if self.cfg.arch == "conv" and self.cfg.learning == "auto":
                if nn.is_interactive(): plt.figure(num=1, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')    
                plt.figure(1)             
                # w2,b2 = self.weights[-1]

                # x=X[:,:,:,:1]
                # self.feedforward(x)
                if self.cfg.dataset in ("cifar10-second","svhn-second","mnist-second"): #CIFAR10
                    plt.subplot(131);    
                    # print self.H[0].shape,self.H[-1].shape,self.H[-1].max()
                    nn.show_images(np.swapaxes(self.H[0][:1,:16,:,:].as_numpy_array(),0,1),(4,4),bg="white")  
                    plt.subplot(132); 
                    nn.show_images(np.swapaxes(self.H[-1][:1,:16,:,:].as_numpy_array(),0,1),(4,4),bg="white")  
                    plt.subplot(133); 
                    nn.show_images(np.swapaxes(self.H[-2][:1,:16,:,:].as_numpy_array(),0,1),(4,4),bg="white")
                    # print self.H[-1]
                else:
                    plt.subplot(231);    
                    nn.show_images(self.H[0][0,:,:,:].reshape(1,self.H[0].shape[1],self.H[0].shape[2],self.H[0].shape[3]),(1,1));      
                    plt.subplot(232); 
                    nn.show_images(self.H[-1][0,:,:,:].reshape(1,self.H[-1].shape[1],self.H[-1].shape[2],self.H[-1].shape[3]),(1,1));      

                    plt.subplot(233); 

                    self.plot_train() 
                    plt.subplot(234)
                    # if self.H[1].shape[1]>=16:

                    # H1 = self.H[1].as_numpy_array()
                    # H1 = H1.reshape(16*100,28*28)
                    # print np.nonzero(H1)[0].shape
                    nn.show_images(np.swapaxes(self.H[-2][:1,:16,:,:].as_numpy_array(),0,1),(4,4),bg="white")                              
                    # else:
                       # nn.show_images(np.swapaxes(self.H[1][:1,:8,:,:].as_numpy_array(),0,1),(2,4),bg="white")                              
                    plt.subplot(235)
                    # if w1.shape[0]>16:
                    nn.show_images(w1[:16,:,:,:],(4,4))    
                    # else: 
                        # nn.show_images(w1[:8,:,:,:],(2,4))
                    plt.subplot(236)     
                    # if w2.shape[0]>=16:       
                    # print w2.shape
                    # if self.cfg.dataset == "svhn-torch":
                        # nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:16,:,:,:],(4,4),unit=1,yuv=1)

                    if self.cfg[-1].type=="convolution":
                        nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:16,:,:,:],(4,4),unit=1)
                    if self.cfg[-1].type=="deconvolution":
                        nn.show_images(w2[:16,:,:,:],(4,4),unit=1)
                # else:
                    # nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:,:8,:,:],(2,4),unit=True)             

            if nn.is_interactive(): plt.show()
            else: plt.draw(); plt.pause(.01)  






            if self.cfg.dataset_extra=="generate": #dense
                for k in self.cfg.index_dense:
                    if self.cfg[k].l2_activity!=None: index = k 

                if nn.is_interactive(): plt.figure(num=1, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')    
                plt.figure(1)    
                plt.subplot2grid((2,3), (0,0), colspan=1); self.plot_train()    
                
                plt.subplot2grid((2,3), (0,1), colspan=2)       

                if self.cfg.dataset == "mnist": 
                    nn.show_images(w2[:50],(5,10))         
                plt.subplot2grid((2,3), (1,1), colspan=1); 
                # if self.cfg.dataset == "mnist": 
                # print self.H[index].shape
                x = self.H[index][:,0].as_numpy_array()
                y = self.H[index][:,1].as_numpy_array()
                plt.plot(x,y,'bo')
                # plt.grid()

                x = self.T_sort[:,0].as_numpy_array()
                y = self.T_sort[:,1].as_numpy_array()
                plt.plot(x,y,'ro')
                # plt.grid()    

                # x = self.test_rand[:,0].as_numpy_array()
                # y = self.test_rand[:,1].as_numpy_array()
                # plt.plot(x,y,'go')
                plt.grid()   

                # nn.show_images(self.H[0][0].reshape(1,1,28,28),(1,1))        
 
                plt.subplot2grid((2,3), (1,2), colspan=1); 
                if self.cfg.dataset == "mnist": 
                    nn.show_images(self.H[-1][0].reshape(1,1,28,28),(1,1))        


        if self.cfg.dataset == "mnist" and self.cfg.dataset_extra  in ("vae","generate"):
            plt.figure(num=5, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')
            temp = nn.randn((64,784))
            self.test_mode = True    
            self.feedforward(temp)
            self.test_mode = False
            nn.show_images(self.H[-1].reshape(64,1,28,28),(8,8))  
            if visual_params['save']:
                plt.savefig(self.cfg.directory+self.cfg.name+"_samples.png",format="png")
            else:
                plt.draw(); plt.pause(.01)

                    # plt.figure(num=5, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')
                    # temp = nn.randn((64,784))

                    # NUM = 1000

                # X = dp.X[:1000,:]
                # self.feedforward(X)
                    # fig = plt.figure(num=6, figsize=(15,10), dpi=80, facecolor='w', edgecolor='k')
                    # plt.clf() 
                    # plt.plot(self.mu[:,0].as_numpy_array(),self.mu[:,1].as_numpy_array(),'o')
                # print 'max sigma',self.sigma.max()
                # print 'mean sigma',self.sigma.mean()

                # print 'max mu',self.mu.max()
                # print 'mean mu',self.mu.mean()
                    # ells = [Ellipse(xy=(self.mu[i,0],self.mu[i,1]), width=self.sigma[i,0], height=self.sigma[i,1]) for i in range(NUM)]

                    # ax = fig.add_subplot(111, aspect='equal')
                    # for e in ells:
                    #     ax.add_artist(e)
                    # ax.set_xlim(-5, 5)
                    # ax.set_ylim(-5, 5)
                    # plt.draw(); plt.pause(.01)
        














        # if self.cfg.dataset in ("mnist","cifar10") and self.cfg.arch=='dense' and self.cfg.learning=="auto" and len(self.cfg)>3:
        #     plt.figure(2)   
        #     w3,b3 = self.weights[3]
        #     w4,b4 = self.weights[4]
        #     H2 = gp.eye(self.H[2].shape[1])[:100]
        #     H3 = nn.dot(H2,w3)
        #     H4 = nn.dot(H3,w4)
        #     nn.show_images(H4,(10,10))
        #     if visual_params['save']:
        #         plt.savefig(self.cfg.directory+self.cfg.name+"_deep.png",format="png")  
        #     else: plt.draw(); plt.pause(.01)                           

###########################shows H[0] and H[-1]###################3
        # if self.cfg.learning == "auto":
        #     plt.figure(2)
        #     if self.cfg.dataset == "mnist" and self.cfg.arch == "dense":
        #         plt.subplot(121); nn.show_images(self.H[0][:1,:],(1,1)) 
        #         plt.subplot(122); nn.show_images(self.H[-1][:1,:],(1,1));        
        #     if self.cfg.dataset == "mnist" and self.cfg.arch == "conv": 
        #         plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(1, 28, 28,1)) 
        #         plt.subplot(122); nn.show_images(self.H[-1][:,:,:,0].reshape(1,28,28,1));
        #     elif self.cfg.dataset == "cifar10": #CIFAR10
        #         plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(3, 32, 32,1),(1,1)) 
        #         plt.subplot(122); nn.show_images(self.H[-1][0].T.reshape(3,32,32,1),(1,1))
        #     elif self.cfg[0].shape==[1, 8, 8]: #MNIST patches
        #             plt.subplot(121); nn.show_images(self.H[0][:,:,:,8:9],(1,1)) 
        #             plt.subplot(122); nn.show_images(self.H[-1][8].T.reshape(1,8,8,1),(1,1))
        #     elif self.cfg[0].shape==[3, 8, 8]: #CIFAR10 patches
        #         if u==None: 
        #             plt.subplot(121); nn.show_images(self.H[0][:,:,:,0].reshape(3, 8, 8,1),(1,1)) 
        #             plt.subplot(122); nn.show_images(self.H[-1][0].T.reshape(3,8,8,1),(1,1))
        #         else:
        #             plt.subplot(121); nn.show_images(whiten_undo(X.reshape(192,1000000).T[0].as_numpy_array(),u,s).T.reshape(3,8,8,1),(1,1),unit=True)
        #             self.feedforward(x.reshape(3,8,8,1))
        #             plt.subplot(122); nn.show_images(whiten_undo(self.H[-1][0].as_numpy_array(),u,s).T.reshape(3,8,8,1),(1,1),unit=True)
###########################shows H[0] and H[-1]###################3




            # x=X[:,:,:,:1]
            # self.feedforward(x)
            # plt.subplot(131)
            # nn.show_images(self.H[2][:,:,:,:1],(1,1))

            # plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
            # plt.subplot(131)            
            # self.plot_train(a=2)            
            # plt.subplot(132)
            # nn.show_images(np.swapaxes(self.H[1][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
            # plt.subplot(133)
            # # if self.dataset == "cifar": nn.show_images(self.H1[:,:,:,:1].sum(0).reshape(1,32,32,1),(1,1))
            # # elif self.dataset == "mnist": nn.show_images(self.H1[:,:,:,:1].sum(0).reshape(1,28,28,1),(1,1))
            # plt.show() 



        # elif self.cfg.arch == "conv": 
        #     if self.cfg[0].shape[0]<4: nn.show_images(w1[:16,:,:,:],(4,4)) #convnet
        

                
        
        # if self.cfg.arch == "conv":
        #     plt.figure(2)
        #     self.feedforward(x)        
        #     if nn.is_interactive(): plt.figure(num=None, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
        #     plt.subplot(121); nn.show_images(np.swapaxes(self.H[1][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
        #     plt.subplot(122); nn.show_images(np.swapaxes(self.H[2][:16,:,:,:1].as_numpy_array(),0,3),(4,4),bg="white")
        #     if nn.is_interactive(): plt.show()
        #     else: plt.draw(); plt.pause(.01)            
class NeuralNetPredict:
    def predict(self,address,dp):
        img = scipy.ndimage.imread(address)
        print img.shape
        size_1 = int(img.shape[1]*1.0/img.shape[0]*224.0)
        offset = int((size_1-224.0)/2)
        # print size_1,offset
        img = scipy.misc.imresize(img, (224,size_1), interp='bilinear', mode=None)[:,offset:offset+224,:]
        print img.shape
        img = img.reshape(224**2,3).T.reshape(1,3,224,224)
        img = img.astype("double")
        nn.show_images(img,(1,1),unit = True)
        assert nn.backend == nn.GnumpyBackend
        img = nn.garray(img - dp.data_mean.reshape(1,3,256,256)[:,:,:224,:224])

        if dp.mode == "vgg": img = img[:,::-1,:,:]        
        
        self.feedforward(img)
        H_cpu = self.H[-1].as_numpy_array()
        mask = nn.NumpyBackend.k_sparsity_mask(H_cpu,5,1)
        # H_cpu *= mask
        index = np.nonzero(mask)
        # print index[1]
        # print H_cpu[:100]
        for i in xrange(len(index[1])):
            print index[1][i],dp.words(index[1][i]),H_cpu[index][i]


class NeuralNetOther:


    def init_weights(self,initial_weights,silent_mode):
        if initial_weights == "layers": 
            for k in range(1,self.size):
                if not(self.cfg[k].type=="pooling"):
                    layer = self.cfg[k]
                    wk,bk = self.weights[k]
                    assert (self.cfg[k].initW != None and self.cfg[k].initB != None)
                    wk.ravel()[:] = self.cfg[k].initW * nn.randn(layer.num_weights-layer.num_filters)
                    if type(self.cfg[k].initB) == str:
                        bk[:] = 1.0*int(self.cfg[k].initB)*nn.ones(layer.num_filters)
                    else:
                        bk[:] = self.cfg[k].initB * nn.randn(layer.num_filters)             
            if not silent_mode: print "Initialized Using Layers"        
        elif (type(initial_weights) == gp.garray or type(initial_weights) == np.ndarray): 
            self.weights.mem[:] = initial_weights
            if not silent_mode: print "Initialized Using initial_weights"
        elif type(initial_weights)==float: self.weights.randn(initial_weights)
        elif initial_weights != None: raise Exception("Wrong Initialization!")
        else: print "Continue ..."

        if self.cfg.want_tied: 
            for hidden_pairs in self.cfg.tied_list:  self.weights.make_tied_copy(*hidden_pairs)
        
        # print self.weights.mem.as_numpy_array()
        # w1,b1 = self.weights[1] 
        # w2,b2 = self.weights[2] 
        # print w1[:5,:5]
        # print w2[:5,:5].T
        # print "------"



  
    def plot_train(self,a=0):
        plt.grid(True)
        plt.plot(self.err_train_epoch,self.err_train)
        
    def plot_test(self):
        # if b==None: b=self.epoch
        plt.grid(True)
        plt.plot(self.err_test_epoch,self.err_test)   

    def load_text(self,name):
        if nn.backend==nn.GnumpyBackend: print "gnumpy"; self.weights.mem=nn.garray(np.loadtxt("./out/"+name, delimiter=','))
        else:                            print "numpy" ; self.weights.mem=np.loadtxt("./out/"+name, delimiter=',')

    def save_text(self,name):       
        if nn.backend==nn.GnumpyBackend: print "gnumpy"; np.savetxt("./out/"+name, self.weights.mem.as_numpy_array(), delimiter=',')
        else:                            print "numpy" ; np.savetxt("./out/"+name, self.weights.mem, delimiter=',')

    def load_npz(self,name):
        f=np.load(name); w = f['w']
        # print w.shape
        # print self.weights.mem.shape
        if nn.backend==nn.GnumpyBackend: self.weights.mem[:]=nn.garray(w)
        else:                            self.weights.mem[:]=w

    def save_npz(self,name):                         
        if nn.backend==nn.GnumpyBackend: np.savez(name, w = self.weights.mem.as_numpy_array().astype("float32"))
        # else:                            np.savetxt(name, self.weights.mem, delimiter=',')        

    def save(self,name):
        self.weights.save(name)

    def load(self,name):
        self.weights.load(name)


    def show_filters(self,size=(10,20),scale = 1,unit = 1):
        w1,b1=self.weights[1]
        w2,b2=self.weights[-1]
        if self.cfg.arch == "dense" and self.cfg.learning == "auto":
            # plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')            
            if self.cfg.dataset == "mnist":
                # print w2.shape
                nn.show_images(w2[:size[0]*size[1]],(size[0],size[1]),scale=scale,unit=unit)            
            elif self.cfg.dataset == "cifar10": 
                nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],3,32,32),(size[0],size[1]),scale=scale,unit=unit)      
            elif self.cfg.dataset == "frey": 
                nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],3,20,20),(size[0],size[1]),scale=scale,unit=unit)                           
            elif self.cfg.dataset == "cifar10-patch": #CIFAR10 dense patches
                size_ = int(np.sqrt(self.cfg[0].shape/3))
                nn.show_images(w2[:size[0]*size[1]].reshape(size[0]*size[1],3,size_,size_),(size[0],size[1]),scale=scale,unit=unit)  
            elif self.cfg.dataset == "natural": #CIFAR10 dense patches
                    size = int(np.sqrt(self.cfg[0].shape))
                    nn.show_images(w2[:256].reshape(256,1,size,size),(16,16),unit=unit,scale=scale) 

        if self.cfg.learning == "auto" and self.cfg.arch == "conv":
            # plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')
            
            # if self.cfg.dataset == "mnist": 
            # nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:size[0]*size[1],:,:,:],(size[0],size[1]),unit=unit)
            # print w2.shape               
            if self.cfg[-1].type=="convolution":
                num_filters = w2.shape[1] 
                if size==None:
                    nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:num_filters,:,:,:],(4,num_filters/4),scale=scale)                
                else:
                    print 'ali',np.swapaxes(w2.as_numpy_array(),0,1).shape
                    nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:size[0]*size[1],:,:,:],(size[0],size[1]),scale=scale)                

            if self.cfg[-1].type=="deconvolution":  
                # print 'ali',size
                num_filters = w2.shape[0] 
                nn.show_images(w2.as_numpy_array()[:num_filters,:,:,:],(4,num_filters/4),unit=1,scale=scale)            # elif:
                # nn.show_images(np.swapaxes(w2.as_numpy_array(),0,1)[:num_filters,:,:,:],(4,num_filters/4))                


        if self.cfg.learning == "disc" and self.cfg.arch == "dense":            
            if self.cfg[1].type=="dense":
                plt.figure(num=None, figsize=(30,90), dpi=80, facecolor='w', edgecolor='k')
                if type(self.cfg[0].shape)==int: nn.show_images(w1[:,:size[0]*size[1]].T,size) #MNIST dense
                elif self.cfg[0].shape[0]==1: nn.show_images(w1[:,:size[0]*size[1]].T,size) #MNIST dense
                elif self.cfg[0].shape[0]==3: nn.show_images(w1[:,:size[0]*size[1]].reshape(3,32,32,size[0]*size[1]),size[::-1]) #CIFAR10 dense     
        if self.cfg.learning == "disc" and self.cfg.arch == "conv":                  
            nn.show_images(w1[:size[0]*size[1],:,:,:],size,scale=scale,unit=unit)
        plt.show()


            
    def gradient_check(self):

        backend_backup = nn.backend
        nn.set_backend("numpy")
        assert self.cfg.want_dropout == False
        # assert self.cfg[-1].activation==nn.softmax
        assert nn.backend == nn.NumpyBackend

        if self.cfg[0].type=="convolution": x=nn.randn((2,self.cfg[0].shape[0],self.cfg[0].shape[1],self.cfg[0].shape[2]))
        else: x=nn.randn((2,self.cfg[0].shape))

        if self.cfg[self.size-1].type == "dense": 
            t=nn.zeros((2,self.cfg[-1].shape))
            t[0,np.random.randint(self.cfg[-1].shape)] = 1; t[1,np.random.randint(self.cfg[-1].shape)] = 1; #since we are using a trick in cross-entropy cost function in order not to get nan, t values should be either 0 or 1.
            # row_sums = t.sum(axis=1);t = t / row_sums[:, np.newaxis] #for softmax gradient checking, rows should sum up to one.
        else: 
            t=nn.randn((2,self.cfg[-1].shape[0],self.cfg[-1].shape[1],self.cfg[-1].shape[2]))
            # print self.cfg[-1].shape

        epsilon=.00001
        for k in range(0,self.size):
            if self.cfg[k].type == "dense":
                self.weights.randn(.01)
                self.compute_grad(x,t) #is it necessary to have this inside the loop? don't think so. 

                if k==0: continue
                wk,bk=self.weights[k]
                dwk,dbk=self.dweights[k]
                f=self.feedforward(x,t)
                wk[0,0]+=epsilon
                f_new=self.feedforward(x,t)
                df=(f_new-f)
                print k,df/epsilon/dwk[0,0]
                f=self.feedforward(x,t)
                bk[0,0]+=epsilon
                f_new=self.feedforward(x,t)
                df=(f_new-f)
                print k,df/epsilon/dbk[0,0]
            if self.cfg[k].type in ("convolution","deconvolution"):
                self.weights.randn(.01)
                self.compute_grad(x,t)

                if k==0: continue
                wk,bk=self.weights[k]
                dwk,dbk=self.dweights[k]
                f=self.feedforward(x,t)
                # print wk.shape
                wk[0,0,2,0]+=epsilon
                f_new=self.feedforward(x,t)
                df=(f_new-f)
                # print f_new,f
                print k,df/epsilon/dwk[0,0,2,0]
                f=self.feedforward(x,t)
                bk[0,0]+=epsilon
                f_new=self.feedforward(x,t)
                df=(f_new-f)
                print k,df/epsilon/dbk[0,0]

        nn.backend = backend_backup          

    # def gradient_check_numpy_ae(self,X,T):
    #     self.weights.randn(.01)
    #     self.compute_grad(X,T,0,2)
    #     epsilon=.00001
    #     for k in self.cfg.index_dense:
    #         wk,bk=self.weights[k]
    #         dwk,dbk=self.dweights[k]
    #         f=self.compute_cost(x,t)
    #         wk[0,0]+=epsilon
    #         f_new=self.compute_cost(x,t)
    #         df=(f_new-f)
    #         print k,df/epsilon/dwk[0,0]
    #         f=self.compute_cost(x,t)
    #         bk[0]+=epsilon
    #         f_new=self.compute_cost(x,t)
    #         df=(f_new-f)
    #         print k,df/epsilon/dbk[0]
    #     for k in self.cfg.index_convolution:
    #         if k==0: continue
    #         wk,bk=self.weights[k]
    #         dwk,dbk=self.dweights[k]
    #         f=self.compute_cost(x,t)
    #         wk[0,1,2,3]+=epsilon
    #         f_new=self.compute_cost(x,t)
    #         df=(f_new-f)
    #         print k,df/epsilon/dwk[0,1,2,3]
    #         f=self.compute_cost(x,t)
    #         bk[0]+=epsilon
    #         f_new=self.compute_cost(x,t)
    #         df=(f_new-f)
    #         print k,df/epsilon/dbk[0]


    def load_weights_from(self,net_from,lst_from = None, lst_to = None, init = None):
        if lst_from == None:
            assert lst_to == None
            self.weights.mem[:] = net_from.weights.mem 
        else:
            assert lst_to != None and init != None
            nn.fill_randn(self.weights.mem)
            nn.multiply(self.weights.mem,init)
            for k in lst_from:
                print k
                w_from, b_from = net_from.weights[k]
                w_to, b_to = self.weights[k]
                assert w_from.shape == w_to.shape
                w_to[:], b_to[:] = w_from, b_from


# def whiten_undo(x,u,s):
#     return np.dot(np.dot(np.dot(x,u),np.diag(np.sqrt(1.0*s))),u.T)









    # def train(self,X,T,X_test,T_labels,
    #           momentum,
    #           batch_size,dataset_size,
    #           learning_rate,learn_params=None,
    #           initial_weights = None,
    #           visual=False,visual_params=None,
    #           report=True,report_params=None,
    #           num_epochs=10000,
    #           hyper=False,silent_mode = False):

    #     print "Type of X: ",type(X)
    #     print "Backend: ",nn.backend

    #     if self.cfg.cost == "hinge": 
    #         T = ((T-.5)*2).copy()
        
    #     self.init_weights(initial_weights,silent_mode)
    #     self.finalize(batch_size)   

    #     num_batch = int(dataset_size/batch_size)
    #     v = cn.WeightSet(self.cfg)        
    #     tic = time.time()
        
    #     self.err_train = np.zeros((num_epochs),'float32')
    #     self.err_test = np.zeros((num_epochs),'int32')

    #     if visual and not nn.is_interactive(): 
    #         plt.figure(num=1, figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')
    #         if self.cfg.learning == "auto": plt.figure(2)
    #         if self.cfg.arch == "conv": plt.figure(2)            

    #     if report_params['save']: 
    #         f = open("./save/"+report_params['save']+".txt", 'w')
    #         for dic_k,dic_v in self.cfg.dic.items():
    #             f.write(str(dic_k)+"   :   "+str(dic_v)+"\n")
    #         f.write('-----------------------------------\n')
    #         f.close()

    #     self.cost_epoch = 0
    #     for epoch in range(1,num_epochs+1):       
    #         self.epoch = epoch
    #         self.err_train[epoch-1] = self.cost_epoch/num_batch #dataset size or batch size greater than the actual still works!!!!!!
    #         self.cost_epoch = 0
            
    #         if learn_params['epoch_1'] and learn_params['new_learning_rate_1']:
    #             if epoch >= learn_params['epoch_1']: learning_rate = learn_params['new_learning_rate_1']

    #         if learn_params['epoch_2'] and learn_params['new_learning_rate_2']:
    #             if epoch >= learn_params['epoch_2']: learning_rate = learn_params['new_learning_rate_2']                

    #         train_str = "Epoch:"+str(epoch)+"   Training Error:"+str(self.err_train[epoch-1])+"   Time:"+str(round(time.time()-tic,2))+"   Learning Rate:"+str(learning_rate)
    #         if (epoch % 1 == 0 and not silent_mode): 
    #             print train_str
    #         if report and self.epoch%report_params['interval']==0: 
    #             self.err_test[epoch-1:epoch-1+report_params['interval']] = self.test(X_test,T_labels);               
    #             test_str =  "Test Error:"+str(self.err_test[epoch-1])
    #             print test_str
    #             if report_params['save']: 
    #                 f = open("./save/"+report_params['save']+".txt", 'a')
    #                 f.write(train_str+"\n")
    #                 f.write(test_str+"\n")
    #                 f.close()

    #         tic = time.time()

    #         if visual and self.epoch%visual_params['interval']==0 and not silent_mode: self.visualize(X,visual_params)

    #         plt.sys.stdout.flush()            
            
    #         for l in range(num_batch):

    #             x = self.data_convertor(X,batch_size*l,batch_size*(l+1))
    #             t = self.data_convertor(T,batch_size*l,batch_size*(l+1))

    #             cost = self.compute_grad(x,t)
    #             self.cost_epoch += cost

    #             self.dweights*=-learning_rate
    #             v*=momentum
    #             v+=self.dweights #v = momentum*v - learning_rate*self.dweights
    #             self.weights+=v #v=m*v-l*dw; w=w+v


    #         if not silent_mode and epoch!=num_epochs and visual and self.epoch%visual_params['interval']==visual_params['interval']-1: clear_output()

    #     if silent_mode: return self.err_test
    #     if visual and self.epoch == num_epochs: plt.show()     
