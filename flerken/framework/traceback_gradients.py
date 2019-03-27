#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Juan Montesinos"
__copyright__ = "Copyright 2019"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"

import torch
import numpy as np
from math import ceil
import matplotlib as mpl
import os

__all__ = ['tracegrad']
def mps(function,input_list,processes):
    pool = mp.Pool(processes=processes)
    results = [pool.apply(function, args =(file,)) for file in input_list]
    pool.close()
    return results

def rescale(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def get_parents(named_params):
    return list(set(map(lambda x:x[:x.find('.')+1])))
def invariant(x):
    return x
class tracegrad_set():
    def __new__(cls,obj,bool_list,filter_key):
        obj.named_params = [value for keep, value in zip(bool_list, obj.named_params) if keep]
        obj.filter_key.append(filter_key+'|')
        obj.N = len(obj.named_params)
        bool_list = np.asarray(bool_list)
        if bool_list.ndim >1:
            bool_list = np.tile(bool_list,(bool_list.shape[0],1))
        obj.mean = obj.mean[bool_list]
        obj.std = obj.std[bool_list]
        obj.ma = obj.ma[bool_list]
        obj.mi = obj.mi[bool_list]
        obj.ms = obj.ms[bool_list]
        
        return obj     
class tracegrad():
    def __init__(self,model,verbose=False):
        self.N = len(list(model.named_parameters()))
        self.model = model
        self.verbose = verbose
        self.named_params = [name if 'module.' != name[:7] else name[7:] for name,_ in model.named_parameters()]
        
        
    def _reset(self):
        self.mean = []
        self.std = []
        self.ma = []
        self.mi = []
        self.ms = []
        self.filter_key = []
    def __call__(self,model_named_parameters):
        self._reset()
        for name,param in model_named_parameters:
            if self.verbose:            
                print(name)
            mean=param.grad.mean().detach().cpu().clone()
            std=param.grad.std().detach().cpu().clone()
            ma=param.grad.max().detach().cpu().clone()
            mi=param.grad.min().detach().cpu().clone()
            ms=param.grad.pow(2).mean().pow(0.5).detach().cpu().clone()
            
            self.mean.append(mean)
            self.std.append(std)
            self.ma.append(ma)
            self.mi.append(mi)
            self.ms.append(ms)
            if self.verbose:            
                print('MS: {0}, MEAN: {1}, STD:{2}'.format(ms,mean,std))
                print('Max: {0}, Min: {1}'.format(ma,mi))
        self.mean = np.asarray(self.mean)
        self.std = np.asarray(self.std)
        self.ma = np.asarray(self.ma)
        self.mi = np.asarray(self.mi)
        self.ms = np.asarray(self.ms)
    def filtro(self,layer):        
        return tracegrad_set(self,list(map(lambda x: True if layer in x else False,self.named_params)),layer)
        
    def _generate_dict(self,numpy):
        if numpy:
            return {'mean':np.asarray(self.mean),'std':np.asarray(self.std),'max':np.asarray(self.ma),'min':np.asarray(self.mi),'ms':np.asarray(self.ms),'named_params':self.named_params,'filter':self.filter_key}
        else:
            return {'mean':torch.tensor(self.mean),'std':torch.tensor(self.std),'max':torch.tensor(self.ma),'min':torch.tensor(self.mi),'ms':torch.tensor(self.ms),'named_params':self.named_params,'filter':self.filter_key}
    def grad(self,numpy=True):
        return self._generate_dict(numpy)
    def _as_image(self,lista):
        H = int(self.N**0.5//1)
        W = ceil(self.N/H)
        dif = self.N - H*W
        vector = rescale(lista,1,0)
        filled_vector = np.zeros((self.N+dif))
        filled_vector[:self.N]=vector
        return filled_vector.reshape(H,W)
    def as_image(self,function=invariant,colormap=None):

        H = int(self.N**0.5//1)
        N = ceil(self.N/H)
        s=3
        splitter = np.zeros((H,s))
#        splitter[::2,:] = torch.ones(3)
        splitter[::2,0]=1
        splitter[1::2,1]=1
        splitter[::2,2]=1
        mean = invariant(self._as_image(self.mean))
        ms = invariant(self._as_image(self.ms))
        ma = invariant(self._as_image(self.ma))
        mi = invariant(self._as_image(self.mi))
        std =invariant( self._as_image(self.std))
        
        output = np.zeros((H,5*N+4*3))
        output[:,:N]=ms
        output[:,N:N+s]=splitter
        output[:,N+s:2*N+s]=mean
        output[:,2*N+s:2*N+2*s]=splitter
        output[:,2*N+2*s:3*N+2*s]=std
        output[:,3*N+2*s:3*N+3*s]=splitter
        output[:,3*N+3*s:4*N+3*s]=ma
        output[:,4*N+3*s:4*N+4*s]=splitter
        output[:,4*N+4*s:5*N+4*s]=mi
        if colormap is not None:
            cm = mpl.cm.get_cmap(colormap)
            output = torch.from_numpy(cm(output)).permute(2,0,1)[:3,...]
        return output

def ss(x):
    return np.load(x).item()
class tracegrad_analysis(tracegrad):
    def __init__(self,path):
        if isinstance(path,str):
            path_list = self.read_folder(path)
        elif isinstance(path,list):
            path_list = path
        else:
            NotImplementedError
        dics = mps(ss,path_list)
        self._reset()
        for x in dics:
            self.mean.append(x['mean'])
            self.std.append(x['std'])
            self.ma.append(x['max'])
            self.mi.append(x['min'])
            self.ms.append(x['ms'])
        self.named_params = x['named_params']
        self.filter_key = x['filter']
        self.mean = np.asarray(self.mean)
        self.std = np.asarray(self.std)
        self.ma = np.asarray(self.ma)
        self.mi = np.asarray(self.mi)
        self.ms = np.asarray(self.ms)
    def read_folder(self,path):
        files = os.listdir(path)
        files = [os.path.join(path,file) for file in files]
        return files
    def __call__(self):
        NotImplementedError
    def _as_image(self,lista):
        NotImplementedError
    def as_image(self,lista,function=invariant,colormap=None):
        lista = invariant(lista)
        if colormap is not None:
            cm = mpl.cm.get_cmap(colormap)
            output = torch.from_numpy(cm(lista)).permute(2,0,1)[:3,...]
        else:
            output = lista
        return output
####TRACEGRAD#####            
#import sys
#sys.path.append('../')
#from models.Wnet import WNet
#model = WNet([64,128,256,512,1024,2048,4096],2,128,'ResNet3D,MP',useBN=True,input_channels=1,
#          latent_space=False,just_encoder = False,activation = None,vae=False,print_bool=False,sum_it=False) 
#loss = torch.nn.L1Loss()
#inputs=[torch.rand(1,1,256,256).requires_grad_(),torch.rand(1,2,128,3,224,224).requires_grad_()]
#output=model(*inputs)
#z = loss(output,torch.rand(1,2,256,256).requires_grad_())
#z.backward()
#
#tracker=tracegrad(model)
#tracker(model.named_parameters())
#i=tracker.as_image(colormap='viridis')
#a=tracker.filtro('conv').filtro('of_model').grad()
        

# =============================================================================
# #TRACEGRAD_ANALYSIS
# =============================================================================
#tracker=tracegrad_analysis('/home/jfm/wnetSiamese/0e6f4e8/gradient_tracking')
