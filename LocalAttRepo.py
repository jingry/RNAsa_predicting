# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:45:46 2021

@author: huangyuyao
"""

import os, sys, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statistics
from math import sqrt
import pickle
import pandas as pd
import re
import time

import copy
from torch.utils.checkpoint import checkpoint



def graphTrans(subDataIn ):
    d_onehot, l_windowsize  = subDataIn.shape
    adjMat = torch.zeros([d_onehot,d_onehot],dtype=torch.float)
    sqrtDegMat = torch.zeros([d_onehot,d_onehot],dtype=torch.float)
    covMat = torch.zeros([d_onehot,d_onehot],dtype=torch.float)
    
    for i in range(l_windowsize-1):
        if subDataIn[:,i].sum()==0 or subDataIn[:,i+1].sum()==0:
            continue
        pos1 = np.argmax(subDataIn[:,i])
        pos2 = np.argmax(subDataIn[:,i+1])
        adjMat[pos1,pos2] += 1
    
    for i in range(d_onehot):
        sqrtDegMat[i,i] += torch.sqrt(adjMat[i,:].sum() + adjMat[:,i].sum() - adjMat[i,i] )
    
    covMat = torch.eye(d_onehot) + sqrtDegMat.matmul(adjMat).matmul(sqrtDegMat)
    
    return adjMat, sqrtDegMat,covMat



class computAttMask(nn.Module):
    def __init__(self,oneHotDimOut,dropout_rate,localWindowSize = 3,device=None,NFusioniter=3,headNum=4):
        super().__init__()
        self.NFusioniter = NFusioniter
        self.fusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.attFusion = seqFusionAttentionNoBias(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device)
        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate,withFF=True)

        self.biasModel = DNN(oneHotDimOut, dropout_rate,dimOut=1,withLN=True)
        self.device = device
        self.localWindowSize = localWindowSize
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,fusionArr,seqMask=None):
        batchSize = fusionArr.size(0)
        seqLen = fusionArr.size(1)

        biasArr = (self.sigmoid(self.biasModel(fusionArr)) * seqLen).round()

        
        localMask = torch.arange(seqLen,device=self.device).view([1,1,seqLen]).expand([batchSize,seqLen,seqLen]) \
            - biasArr.view([batchSize,seqLen,1])
        localMask = localMask.abs() + 1
        tmpBool = localMask>self.localWindowSize+1
        localMask = localMask/localMask.abs()
        localMask.masked_fill_(tmpBool,float('-inf'))

        
        return localMask
    

    
class computeAttentionWeightProj(nn.Module):
    def __init__(self,cDim,headNum,device=None,dtype=None,localWindowSize=10):
        super().__init__()
        self.cDim = cDim
        self.device = device
        self.type = torch.float
        self.localWindowSize = localWindowSize
        self.headNum = headNum
        self.softMaxLayer = nn.Softmax(dim=-1)
        
    def forward(self,Q,K,V,G,seqMask=None,ADPMask=None):
        batchSize = Q.size(1)
        seqLen = Q.size(2)
        
       
        A = torch.einsum('hnsi,hnti->hnst',Q,K)/torch.sqrt(torch.tensor(self.cDim,device=self.device,dtype=self.type)) #HxNxSxS
        if not seqMask is None:
            seqMask = seqMask.view([1,seqMask.size(0),1,seqMask.size(1)]).expand(self.headNum,-1,seqLen,-1) #HxNxSxS

            A.masked_fill_(seqMask,float('-inf'))
            
        if not self.localWindowSize is None:
            localMask = torch.zeros([seqLen,seqLen],device=self.device)
            maskLocation = ((torch.arange(seqLen,device=self.device)*seqLen+ torch.arange(seqLen,device=self.device)).view([seqLen,1])+torch.arange(-self.localWindowSize,self.localWindowSize+1,device=self.device))
            tmpMin = torch.arange(seqLen,device=self.device)*seqLen
            tmpMax = torch.arange(seqLen,device=self.device)*seqLen + seqLen - 1
            maskLocation = torch.max(torch.min(maskLocation,tmpMax.view([-1,1])),tmpMin.view([-1,1]))
            localMask = localMask.view([-1]).scatter_(0,maskLocation.flatten(),1).view([1,1,seqLen,seqLen]).expand([self.headNum,batchSize,-1,-1])

            A.masked_fill_(~localMask.bool(),float('-inf'))
        
        A = self.softMaxLayer(A) #HxNxSxS
        if not ADPMask is None:
            A = A * ADPMask
        V = torch.einsum('hnst,hnti,hnsi->nshi',A,V,G)#NxSxHxDc
        
        return V

class seqFusionAttentionWeightProjCP(nn.Module):
    '''
    N_batch x S_seq x D_dim
    '''
    def __init__(self,qDim,kDim=None,vDim=None,outDim=None,cDim=None,headNum=4,
                 dropout_rate=0.15,device=None,dtype=None,localWindowSize=10):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if kDim is None:
            kDim = qDim
        if vDim is None:
            vDim = kDim
        
        if outDim is None:
            outDim = vDim
        if cDim is None:
            cDim = round(qDim/headNum)
        self.register_buffer('QTrans',None)
        self.QTrans = nn.Parameter(torch.empty((qDim, cDim*headNum), **factory_kwargs), requires_grad = True)
                
        self.register_buffer('KTrans',None)
        self.KTrans = nn.Parameter(torch.empty((kDim, cDim*headNum), **factory_kwargs), requires_grad = True)
        
        self.register_buffer('VTrans',None)
        self.VTrans = nn.Parameter(torch.empty((vDim, cDim*headNum), **factory_kwargs), requires_grad = True)


        
        self.GTrans = nn.Linear(vDim, cDim*headNum)
        
        self.outTrans = nn.Linear(cDim*headNum,outDim,bias=True)
        self.softMaxLayer = nn.Softmax(dim=-1)

        self.cDim = cDim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.headNum = headNum
        self.device = device
        self._reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        self.localWindowSize=localWindowSize

        
        if device == 'cpu':
            self.type = torch.float
        else:
            self.type = torch.float
        self.computeAttention = computeAttentionWeightProj(cDim,headNum=headNum,device=device,dtype=self.type,localWindowSize=localWindowSize)
        
        
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QTrans)
        torch.nn.init.xavier_uniform_(self.KTrans)
        torch.nn.init.xavier_uniform_(self.VTrans)

        torch.nn.init.zeros_(self.GTrans.weight)
        torch.nn.init.ones_(self.GTrans.bias)
        torch.nn.init.zeros_(self.outTrans.weight)
        torch.nn.init.zeros_(self.outTrans.bias)
        
    def forward(self,Qin,Kin,Vin,seqMask=None):
        batchSize = Qin.size(0)
        seqLen = Qin.size(1)
        Q = torch.einsum('nsi,ij->nsj',Qin,self.QTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        K = torch.einsum('nsi,ij->nsj',Kin,self.KTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        V = torch.einsum('nsi,ij->nsj',Vin,self.VTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
  
        ADPMask = torch.empty([self.headNum,batchSize,seqLen,seqLen],device=self.device,dtype=torch.float16).uniform_() > self.dropout_rate
        
        G = self.GTrans(Vin).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        G = self.sigmoid(G)

        if self.training:
            V = checkpoint(self.computeAttention, Q,K,V,G,seqMask,ADPMask)
        else:
            V = self.computeAttention(Q, K, V, G, seqMask=seqMask, ADPMask=None)
         
        
        return self.outTrans(V.reshape([V.size(0),V.size(1),-1]))
    
    
class seqFusionAttentionWeightProj(nn.Module):
    '''
    N_batch x S_seq x D_dim
    '''
    def __init__(self,qDim,kDim=None,vDim=None,outDim=None,cDim=None,headNum=4,
                 dropout_rate=0.15,device=None,dtype=None,localWindowSize=10):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if kDim is None:
            kDim = qDim
        if vDim is None:
            vDim = kDim
        
        if outDim is None:
            outDim = vDim
        if cDim is None:
            cDim = round(qDim/headNum)
        self.register_buffer('QTrans',None)
        self.QTrans = nn.Parameter(torch.empty((qDim, cDim*headNum), **factory_kwargs), requires_grad = True)
                
        self.register_buffer('KTrans',None)
        self.KTrans = nn.Parameter(torch.empty((kDim, cDim*headNum), **factory_kwargs), requires_grad = True)
        
        self.register_buffer('VTrans',None)
        self.VTrans = nn.Parameter(torch.empty((vDim, cDim*headNum), **factory_kwargs), requires_grad = True)

        
        self.GTrans = nn.Linear(vDim, cDim*headNum)
        
        self.outTrans = nn.Linear(cDim*headNum,outDim,bias=True)
        self.softMaxLayer = nn.Softmax(dim=-1)

        self.cDim = cDim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.headNum = headNum
        self.device = device
        self._reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        self.localWindowSize=localWindowSize

        
        if device == 'cpu':
            self.type = torch.float
        else:
            self.type = torch.float
        
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QTrans)
        torch.nn.init.xavier_uniform_(self.KTrans)
        torch.nn.init.xavier_uniform_(self.VTrans)

        torch.nn.init.zeros_(self.GTrans.weight)
        torch.nn.init.ones_(self.GTrans.bias)
        torch.nn.init.zeros_(self.outTrans.weight)
        torch.nn.init.zeros_(self.outTrans.bias)
        
    def forward(self,Qin,Kin,Vin,seqMask=None):
        batchSize = Qin.size(0)
        seqLen = Qin.size(1)
        Q = torch.einsum('nsi,ij->nsj',Qin,self.QTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        K = torch.einsum('nsi,ij->nsj',Kin,self.KTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        V = torch.einsum('nsi,ij->nsj',Vin,self.VTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        
        G = self.GTrans(Vin).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        G = self.sigmoid(G)

        
       
        A = torch.einsum('hnsi,hnti->hnst',Q,K)/torch.sqrt(torch.tensor(self.cDim,device=self.device,dtype=self.type)) #HxNxSxS
        if not seqMask is None:
            seqMask = seqMask.view([1,seqMask.size(0),1,seqMask.size(1)]).expand(self.headNum,-1,seqLen,-1) #HxNxSxS

            A.masked_fill_(seqMask,float('-inf'))
            
        if not self.localWindowSize is None:
            localMask = torch.zeros([seqLen,seqLen],device=self.device)
            maskLocation = ((torch.arange(seqLen,device=self.device)*seqLen+ torch.arange(seqLen,device=self.device)).view([seqLen,1])+torch.arange(-self.localWindowSize,self.localWindowSize+1,device=self.device))
            tmpMin = torch.arange(seqLen,device=self.device)*seqLen
            tmpMax = torch.arange(seqLen,device=self.device)*seqLen + seqLen - 1
            maskLocation = torch.max(torch.min(maskLocation,tmpMax.view([-1,1])),tmpMin.view([-1,1]))
            localMask = localMask.view([-1]).scatter_(0,maskLocation.flatten(),1).view([1,1,seqLen,seqLen]).expand([self.headNum,batchSize,-1,-1])

            A.masked_fill_(~localMask.bool(),float('-inf'))
        
        A = self.softMaxLayer(A) #HxNxSxS
        if self.dropout_rate > 0:
            A = self.dropout(A)    
                
        V = torch.einsum('hnst,hnti,hnsi->nshi',A,V,G)#NxSxHxDc

        return self.outTrans(V.reshape([V.size(0),V.size(1),-1]))

class seqFusionAttentionNoBias(nn.Module):
    '''
    N_batch x S_seq x D_dim
    '''
    def __init__(self,qDim,kDim=None,vDim=None,outDim=None,cDim=None,headNum=4,
                 dropout_rate=0.15,device=None,dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(seqFusionAttentionNoBias,self).__init__()
        if kDim is None:
            kDim = qDim
        if vDim is None:
            vDim = kDim
        
        if outDim is None:
            outDim = vDim
        if cDim is None:
            cDim = round(qDim/headNum)
        self.register_buffer('QTrans',None)
        self.QTrans = nn.Parameter(torch.empty((qDim, cDim*headNum), **factory_kwargs), requires_grad = True)
                
        self.register_buffer('KTrans',None)
        self.KTrans = nn.Parameter(torch.empty((kDim, cDim*headNum), **factory_kwargs), requires_grad = True)
        
        self.register_buffer('VTrans',None)
        self.VTrans = nn.Parameter(torch.empty((vDim, cDim*headNum), **factory_kwargs), requires_grad = True)


        
        self.GTrans = nn.Linear(vDim, cDim*headNum)
        
        self.outTrans = nn.Linear(cDim*headNum,outDim,bias=True)
        self.softMaxLayer = nn.Softmax(dim=-1)

        self.cDim = cDim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.headNum = headNum
        self.device = device
        self._reset_parameters()
        self.sigmoid = torch.nn.Sigmoid()
        if device == 'cpu':
            self.type = torch.float
        else:
            self.type = torch.float
        
    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.QTrans)
        torch.nn.init.xavier_uniform_(self.KTrans)
        torch.nn.init.xavier_uniform_(self.VTrans)

        torch.nn.init.zeros_(self.GTrans.weight)
        torch.nn.init.ones_(self.GTrans.bias)
        torch.nn.init.zeros_(self.outTrans.weight)
        torch.nn.init.zeros_(self.outTrans.bias)
        
    def forward(self,Qin,Kin,Vin,seqMask=None,attMask=None):
        batchSize = Qin.size(0)
        seqLen = Qin.size(1)
        Q = torch.einsum('nsi,ij->nsj',Qin,self.QTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        K = torch.einsum('nsi,ij->nsj',Kin,self.KTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        V = torch.einsum('nsi,ij->nsj',Vin,self.VTrans).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        
        G = self.GTrans(Vin).view([batchSize,seqLen,self.headNum,self.cDim]).permute([2,0,1,3]) #HxNxSxDc
        G = self.sigmoid(G)

        
       
        A = torch.einsum('hnsi,hnti->hnst',Q,K)/torch.sqrt(torch.tensor(self.cDim,device=self.device,dtype=self.type)) #HxNxSxS
        if not seqMask is None:
           
            seqMask = torch.einsum('ns,nt->nst',seqMask.float(),seqMask.float()).bool().view([1,batchSize,seqLen,seqLen]).expand([self.headNum,-1,-1,-1])
            A.masked_fill_(seqMask,float('-inf'))
        if not attMask is None:
            A = A + attMask
        
        A = self.softMaxLayer(A) #HxNxSxS
        if self.dropout_rate > 0:
            A = self.dropout(A)
        

        V = torch.einsum('hnst,hnti,hnsi->nshi',A,V,G)#NxSxHxDc
        out = self.outTrans(V.reshape([V.size(0),V.size(1),-1]))

        return out
    
class DNN(nn.Module):
    def __init__(self, dimIn,dropout_rate, dimOut=1, withLN=False,zeroLL=False):
        
        super(DNN, self).__init__()

        self.fc1 = nn.Linear( dimIn,128)
      
        self.fc2 = nn.Linear( 128,128)
        self.fc3 = nn.Linear( 128,dimOut)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.layerNorm = nn.LayerNorm(dimIn)
        self.withLN = withLN
        self._reset_parameters()
        if zeroLL:
            self.zeroLastLayer()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc3.weight)  
        
    def zeroLastLayer(self):
        torch.nn.init.zeros_(self.fc3.weight) #last output layer
        torch.nn.init.zeros_(self.fc3.bias) #last output layer
        
    def forward(self, x):
        if self.withLN:
            x = self.layerNorm(x)
        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)
        
        x = self.fc3(x)        
        x = self.relu(x) 
        x = x.squeeze(-1)
        return x 

class DNN_simple_last(nn.Module):
    def __init__(self, dimIn,dropout_rate, dimOut=1, withLN=False, zeroLL=False):
        
        super(DNN_simple_last, self).__init__()

        self.fc1 = nn.Linear( dimIn,128)
      
        self.fc2 = nn.Linear( 128,128)
        self.fc3 = nn.Linear( 128,dimOut)
        
        
        
        self.relu = nn.ReLU()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.layerNorm = nn.LayerNorm(dimIn)
        self.withLN = withLN
        self._reset_parameters()
        if zeroLL:
            self.zeroLastLayer()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc3.weight)  
    
    def zeroLastLayer(self):
        torch.nn.init.zeros_(self.fc3.weight) #last output layer
        torch.nn.init.zeros_(self.fc3.bias) #last output layer
    
    def forward(self, x):
        if self.withLN:
            x = self.layerNorm(x)
        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)
        
        x = self.fc3(x)        
 
        x = x.squeeze(-1)
        return x 
        
class FeedForward(nn.Module):
    def __init__(self, dimIn,dropout_rate,alpha=8 ,zeroLL=True, withFF=True):
        
        super(FeedForward, self).__init__()
        self.layerNorm = nn.LayerNorm(dimIn)
        self.layerNorm1 = nn.LayerNorm(dimIn*alpha)
        self.fc1 = nn.Linear( dimIn,dimIn*alpha)
      
        self.fc2 = nn.Linear( dimIn*alpha,dimIn)
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.withFF = withFF
        self._reset_parameters()
        if zeroLL:
            self.zeroLastLayer()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc2.weight)  

    def zeroLastLayer(self):
        torch.nn.init.zeros_(self.fc2.weight) #last output layer
        torch.nn.init.zeros_(self.fc2.bias) #last output layer
        
    def forward(self, x):
        if self.withFF:
            x = self.layerNorm(x)

        x = self.fc1(x)

        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)

        
        return x 

class Transition(nn.Module):
    def __init__(self, dimIn,dropout_rate ,zeroLL=True):
        
        super( ).__init__()
        self.layerNorm1 = nn.LayerNorm(dimIn)
        self.layerNorm2 = nn.LayerNorm(dimIn)
        self.fc1 = nn.Linear( dimIn,dimIn)
      
        self.fc2 = nn.Linear( dimIn,dimIn)
        self.fc3 = nn.Linear( dimIn,dimIn)
        
        self.relu = nn.ReLU()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self._reset_parameters()
        if zeroLL:
            self.zeroLastLayer()
    
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc3.weight)    

    def zeroLastLayer(self):
        torch.nn.init.zeros_(self.fc3.weight) #last output layer
        torch.nn.init.zeros_(self.fc3.bias) #last output layer
        
    def forward(self, x):
        x = self.layerNorm1(self.dropout(x))
        x = x + self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        x = self.layerNorm2(x)
        return x 
    
class fusionEncodeBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum=4,dropout_rate=0.15,NEBDiter=6,device='cpu'):
        super().__init__()
        localWindowSize = 10
        
        self.NEBDiter = NEBDiter
        self.OneHotlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm = nn.LayerNorm(oneHotDimOut)
        self.OneHotlayerNorm_i1 = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm_i1 = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm_i1 = nn.LayerNorm(oneHotDimOut)
        self.OneHotlayerNorm_i2 = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm_i2 = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm_i2 = nn.LayerNorm(oneHotDimOut)
        self.OneHotlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.OneHotlayerNorm_o2 = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm_o2 = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm_o2 = nn.LayerNorm(oneHotDimOut)

        self.attOneHot = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attGraphNext = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attGraphLast = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attOneHot_i1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attOneHot_i2 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attGraphNext_i1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.attGraphLast_i1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=localWindowSize)
        self.OneHotFF = FeedForward(oneHotDimOut, dropout_rate)
        self.NextStepFF = FeedForward(oneHotDimOut, dropout_rate)
        self.LastStepFF = FeedForward(oneHotDimOut, dropout_rate)
        
        self.OneHotFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
        self.NextStepFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
        self.LastStepFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
        
    def forward(self,graphEbdArrLast,graphEbdArrNext,oneHotEbdArr,seqMask):
        for iterEbd in range(self.NEBDiter):
            

            
            tmpGraphEbdArrLast = self.LastSteplayerNorm(graphEbdArrLast)
            graphEbdArrLast = graphEbdArrLast + self.attGraphLast(tmpGraphEbdArrLast,tmpGraphEbdArrLast,tmpGraphEbdArrLast,seqMask =seqMask)
            graphEbdArrLast = graphEbdArrLast + self.LastStepFF(graphEbdArrLast)
            
            tmpGraphEbdArrNext = self.NextSteplayerNorm(graphEbdArrNext)
            graphEbdArrNext = graphEbdArrNext + self.attGraphNext(tmpGraphEbdArrNext,tmpGraphEbdArrNext,tmpGraphEbdArrNext,seqMask =seqMask)
            graphEbdArrNext = graphEbdArrNext + self.NextStepFF(graphEbdArrNext)
            
            tmpOneHotEbdArr = self.OneHotlayerNorm(oneHotEbdArr)
            oneHotEbdArr = oneHotEbdArr + self.attOneHot(tmpOneHotEbdArr,tmpOneHotEbdArr,tmpOneHotEbdArr, seqMask  = seqMask)
            oneHotEbdArr = oneHotEbdArr + self.OneHotFF(oneHotEbdArr)
            
            tmpOneHotEbdArr = self.OneHotlayerNorm_i1(oneHotEbdArr)
            
            tmpGraphEbdArrLast = self.LastSteplayerNorm_i1(graphEbdArrLast)
            graphEbdArrLast = graphEbdArrLast + self.attGraphLast_i1(tmpGraphEbdArrLast,tmpOneHotEbdArr,tmpOneHotEbdArr,seqMask =seqMask)
            graphEbdArrLast = graphEbdArrLast + self.LastStepFF_i1(graphEbdArrLast) 
            
            tmpGraphEbdArrNext = self.NextSteplayerNorm_i1(graphEbdArrNext)
            graphEbdArrNext = graphEbdArrNext + self.attGraphNext_i1(tmpGraphEbdArrNext,tmpOneHotEbdArr,tmpOneHotEbdArr,seqMask =seqMask)
            graphEbdArrNext = graphEbdArrNext + self.NextStepFF_i1(graphEbdArrNext)
            
            tmpGraphEbdArrLast = self.LastSteplayerNorm_i2(graphEbdArrLast)
            tmpGraphEbdArrNext = self.NextSteplayerNorm_i2(graphEbdArrNext)
            tmpOneHotEbdArr = self.OneHotlayerNorm_i2(oneHotEbdArr)
            
            tmpOut1 = self.attOneHot_i1(tmpOneHotEbdArr,tmpGraphEbdArrNext, tmpGraphEbdArrNext, seqMask  = seqMask)
            tmpOut2 = self.attOneHot_i2(tmpOneHotEbdArr,tmpGraphEbdArrLast, tmpGraphEbdArrLast, seqMask  = seqMask)
            oneHotEbdArr = oneHotEbdArr + tmpOut1 + tmpOut2
            oneHotEbdArr = oneHotEbdArr + self.OneHotFF_i1(oneHotEbdArr)
        return graphEbdArrLast,graphEbdArrNext,oneHotEbdArr
    

class selfEncodeLocalBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum,dropout_rate,NFusioniter,device):
        super().__init__()
        self.NFusioniter = NFusioniter
        self.fusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm2 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.attFusion = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=1)
        self.attFusion1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=3)
        self.attFusion2 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=5)

        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.fusionFF1 = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
    
    def forward(self,fusionArr,seqMask=None):
        for iterFusion in range(self.NFusioniter):
            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusion(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion1(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion2(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.fusionFF(fusionArr)

        return fusionArr
    
class selfEncodeBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum,dropout_rate,NFusioniter,device):
        super().__init__()
        self.NFusioniter = NFusioniter
        self.fusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm2 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.attFusion = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=1)
        self.attFusion1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=3)
        self.attFusion2 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=5)
        self.attFusion3 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)
        self.attFusion4 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)
        self.attFusion5 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)

        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.fusionFF1 = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
    
    def forward(self,fusionArr,seqMask=None):
        for iterFusion in range(self.NFusioniter):
            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusion(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion1(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion2(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.fusionFF(fusionArr)
            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusion3(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion4(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion5(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.fusionFF1(fusionArr)
        return fusionArr
    

    
    

class selfEncodeLongShortRangeBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum,dropout_rate,NFusioniter,device):
        super().__init__()
        self.NFusioniter = NFusioniter
        self.fusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.attFusion = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=2)
        self.attFusion1 = seqFusionAttentionWeightProj(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=6)

        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.fusionFF1 = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.maskGenerator = computAttMask(oneHotDimOut, dropout_rate,localWindowSize = 2,device=device,NFusioniter=device,headNum=headNum)
        self.maskGenerator1 = computAttMask(oneHotDimOut, dropout_rate,localWindowSize = 6,device=device,NFusioniter=device,headNum=headNum)
        self.attFusionLong = seqFusionAttentionNoBias(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device)
        self.attFusionLong1 = seqFusionAttentionNoBias(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device)
        self.headNum = headNum
        
    def forward(self,fusionArr,seqMask=None):
        batchSize=fusionArr.size(0)
        seqLen = fusionArr.size(1)
        for iterFusion in range(self.NFusioniter):
            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusion(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion1(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.fusionFF(fusionArr)
            mask1 = self.maskGenerator(fusionArr).view([1,batchSize,seqLen,seqLen]).expand([self.headNum,-1,-1,-1]).clone()
            mask2 = self.maskGenerator1(fusionArr).view([1,batchSize,seqLen,seqLen]).expand([self.headNum,-1,-1,-1]).clone()

            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusionLong(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask,attMask=mask1)
            fusionArr = fusionArr + self.attFusionLong1(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask,attMask=mask2)
            fusionArr = fusionArr + self.fusionFF1(fusionArr)
        return fusionArr
    
   
class selfEncodeConvBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum,dropout_rate,NFusioniter,device):
        super().__init__()
        self.NFusioniter = NFusioniter
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.fusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.fusionlayerNorm_o1 = nn.LayerNorm(oneHotDimOut)

        self.attFusion3 = seqFusionAttentionWeightProjCP(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)
        self.attFusion4 = seqFusionAttentionWeightProjCP(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)
        self.attFusion5 = seqFusionAttentionWeightProjCP(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device,localWindowSize=None)

        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.fusionFF1 = FeedForward(oneHotDimOut, dropout_rate,withFF=True)
        self.conv1 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=3,padding=1,dilation=1)
        self.conv2 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=5,padding=4,dilation=2)
        self.conv3 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=7,padding=12,dilation=4)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.binNorm1 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm2 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm3 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm4 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        
    def forward(self,fusionArr,seqMask=None):
        for iterFusion in range(self.NFusioniter):
            tmpfusionArr = fusionArr.clone().transpose(-2,-1)
            tmpfusionArr =self.dropout(self.elu(self.binNorm2(tmpfusionArr)))
            tmpfusionArr = self.conv2(tmpfusionArr)

            tmpfusionArr =self.dropout(self.elu(self.binNorm3(tmpfusionArr)))
            fusionArr = fusionArr + self.conv3(tmpfusionArr).transpose(-2,-1)
            

            tmpfusionArr = self.fusionlayerNorm(fusionArr)
            fusionArr = fusionArr + self.attFusion3(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion4(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.attFusion5(tmpfusionArr,tmpfusionArr,tmpfusionArr,seqMask =seqMask)
            fusionArr = fusionArr + self.fusionFF1(fusionArr)
        return fusionArr
    

class restoreDecoderBlock(nn.Module):
    def __init__(self,oneHotDimOut,headNum,dropout_rate,NDeciter,device):
        super().__init__()
        self.NDeciter = NDeciter
        self.selfAttlayer = seqFusionAttentionNoBias(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device)
        self.decAttlayer = seqFusionAttentionNoBias(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,device=device)
        self.decFF = FeedForward(oneHotDimOut, dropout_rate)
        self.decLayerNorm = nn.LayerNorm(oneHotDimOut)
        self.decLayerNorm_o1 = nn.LayerNorm(oneHotDimOut)
        self.decLayerNorm_o2 = nn.LayerNorm(oneHotDimOut)
        self.encLayerNorm = nn.LayerNorm(oneHotDimOut)
        
    def forward(self,decIn,encOut,windowMask,seqMask):
        decodeArr = decIn
        tmpEncArr = self.encLayerNorm(encOut)
        for iterDec in range(self.NDeciter):
            tmpDecArr = self.decLayerNorm(decodeArr)
            tmpDecArr1 = self.decLayerNorm_o1(self.selfAttlayer(tmpDecArr,tmpDecArr,tmpDecArr,seqMask = seqMask))
            tmpDecArr2 = self.decLayerNorm_o2(tmpDecArr + self.selfAttlayer(tmpDecArr1,tmpEncArr,tmpEncArr,seqMask = windowMask))
            decodeArr = decodeArr + self.decFF(tmpDecArr2) + tmpDecArr1
        return decodeArr

class convEncoder(nn.Module):
    def __init__(self,dimIn,dimOut,seqLast=False):
        super().__init__()
        self.conv = torch.nn.Conv1d(dimIn, dimOut, kernel_size=3,padding=1,dilation=1)
        self.seqLast = seqLast
    def forward(self,x):
        if not self.seqLast:
            x = x.transpose(-2,-1)
        x = self.conv(x)
        if not self.seqLast:
            return x.transpose(-2,-1)
        else:
            return x
class convFusionResBlock(nn.Module):
    def __init__(self,oneHotDimOut,dropout_rate=0.15,repeatTime=3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.conv1 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=3,padding=1,dilation=1)
        self.conv2 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=5,padding=4,dilation=2)
        self.conv3 = torch.nn.Conv1d(oneHotDimOut, oneHotDimOut, kernel_size=7,padding=12,dilation=4)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.repeatTime = repeatTime
        self.binNorm1 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm2 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm3 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
        self.binNorm4 = nn.InstanceNorm1d(oneHotDimOut, affine=True)
    

        
    def forward(self,x,seqLast=False):
        if not seqLast:
            x = x.transpose(-2,-1)
        x = self.binNorm1(self.conv1(x))

        for i in range(self.repeatTime):
            x1 =self.dropout(self.elu(self.binNorm2(x)))
            x1 = self.conv2(x1)

            x1 =self.dropout(self.elu(self.binNorm3(x1)))
            x = x1 + self.conv3(x1)
        if not seqLast:
            return x.transpose(-2,-1)
        else:
            return x

        
class myModel(nn.Module):
    def __init__(self,oneHotDimIn, oneHotDimOut,RNNDimOut,AELatentDim,windowSize,
                 dropout_rate,headNum,sampleNum, lossBias=0.,AEBias=1.,preBias=1., 
                 classNum=128,bool_parameter = False, device=None,
                 NEBDiter = 4, NFusioniter = 4, NCycleiter=4, NDeciter=6,asaClamp=[-50., 500.],
                 randMaskThres=0.15):
        super(myModel, self).__init__()
        self.oneHotDim = oneHotDimOut
        self.onehotEbd = nn.Linear(oneHotDimIn, oneHotDimOut)
        self.onehotEbdConv = convEncoder(oneHotDimIn, oneHotDimOut)
        self.graphEbdNext = nn.Linear(oneHotDimIn, oneHotDimOut)
        self.graphEbdLast = nn.Linear(oneHotDimIn, oneHotDimOut)
        self.onehotOutEbd = nn.Linear(oneHotDimIn, oneHotDimOut)
                
        self.graphTrans = graphTrans
        self.w = 1
        self.NEBDiter = NEBDiter 
        self.NFusioniter = NFusioniter
        self.NCycleiter = NCycleiter
        
        self.asaClamp = asaClamp
        
        self.layerNorm = nn.LayerNorm(oneHotDimOut)
        self.layerNormAtt = nn.LayerNorm(RNNDimOut)
        
        
        self.inputLayerNorm1 = nn.LayerNorm(oneHotDimOut)
        self.inputLayerNorm2 = nn.LayerNorm(oneHotDimOut)
        self.inputLayerNorm3 = nn.LayerNorm(oneHotDimOut)
        self.decInputLayerNorm = nn.LayerNorm(oneHotDimOut)
        
        self.OneHotlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.NextSteplayerNorm = nn.LayerNorm(oneHotDimOut)
        self.LastSteplayerNorm = nn.LayerNorm(oneHotDimOut)

        self.classEbdlayerNorm = nn.LayerNorm(oneHotDimOut)

        
        self.lastFusionlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.lastlayerNorm = nn.LayerNorm(oneHotDimOut)
        self.recyclingNorm = nn.LayerNorm(oneHotDimOut)
        
        self.OneHotFF = FeedForward(oneHotDimOut, dropout_rate)
        self.NextStepFF = FeedForward(oneHotDimOut, dropout_rate)
        self.LastStepFF = FeedForward(oneHotDimOut, dropout_rate)
        
        self.OneHotFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
        self.NextStepFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
        self.LastStepFF_i1 = FeedForward(oneHotDimOut, dropout_rate)
       

        
        self.fusionEncodeBlock = fusionEncodeBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NEBDiter=NEBDiter,device=device)
        self.selfEncodeBlock = selfEncodeBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NFusioniter=NFusioniter,device=device)
        self.selfEncodeLocalBlock = selfEncodeLocalBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NFusioniter=NFusioniter,device=device)
        self.restoreDecoderBlock = restoreDecoderBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NDeciter=NDeciter,device=device)
        
        self.selfEncodeConvBlock = selfEncodeConvBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NFusioniter=NFusioniter,device=device)
        self.selfEncodeLongShortRangeBlock = selfEncodeLongShortRangeBlock(oneHotDimOut,headNum=headNum,dropout_rate=dropout_rate,NFusioniter=NFusioniter,device=device)

        self.dnn = nn.Linear(oneHotDimOut,1)
        

        self.register_buffer('lossWeight',torch.nn.Parameter(data=torch.Tensor([1,2,1]), requires_grad = bool_parameter))
        
        self.convBlock = convFusionResBlock(oneHotDimOut,dropout_rate=dropout_rate)
        
        self.preLossFunc = nn.MSELoss(reduction='none')

        self.classLossFunc = nn.CrossEntropyLoss(reduction='sum',ignore_index=-10)
        self.classWindowLossFunc = nn.CrossEntropyLoss(reduction='sum')
        self.conflictLossFunc = nn.CrossEntropyLoss(reduction='sum')
        self.restoreLoss = nn.CrossEntropyLoss(reduction='sum',ignore_index=-10)
        
        self.lossRelu = nn.ReLU()
        self.lossBias = lossBias
        
        self.adjMatDict = {}
        

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.elu = nn.ELU()
        
        self.classNum = classNum
        self.M1 = DNN_simple_last(oneHotDimOut,
                                  self.dropout_rate,dimOut=self.classNum,
                                  withLN=True,zeroLL=True)
        self.M1bias = DNN_simple_last(oneHotDimOut,
                                  self.dropout_rate,dimOut=self.classNum,
                                  withLN=True,zeroLL=True)
        
        

        self.fusionFF = FeedForward(oneHotDimOut, dropout_rate)

        self.transition = Transition(oneHotDimOut,dropout_rate)
        
        self.MRestore = DNN_simple_last(oneHotDimOut,
                                  self.dropout_rate,dimOut=oneHotDimIn,
                                  withLN=True,zeroLL=True)
        
        self.softmax = nn.Softmax(-1)
        
        self.classBin = None
        self.stepSize = None
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        
        self.device = device

        self.sampleIndex = 0
        self.iterationNum = 0
        self.randMaskThres = randMaskThres
        self.asa_std = torch.tensor([400, 400, 350, 350],dtype=torch.float,device=self.device)

    
    def _reset_parameters(self):
        nn.init.normal_(self.onehotEbd.weight)
        nn.init.normal_(self.graphEbdNext.weight)
        nn.init.normal_(self.graphEbdLast.weight)

        nn.init.normal_(self.onehotOutEbd.weight)

    def setBinsBoundary(self,lb=0,ub=400):
        stepSize = (ub-lb)/self.classNum

        self.classBin = torch.arange(lb,ub+stepSize*0.5,stepSize).to(self.device)
        self.stepSize = stepSize
    
    def findBinsFromBoundary(self,asaIn):
        lb = torch.min(asaIn)
        ub =torch.max(asaIn)
        stepSize = (ub-lb)/self.classNum

        self.classBin = torch.arange(lb,ub+stepSize*0.5,stepSize).to(self.device)
    
    def generateClassOneHot(self,yIn):

        tmpPos = torch.argmin(torch.abs(yIn.view(yIn.size(0),1)-self.classBin[:-1]),axis=-1)

        out = torch.zeros([yIn.size(0),yIn.size(1),self.classNum],dtype=torch.float).to(self.device)
        indices = tmpPos + torch.arange(yIn.size(1),device=self.device).reshape([yIn.size(0),yIn.size(1)])*self.classNum
        out.view([yIn.size(0),-1]).scatter_(1,indices.long(),1)

        return out
           
    def generateClassOneHotPos(self,yIn):
        tmpPos = torch.argmin(torch.abs(yIn.view([yIn.size(0),yIn.size(1),1])-self.classBin[:-1]),axis=-1)

        return tmpPos
    
    def forward(self, oneHotOriArrIn, covMats, posEncMats, bool_parameter=False, optimizer=None,predict=False,preTrain=False, fillNum=0):
        
        batchSize = oneHotOriArrIn.size(0)
        seqLen = oneHotOriArrIn.size(-1)

        adjMat = self.w * covMats
        oneHotArr = oneHotOriArrIn.permute([0,2,1])
        graphArrLast = adjMat.matmul(oneHotOriArrIn).permute([0,2,1])
        graphArrNext = (oneHotOriArrIn.permute([0,2,1]).matmul(adjMat))

        fusionArr = self.onehotEbdConv(oneHotArr)
        
        asa_max = torch.einsum('d,nds->ns',self.asa_std,oneHotOriArrIn)

        
        tmpArr = self.selfEncodeLocalBlock(fusionArr)

        pred_asa = self.sigmoid(self.dnn(self.dropout(self.elu(self.layerNorm(tmpArr ))))).reshape(batchSize,seqLen)
        return pred_asa,asa_max


    def computeLoss(self,forwardOut,y,xOneHotOri,extraBias=1600):
        pre,asa_max = forwardOut

        preLossPerBase = self.preLossFunc(pre*asa_max,y)

        
        preLoss = preLossPerBase.sum() 

        return preLoss

    def preTrainLoss(self,forwardOut, xOneHotOri):

        restoreArr, restoreMask, seqMask = forwardOut
        restoreTarget = torch.argmax(xOneHotOri.clone().transpose(-2,-1),axis=-1)#NxS
        AETarget = restoreTarget.clone()
        restoreTarget.masked_fill_(~restoreMask,-10)
        AETarget.masked_fill_(seqMask,-10)
        resLoss = self.restoreLoss(restoreArr.view(-1,restoreArr.size(-1)), restoreTarget.flatten()) 
        AELoss = self.restoreLoss(restoreArr.view(-1,restoreArr.size(-1)), AETarget.flatten()) 
        return resLoss + AELoss


class PositionalEncoding:

    def __init__(self, d_model: int, max_len: int = 5000):

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        self.pe = pe
        self.d_model = d_model
        
    def encoding(self,endPos):
        '''
        output: seqLenth * d_model
        '''
        return self.pe[:endPos, :]  
        
    
    
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataDict,PEobj):
        'Initialization'
        self.dataDict = dataDict
        self.idSeq = np.sort(list(dataDict.keys()))

        self.dict_base = dict(zip('AGCU', [3,2,1,0]))   

        self.baseDim = 4
        self.PE = PEobj
        self.dtype = torch.float

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idSeq)
    
    def norn_adj(self, X):
        input_dim_1 = self.baseDim
        A = torch.zeros((input_dim_1, input_dim_1), dtype = torch.float)
            
        A_view = A.view(-1)
        x_size = X.size(-1)
        indices = X.narrow(-1, 0, x_size - 1) * A.stride(0) * A.stride(1) + X.narrow(-1, 1, x_size - 1) * A.stride(1)

        A_view.scatter_(0, indices.long(), 1,reduce='add')
        
        
        A_hat = A + torch.eye(input_dim_1, dtype = torch.float)
        D_hat = (A.sum(axis = 0).flatten() + A.sum(axis = 1).flatten() - A.diag() + torch.ones([input_dim_1])*2).pow(-1.0).diag_embed()
        
        return A_hat,D_hat
    
    def to_oneHot(self, X):
        input_dim_1 = self.baseDim
        out = torch.zeros(X.size(0),input_dim_1)
        indices = X + torch.arange(X.size(0))*input_dim_1
        out.view(-1).scatter_(0,indices.long(),1)
        return out
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        seqId = self.idSeq[index]
        seq = self.dataDict[seqId]['seq']
        
        if not 'onehot' in self.dataDict[seqId]:
            Xnum = torch.tensor([self.dict_base[s] for s in seq])
            Xonehot = self.to_oneHot(Xnum)
            A_hat,D_hat = self.norn_adj(Xnum)
            covMat = (D_hat**0.5).matmul(A_hat).matmul(D_hat**0.5)
            PosEnc = self.PE.encoding(len(seq))
            self.dataDict[seqId]['onehot'] = Xonehot
            self.dataDict[seqId]['covMat'] = covMat
            self.dataDict[seqId]['PosEnc'] = PosEnc
        else:
            Xonehot = self.dataDict[seqId]['onehot']
            covMat = self.dataDict[seqId]['covMat']
            PosEnc = self.dataDict[seqId]['PosEnc']
        y = torch.tensor(self.dataDict[seqId]['asa'],dtype=self.dtype)
        
        return Xonehot.transpose(-2,-1).float(),covMat.float(),PosEnc,y



class WrappedModel(nn.Module):
     def __init__(self,oneHotDimIn, oneHotDimOut,RNNDimOut,AELatentDim,
                  windowSize,dropout_rate,sampleNum,headNum=4,lossBias=0.0,
                  device=None,NEBDiter =4,NFusioniter=4,NCycleiter=4,cpPATH=None,
                  earlyStopThres=500):
         
        super(WrappedModel, self).__init__()       
        self.myModel= myModel(oneHotDimIn, oneHotDimOut,RNNDimOut,AELatentDim,
                              windowSize,dropout_rate,headNum=headNum,lossBias=lossBias,
                              sampleNum=sampleNum,bool_parameter = False,device=device,
                              NEBDiter = NEBDiter,NFusioniter=NFusioniter,NCycleiter=NCycleiter)
        if device == 'cpu':
            self.myModel = self.myModel.float()
        else:
            self.myModel = self.myModel.float()
        # self.lossWeightLog = []
        self.device = device
        self.windowSize = windowSize
        self.setBinsBoundary()
        self.bestValLoss = None
        self.bestState_dict = None
        self.bestEpoch = None
        self.cpPATH = cpPATH
        self.earlyStopThres = earlyStopThres
     
     def checkCPFile(self):
         if self.cpPATH is None:
             print('No checkpoint PATH provided')
             return False
         if os.path.exists(self.cpPATH):
             # cp = torch.load(PATH)
             return True
         return False
     
     def setBinsBoundary(self,lb=0,ub=400):
         self.myModel.setBinsBoundary(lb=lb,ub=ub)
     
     def findBinsFromBoundary(self,asaIn):
         self.myModel.findBinsFromBoundary(asaIn)

     def fit(self,training_generator,validation_generator,lr,epoches,batchSize,opt=None,
             bool_parameter = False,stage=None, preTrain=False,opt_state_dict=None,
             reduce_rate=0.99,max_lr = 0.01,least_lr = 0.0001, max_norm=0.1):

        self.train()
        weight_decay = 0
        
        if opt is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 

        else:
            optimizer = opt(self.parameters(), lr=lr, weight_decay=weight_decay) 
 
        if not opt_state_dict is None:
            optimizer.load_state_dict(opt_state_dict)
        stepSize = int(len(training_generator) * 5)
        epoch_start = 0
        savedStage = stage
        if self.checkCPFile():
            cp = torch.load(self.cpPATH)
            self.load_state_dict(cp['model_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            epoch_start = cp['epoch']

            self.bestValLoss = cp['bestValLoss']
            self.bestState_dict = cp['bestState_dict']
            self.bestEpoch = cp['bestEpoch']
            savedStage = cp['savedStage']
            if not savedStage == stage:
                epoch_start = 0
            
        if stage == 'warming up':

            if not stage == savedStage:
                print('warming up has been finished, skip warming up')
                return

            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=int(epoches))
            

        for e in range(epoch_start,epoches):
            self.train()

            
            rowInd = 0
            epochLoss = 0.
            batchLossWeight = None
            batchMse = None
            curr_lr = lr
            stackSize = 0

            for Xb,covMatsb,posEncMatsb,yb in training_generator:

                Xb,covMatsb,posEncMatsb,yb = Xb.to(self.device),covMatsb.to(self.device),posEncMatsb.to(self.device),yb.to(self.device)

                optimizer.zero_grad()

                forwardOut = self.myModel(Xb,covMatsb,posEncMatsb,bool_parameter,optimizer=optimizer,preTrain=preTrain)

                if preTrain:
                    loss = self.myModel.preTrainLoss(forwardOut, Xb)
                else:
                    loss = self.myModel.computeLoss(forwardOut,yb,Xb)
                

                epochLoss += loss.detach().item()
                stackSize += Xb.size(1)
                

                loss.backward()
    
                optimizer.step()

                
                
                rowInd += Xb.size(2)

            if stage == 'warming up':
                scheduler.step()


            if preTrain:
                epochLoss = epochLoss / rowInd
            else:
                epochLoss = epochLoss / rowInd

            if stage == 'warming up':
                curr_lr = scheduler.get_last_lr()[-1]
            if e %1 == 0:
                del(Xb,covMatsb,posEncMatsb,yb)
                del(forwardOut)
                if not self.device == 'cpu':
                    torch.cuda.empty_cache()
                if preTrain:
                    val_loss = self.validate(validation_generator, batchSize, preTrain=preTrain)
                    print('********epoch:%d, lr:%2.e*******' %(e,curr_lr))
                    print('Training Set:',epochLoss)  
                    print('Validation Set:',val_loss)
                else:
                    val_loss,val_mse,val_pcc=self.validate(validation_generator, batchSize, preTrain=preTrain)
                    
                    if self.bestValLoss is None:
                        self.bestValLoss = val_loss
                        self.bestState_dict = copy.deepcopy(self.state_dict())
                        self.bestEpoch = e
                    else:
                        if val_loss < self.bestValLoss:
                            self.bestValLoss = val_loss
                            self.bestState_dict = copy.deepcopy(self.state_dict())
                            self.bestEpoch = e
                    print('********epoch:%d, lr:%2.e*******' %(e,curr_lr))
                    print(time.ctime())
                    print('Training Set:',epochLoss)  
                    print('Validation Set:',val_loss,val_mse,val_pcc)
                    print('Curr Best:',self.bestEpoch,self.bestValLoss)
            if e %20 == 19 and not self.cpPATH is None:
                torch.save({
                    'epoch': e+1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': LOSS,
                    'bestValLoss': self.bestValLoss,
                    'bestState_dict': self.bestState_dict,
                    'bestEpoch': self.bestEpoch,
                    'savedStage': stage,
                    }, self.cpPATH)
            
            #early stop
            if e %1 == 0:
                if not stage == 'warming up':
                    if e - self.bestEpoch > self.earlyStopThres:
                        return optimizer.state_dict().copy()

        return optimizer.state_dict().copy()
     
     def validate(self,validation_generator,batchSize,preTrain=False):
           
           self.eval()
           with torch.no_grad():        
               rowInd = 0
               y_out = []
               y_ori = []
               val_loss = None
               val_mse_in_loss = None
               for Xb,covMatsb,posEncMatsb,yb in validation_generator:
                   Xb,covMatsb,posEncMatsb,yb = Xb.to(self.device),covMatsb.to(self.device),posEncMatsb.to(self.device),yb.to(self.device)
           
                   forwardOut = self.myModel(Xb,covMatsb,posEncMatsb,predict=False,preTrain=preTrain)

                   if preTrain:
                       rowInd += Xb.size(2)     
                       loss = self.myModel.preTrainLoss(forwardOut, Xb)
                       if val_loss is None:
                           val_loss = loss.item()
                       else:
                           val_loss += loss.item()
                   else:
                       loss = self.myModel.computeLoss(forwardOut,yb,Xb)
                       pred = forwardOut[0] * forwardOut[1]
                       y_out += list(pred.flatten().cpu().detach().numpy())
                       rowInd += Xb.size(2)     
                       y_ori += list(yb.flatten().detach().cpu().numpy())
                       if val_loss is None:
                           val_loss = loss.item()

                       else:
                           val_loss += loss.item()

           if not preTrain:
               val_mse = mean_squared_error(y_ori,y_out)
               val_pcc = np.corrcoef(y_ori,y_out)[0,1]
           del(Xb,covMatsb,posEncMatsb,yb)
           del(forwardOut)
           
           if not self.device == 'cpu':
               torch.cuda.empty_cache()
           if preTrain:
               return val_loss/rowInd
           return val_loss/rowInd,val_mse,val_pcc
       
     def predict(self,validation_generator,batchSize):
        
        self.eval()
        with torch.no_grad():        
            # rowInd = 0
            y_out = []
            y_ori = []
            y_out_norm = []
            y_ori_norm = []
            y_out_batch = []
            y_ori_batch = []
            y_out_norm_batch = []
            y_ori_norm_batch = []
            for Xb,covMatsb,posEncMatsb,yb in validation_generator:
                Xb,covMatsb,posEncMatsb,yb = Xb.to(self.device),covMatsb.to(self.device),posEncMatsb.to(self.device),yb.to(self.device)
        
                forwardOut = self.myModel(Xb,covMatsb,posEncMatsb,predict=True)
                pred=forwardOut[0] * forwardOut[1]
                asa_max = forwardOut[-1]
                outList = list(pred.flatten().cpu().detach().numpy())
                oriList = list(yb.flatten().detach().cpu().numpy())
                out_norm_list = list((pred/asa_max).flatten().cpu().detach().numpy())
                ori_norm_list = list((yb/asa_max).flatten().detach().cpu().numpy())
                
                y_out += outList
                y_ori += oriList                
                y_out_norm += out_norm_list
                y_ori_norm += ori_norm_list
                
                y_out_batch.append(outList)
                y_ori_batch.append(oriList)
                y_out_norm_batch.append(out_norm_list)
                y_ori_norm_batch.append(ori_norm_list)
        
        del(Xb,covMatsb,posEncMatsb)
        del(forwardOut)
        if not self.device == 'cpu':
            torch.cuda.empty_cache()
        return y_out,y_ori,y_out_norm,y_ori_norm,y_out_batch,y_ori_batch,y_out_norm_batch,y_ori_norm_batch
    








    

class metrics(nn.Module):
    def __init__(self,predict,real,testname,path,ContactPath):
        super(metrics, self).__init__()    
        self.predict = predict
        self.real = real
        self.testname = testname 
        self.resultPath = path         
        self.ContactPath = ContactPath


        BASES = 'AUCG'
        asa_std = [400, 350, 350, 400]
        self.dict_rnam1_ASA = dict(zip(BASES, asa_std))    

        self.metrics_conbine_final = {}
        self.metrics_conbine_final['PCC'] ={}
        self.metrics_conbine_final['R2'] ={}
        self.metrics_conbine_final['MSE'] ={}


    
    
    def averagePCC(self,finalNameList,dataDict): 
        workDict = dataDict.copy()
        for i in range(len(finalNameList)):
            baseName = finalNameList[i]
            predAsa = self.predict[i]
            tmpEles = baseName.split('_')
            chainID = tmpEles[0] + '_' + tmpEles[1]
            basePos = int(tmpEles[2])
            if not 'predAsa' in workDict[chainID]:
                workDict[chainID]['predAsa'] = np.zeros_like(workDict[chainID]['asa']) 
            workDict[chainID]['predAsa'][basePos] = predAsa
            
            assert workDict[chainID]['asa'][basePos] - self.real[i] < 1e-5
            
            asa_div =  np.array([self.dict_rnam1_ASA[s] for s in workDict[chainID]['seq']])
            workDict[chainID]['asa_n'] = workDict[chainID]['asa'] / asa_div
            workDict[chainID]['predAsa_n'] = workDict[chainID]['predAsa'] / asa_div
        
        tmpList = []#for average
        for chainID in workDict:
            ori = workDict[chainID]['asa']
            pre = workDict[chainID]['predAsa']
            oneChainPCC = np.corrcoef(ori,pre)[0,1]
            oneChainR2 = r2_score(ori,pre)
            oneChainMSE = mean_squared_error(ori,pre)
            workDict[chainID]['pcc'] = oneChainPCC
            workDict[chainID]['r2'] = oneChainR2
            workDict[chainID]['mse'] = oneChainMSE
            ori = workDict[chainID]['asa_n']
            pre = workDict[chainID]['predAsa_n']
            oneChainPCC = np.corrcoef(ori,pre)[0,1]
            oneChainR2 = r2_score(ori,pre)
            oneChainMSE = mean_squared_error(ori,pre)
            workDict[chainID]['pcc_n'] = oneChainPCC
            workDict[chainID]['r2_n'] = oneChainR2
            workDict[chainID]['mse_n'] = oneChainMSE
            tmpList.append((workDict[chainID]['pcc'],workDict[chainID]['r2'],workDict[chainID]['mse'],
                           workDict[chainID]['pcc_n'],workDict[chainID]['r2_n'],workDict[chainID]['mse_n']))
        avglist = np.mean(np.array(tmpList),axis=0)
        No_normal_averagePCC,No_normal_averageR2,No_normal_averageMSE,averagePCC,averageR2,averageMSE = avglist
        
        oriList = []
        preList = []
        oriNList = []
        preNList = []
        for chainID in workDict:
            oriList += list(workDict[chainID]['asa'])
            preList += list(workDict[chainID]['predAsa'])
            oriNList += list(workDict[chainID]['asa_n'])
            preNList += list(workDict[chainID]['predAsa_n'])
        No_normal_overallPCC = np.corrcoef(oriList,preList)[0,1]
        No_normal_overallR2 = r2_score(oriList,preList)
        No_normal_overallMSE = mean_squared_error(oriList,preList)
        overallPCC = np.corrcoef(oriNList,preNList)[0,1]
        overallR2 = r2_score(oriNList,preNList)
        overallMSE = mean_squared_error(oriNList,preNList)
        print('\t\t\t\tNo Normal\tNormal')
        print('average PCC:\t%f\t%f' %(No_normal_averagePCC,averagePCC))
        print('average MSE:\t%f\t%f' %(No_normal_averageMSE,averageMSE))
        print('average R2:\t%f\t%f' %(No_normal_averageR2,averageR2))
        
        print('overall PCC:\t%f\t%f' %(No_normal_overallPCC,overallPCC))
        print('overall MSE:\t%f\t%f' %(No_normal_overallMSE,overallMSE))
        print('overall R2:\t%f\t%f' %(No_normal_overallR2,overallR2))
        return workDict
       
        
 



    
    
