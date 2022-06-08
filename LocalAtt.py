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
from math import sqrt
import pickle
import copy

import LocalAttRepo as myRepo


if __name__ == '__main__':    

    
    testDataPath = sys.argv[1]
    savedCheckPointPath = sys.argv[2]
    oneHotDimOut=int(sys.argv[3])
    dropout_rate=float(sys.argv[4])
    if len(sys.argv) < 6:
        device = 'cpu'
    else:
        device = sys.argv[5] 
    
    
    lr=0.001
    fit_lr = lr*10
    pre_lr = 0.0001
    wr_lr = 1e-3
    finetuning_lr = lr
    reduce_rate = 0.999
    
    batchSize = 1

    epoches_preTrain = 500
    epoches_WU = 1500
    epoches_fit = 1500 * 8
    epoches_FT = 50
    

    oneHotDimIn=4

    RNNDimOut=32
    AELatentDim=64
    windowSize=10
    headNum=4

    NEBDiter = 6
    NFusioniter = 4
    NCycleiter = 1
    
    num_workers = 0

    with open(testDataPath, 'rb') as f:
        testData = pickle.load(f)    
    PEobj = myRepo.PositionalEncoding(oneHotDimOut)
    pred_TS_set = myRepo.Dataset(testData,PEobj)
    TS_generator = torch.utils.data.DataLoader(pred_TS_set, batch_size=batchSize,shuffle=False,num_workers=num_workers)
    sampleNum = 0#not used

    
    print('\n ###   predict  %s ###')    
   
    Model1 = myRepo.WrappedModel(oneHotDimIn, oneHotDimOut,RNNDimOut,AELatentDim,windowSize,
                                       dropout_rate,sampleNum,headNum=headNum,lossBias=0,device=device,
                                       NEBDiter =NEBDiter,NFusioniter=NFusioniter,NCycleiter=NCycleiter)
    
    tmpCP = torch.load(savedCheckPointPath,map_location=torch.device(device))
 
    Model1.load_state_dict(tmpCP['bestState_dict'])
    if not device=='cpu':
        Model1=Model1.to(device)

    predOuts = Model1.predict(TS_generator,batchSize)
    
    predBatch = predOuts[4]
    for i in range(len(testData)):
        chainID = pred_TS_set.idSeq[i]
        seq = testData[chainID]['seq']
        batchOut = predBatch[i]
        assert len(seq) == len(batchOut)
        print('')
        print(">%s" %chainID)
        print(seq)
        print(', '.join(np.array(predBatch[i],dtype=str)))
    

          
    print('\n ###   END  %s ###')


        
      


    
    
