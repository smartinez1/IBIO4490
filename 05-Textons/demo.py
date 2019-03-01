#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:13:44 2019

@author: santiago
"""
import pickle
import numpy as np

with open('python/model.pickle','rb') as pickleFile:
        model=pickle.load(pickleFile)
        
clf=model['Model']        
textons=model['Textons']        
        
def unpickle(file):

    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='latin1')
        _dict['labels'] = np.array(_dict['labels'])
        _dict['data'] = _dict['data'].reshape(_dict['data'].shape[0], 3, 32, 32).transpose(0,2,3,1)

    return _dict

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

testBatch=unpickle('cifar-10-python/cifar-10-batches-py/test_batch')

keys=list(testBatch)

bl_test=testBatch[keys[0]]

lab_test=testBatch[keys[1]]

data_test=testBatch[keys[2]]

filenames_test=testBatch[keys[3]]

import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
supp=2
staSig=0.6
fb = fbCreate(support=supp, startSigma=staSig) # fbCreate(**kwargs, vis=True) for visualization

#Load sample images from disk
from skimage import color
from skimage import io
from fbRun import fbRun

testImages=np.random.randint(10000,size=5)
testImgs=[]
for idx in range(len((testImages))):
    imAux=color.rgb2gray(data_test[testImages[idx],:,:,:])
    testImgs.append(imAux)


from assignTextons import assignTextons

tmapsTest=[]  
for idx in range(len((testImages))):
	mapTest=assignTextons(fbRun(fb,testImgs[idx]),textons.transpose())    
	tmapsTest.append(mapTest)
testHists=[]    
for idx in range(len((testImages))):
    auxHist=histc(tmapsTest[idx].flatten(),np.arange(96))
    testHists.append(auxHist)     

pred_tes=[]
anns=[] 
for idx in range(len((testImages))):
	pred_tes.append(clf.predict(np.array([testHists[idx]])))
	pred_tes[idx]=int(pred_tes[idx])
	anns.append(lab_test[idx])
	anns[idx]=int(anns[idx])

import matplotlib.pyplot as plt

for idx in range(0,5):
    plt.subplot(5,1,idx+1)
    plt.imshow(testImgs[idx])  
    plt.axis('off')
    tit=('Original image, annotation='+str(anns[idx])+', prediction='+str(pred_tes[idx]))
    plt.title(tit)