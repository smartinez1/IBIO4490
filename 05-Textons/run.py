#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:27:10 2019
@author: santiago
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt

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

#rr=np.random.randint(1,5)
rr=2
trainBatch=unpickle('cifar-10-python/cifar-10-batches-py/data_batch_'+str(rr))
testBatch=unpickle('cifar-10-python/cifar-10-batches-py/test_batch')
#testBatch=np.sort(testBatch)
#trainBatch=np.sort(trainBatch)

keys=list(trainBatch)

bl_train=trainBatch[keys[0]]
bl_test=testBatch[keys[0]]

lab_train=trainBatch[keys[1]]
lab_test=testBatch[keys[1]]

data_train=trainBatch[keys[2]]
data_test=testBatch[keys[2]]

filenames_train=trainBatch[keys[3]]
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
imageAmount=1000
testImages=10000

imgs=[]
classes=[]
idx=0
cl=0
usedIdxs=[]

while (len(classes)<imageAmount and cl<=9):
    
    if (lab_train[idx]==cl and usedIdxs.count(idx)==0): 
        actualImage=(data_train[idx,:,:,:])
        for jdx in range(32):
            for kdx in range(32):
                for ldx in range (3):
                    actualImage[jdx,kdx,ldx]=int(actualImage[jdx,kdx,ldx]*255)
        imAux=color.rgb2gray(actualImage)
        imgs.append(imAux)
        classes.append(lab_train[idx])
        usedIdxs.append(idx)
        idx=0
        if classes.count(cl)==(imageAmount/10):
            cl=cl+1
        
    idx=idx+1
    
stacks=imgs[0]
for idx in range(1,imageAmount):
    stacks=np.concatenate((stacks,imgs[idx]),axis=1)
    
filterResponses=fbRun(fb,stacks) 

k=96

#Compute textons from filter
from computeTextons import computeTextons
[map,textons]=computeTextons(filterResponses,k)
#Load more images
testImgs=[]
for idx in range(testImages):
    imAux=color.rgb2gray(data_test[idx,:,:,:])
    testImgs.append(imAux)
    
#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons

tmapsTrain=[]
tmapsTest=[]
for idx in range(imageAmount):
	mapTest=assignTextons(fbRun(fb,imgs[idx]),textons.transpose())    
	tmapsTrain.append(mapTest)

    
for idx in range(testImages):
	mapTest=assignTextons(fbRun(fb,testImgs[idx]),textons.transpose())    
	tmapsTest.append(mapTest)

trainHists=[]
testHists=[]
for idx in range(imageAmount):
    auxHist=histc(tmapsTrain[idx].flatten(),np.arange(k))
    trainHists.append(auxHist)
    
for idx in range(testImages):
    auxHist=histc(tmapsTest[idx].flatten(),np.arange(k))
    testHists.append(auxHist) 
    
trainHists_2=[]
for idx in range(imageAmount):
    trainHists_2.append(trainHists[idx])
    for jdx in range (k):
        trainHists_2[idx][jdx]=int(trainHists_2[idx][jdx])
        
    trainHists_2[idx]=list(trainHists_2[idx])
    classes[idx]=int(classes[idx])
  
from sklearn.ensemble import RandomForestClassifier

n_est=100
clf = RandomForestClassifier(n_estimators=n_est, max_depth=2,
                             random_state=0)
clf.fit(trainHists_2,classes)


pred_tes=[]
anns=[]    
for idx in range(testImages):
	pred_tes.append(clf.predict(np.array([testHists[idx]])))
	pred_tes[idx]=int(pred_tes[idx])
	anns.append(lab_test[idx])
	anns[idx]=int(anns[idx])
    
from sklearn.metrics import confusion_matrix
confussionMat_test=confusion_matrix(anns,pred_tes)*(1/testImages) 

# Saves relevant information into a .pickle file
model={}
model['Model']=clf
model['Textons']=textons


with open('python/model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    
