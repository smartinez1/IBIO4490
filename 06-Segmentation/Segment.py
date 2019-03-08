#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 00:43:20 2019

@author: santiago
"""

def segmentByClustering(rgbImage,featureSpace,clusteringMethod,numberOfClusters):

        import sklearn.cluster as clus
        import numpy as np
        import cv2
        from skimage import color, filters
        from scipy import ndimage as ndi
        from skimage.morphology import watershed, disk
        from sklearn import mixture

        
        rgbImageFull=rgbImage 
        OGM,OGN,OGc=rgbImageFull.shape
        
        if clusteringMethod=='hierarchical':
            rgbImage=cv2.resize(rgbImageFull,(int(OGN/4), int(OGM/4))) 
        else:
            rgbImage=cv2.resize(rgbImageFull,(int(OGN/2), int(OGM/2))) 
            
          
        _map=[]
        spatial=bool(0)
    # Change to specific color space and get pixel descriptors
        if featureSpace=='rgb':
            img=rgbImage
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    pixelDescriptor.append((np.asarray((img[idx,jdx]))).astype(int))
                    
        elif featureSpace=='lab':
            img=color.rgb2lab(rgbImage)
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    aux=np.asarray((img[idx,jdx])).astype(int)
                    aux[0]=255*(aux[0]/100)
                    aux[1]=(aux[1]+127)
                    aux[2]=(aux[2]+128)
                    pixelDescriptor.append(aux.astype(int))
                    _map[idx,jdx]=aux
        elif featureSpace=='hsv':
            img=color.rgb2hsv(rgbImage)
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    aux=(255*np.asarray((img[idx,jdx]))).astype(int)
                    pixelDescriptor.append(aux)
                    _map[idx,jdx]=aux
        elif featureSpace=='rgb+xy':
            spatial=bool(1)
            img=rgbImage
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    aux=(np.asarray((img[idx,jdx]))).astype(int)
                    aux1=np.asarray([255*(idx/M),255*(jdx/N)]).astype(int)
                    aux2=np.concatenate((aux,aux1))
                    pixelDescriptor.append(aux2)

        elif featureSpace=='lab+xy':
            spatial=bool(1)
            img=color.rgb2lab(rgbImage)
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    aux=(np.asarray((img[idx,jdx]))).astype(int)
                    aux[0]=int(255*(aux[0]/100))
                    aux[1]=int((aux[1]+127))
                    aux[2]=int((aux[2]+128))
                    aux1=np.asarray([255*(idx/M),255*(jdx/N)]).astype(int)
                    aux2=np.concatenate((aux,aux1))
                    pixelDescriptor.append(aux2)
                    _map[idx,jdx]=aux
        elif featureSpace=='hsv+xy': 
            spatial=bool(1)
            img=color.rgb2hsv(rgbImage)
            pixelDescriptor=[]
            M,N,c=img.shape
            _map=img
            for idx in range(M):
                for jdx in range(N):
                    aux=(np.asarray(255*img[idx,jdx]).astype(int))
                    aux1=np.asarray([255*(idx/M),255*(jdx/N)]).astype(int)
                    aux2=np.concatenate((aux,aux1))
                    pixelDescriptor.append(aux2)
                    _map[idx,jdx]=aux
#------------------------------------------------------------                    
    # Uses the selected clustering method for segmentation
        if clusteringMethod=='kmeans':
            model=clus.KMeans(n_clusters=numberOfClusters,random_state=0,max_iter=200).fit(pixelDescriptor)
            segmentation=np.zeros((M,N))
            for idx in range(M):
                for jdx in range(N):
                    if spatial:
                        aux=(np.asarray((_map[idx,jdx])))
                        aux1=np.asarray([255*(idx/M),255*(jdx/N)]).astype(int)
                        aux2=np.concatenate((aux,aux1))
                        label=model.predict([aux2])[0]
                        segmentation[idx,jdx]=label
                    else:
                        aux=(np.asarray((_map[idx,jdx])))   
                        label=model.predict([aux])[0]
                        segmentation[idx,jdx]=label
 
        if clusteringMethod=='gmm':
            model=mixture.GaussianMixture(n_components=numberOfClusters, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=200, n_init=1)
            model.fit(pixelDescriptor)
            segmentation=np.zeros((M,N))
            for idx in range(M):
                for jdx in range(N):
                    if spatial:
                        aux=(np.asarray((_map[idx,jdx])))
                        aux1=np.asarray([255*(idx/M),255*(jdx/N)]).astype(int)
                        aux2=np.concatenate((aux,aux1))
                        label=model.predict([aux2])[0]
                        segmentation[idx,jdx]=label
                    else:
                        aux=(np.asarray((_map[idx,jdx])))   
                        label=model.predict([aux])[0]
                        segmentation[idx,jdx]=label

        if clusteringMethod=='hierarchical':
            model=clus.AgglomerativeClustering(n_clusters=numberOfClusters,compute_full_tree=True,affinity='euclidean', linkage='complete')
            if spatial:
                auxDescriptor=np.reshape(pixelDescriptor,(-1,5))
            else:
                auxDescriptor=np.reshape(pixelDescriptor,(-1,3))
            model.fit(auxDescriptor,y=None)
            labels=model.labels_
            segmentation=np.zeros((M,N))
            kdx=0
            for idx in range(M):
                for jdx in range(N):
                    segmentation[idx,jdx]=labels[kdx]
                    kdx=kdx+1
                     
        if clusteringMethod=='watershed':
            img=color.rgb2gray(rgbImageFull)
            denoised = filters.rank.median(img, disk(6))
            for i in range(2, 30):
                for j in range(2,25):
                    
                    marcadores=filters.rank.gradient(denoised, disk(j)) < i
                    marcadores, nro = ndi.label(marcadores)
                    if nro==numberOfClusters:
                        break
                if nro==numberOfClusters:
                    break            
                                            
            gradiente = filters.rank.gradient(denoised, disk(1))
            v_laber4 = watershed(gradiente, marcadores)
            segmentation=v_laber4-1


        result=cv2.resize(segmentation,(OGN,OGM))    +1              
        return(result)
        