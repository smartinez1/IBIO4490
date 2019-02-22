#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:17:53 2019
@author: santiago
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

os.system('rm -rf mergedNoPyramids.jpg')
os.system('rm -rf mergedWithPyramids.jpg')
os.system('rm -rf filteredImageLowPass.jpg')
os.system('rm -rf filteredImageHighPass.jpg')
os.system('rm -rf resultsAfterProcessing_road.jpg')
os.system('rm -rf resultsAfterProcessing_bridge.jpg')
os.system('rm -rf hybrid.jpg')


B=cv2.imread('imgs/roadRainy.jpeg')
A=cv2.imread('imgs/bridge.jpg')

A=np.fliplr(A)
Mb,Nb,cb=B.shape

B=B[350:,200:Nb-150]
A=A[0:,0:]


A=cv2.cvtColor(A,cv2.COLOR_BGR2RGB)
B=cv2.cvtColor(B,cv2.COLOR_BGR2RGB)

A=A.astype('float')
B=B.astype('float')

A=cv2.resize(A,(1024,512))
B=cv2.resize(B,(1024,512))

sz=3
sharpen1=np.zeros((sz,sz))
sharpen1[int(sz/2),int(sz/2)]=2

sharpen=sharpen1-np.ones((sz,sz))*(1/(sz**2))
#A=abs(cv2.filter2D(A,-1,sharpen))
B=abs(cv2.filter2D(B,-1,sharpen))

#--------------------------------
#Size and sigma of Gaussian kernel
def createKernel(sigma,k):
    
    # Declares empty array of kernel's size
    kernel=np.zeros((k,k))
    #Defines Gaussian function in 2 dimensions
    def gauss2d(x,y):
        return (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/sigma**2)
    # iterates over the array giving it the proper values
    kdx=0
    ldx=0
    for idx in range(int(-k/2),int(k/2)+1):
        ldx=0
        for jdx in range(int(-k/2),int(k/2)+1):
            kernel[kdx,ldx]=gauss2d(idx,jdx)
            ldx=ldx+1            
        kdx=kdx+1
           
            #Normalizes the kernel
    kernel=kernel/sum(sum(kernel))
    return kernel
#---------------------------------
kernel_1=createKernel(8,9)
kernel_2=createKernel(10,11)
#---------------------------------
filteredA=cv2.filter2D(A,-1,kernel_1)
filteredB=cv2.filter2D(B,-1,kernel_2)
filteredA=abs(A-filteredA)

hybrid=filteredA+filteredB
hybrid=255*(hybrid/np.amax(hybrid))

A=A.astype('uint8')
B=B.astype('uint8')
filteredA=filteredA.astype('uint8')
filteredB=filteredB.astype('uint8')
hybrid=hybrid.astype('uint8')

plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(B)
plt.axis('off')
plt.title('Original picture')
plt.subplot(2,2,2)
plt.imshow(filteredB)
plt.axis('off')
plt.title('Filtered picture')

plt.figure(1)
plt.subplot(2,2,3)
plt.imshow(A)
plt.axis('off')
plt.title('Original picture')
plt.subplot(2,2,4)
plt.imshow(filteredA)
plt.axis('off')
plt.title('Filtered picture')



plt.show()

plt.figure(2)
plt.imshow(hybrid)
plt.axis('off')
plt.title('Hybrid image')
plt.show()


#----------------------------
# Image Pyramids
Ga = A.copy()
gaussPyramid_A=[Ga]
Gb = B.copy()
gaussPyramid_B=[Gb]
pyramidLevel=6

for idx in range(pyramidLevel):
     Ga = cv2.pyrDown(Ga)
     gaussPyramid_A.append(Ga)
     Gb = cv2.pyrDown(Gb)
     gaussPyramid_B.append(Gb)
     
   
# generate Laplacian Pyramid for A & B
laplacePyramid_A=[gaussPyramid_A[pyramidLevel-1]]
laplacePyramid_B=[gaussPyramid_B[pyramidLevel-1]]
for idx in range(pyramidLevel-1,0,-1):
     GE = cv2.pyrUp(gaussPyramid_A[idx])
     L = cv2.subtract(gaussPyramid_A[idx-1],GE)
     laplacePyramid_A.append(L)     
     GE = cv2.pyrUp(gaussPyramid_B[idx])
     L = cv2.subtract(gaussPyramid_B[idx-1],GE)
     laplacePyramid_B.append(L)
     
#Now add left and right halves of images in each level
blendedList=[]     
for la,lb in zip(laplacePyramid_A,laplacePyramid_B):
     rows,cols,dpt = la.shape
     ls = np.hstack((la[:,0:int(cols/2)],lb[:,int(cols/2):]))
     blendedList.append(ls)     
     
blended=blendedList[0]
for idx in range(1,pyramidLevel):
     blended=cv2.pyrUp(blended)
     blended=cv2.add(blended,blendedList[idx])
     
real=np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))  
real=real.astype('uint8')
blended=blended.astype('uint8')
plt.figure()
plt.subplot(2,1,1)
plt.imshow(blended)
plt.axis('off')
plt.title('Blending using Pyramids') 
plt.subplot(2,1,2)
plt.imshow(real)
plt.axis('off')
plt.title('Blending without Pyramids')
plt.show()

real=cv2.cvtColor(real,cv2.COLOR_RGB2BGR)
blended=cv2.cvtColor(blended,cv2.COLOR_RGB2BGR)
filteredA=cv2.cvtColor(filteredA,cv2.COLOR_RGB2BGR)
filteredB=cv2.cvtColor(filteredB,cv2.COLOR_RGB2BGR)
hybrid=cv2.cvtColor(hybrid,cv2.COLOR_RGB2BGR)
B=cv2.cvtColor(B,cv2.COLOR_RGB2BGR)
A=cv2.cvtColor(A,cv2.COLOR_RGB2BGR)

for idx in range(pyramidLevel):
    aux=cv2.cvtColor(blendedList[idx],cv2.COLOR_RGB2BGR)
    #cv2.imshow(str(idx),aux)
    cv2.imwrite('level_'+str(idx+1)+'.jpg',aux)
    


cv2.imwrite('mergedNoPyramids.jpg',real) 
cv2.imwrite('mergedWithPyramids.jpg',blended)
cv2.imwrite('filteredImageLowPass.jpg',filteredA)  
cv2.imwrite('filteredImageHighPass.jpg',filteredB)
cv2.imwrite('hybrid.jpg',hybrid)
cv2.imwrite('resultsAfterProcessing_road.jpg',B)  
cv2.imwrite('resultsAfterProcessing_bridge.jpg',A)  











