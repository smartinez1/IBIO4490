#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:04:04 2019

@author: santiago
"""
from google_drive_downloader import GoogleDriveDownloader as gdd
import time as tm
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as imm
import cv2
import pydicom

if os.path.isdir('dataVision')==bool(0):
    gdd.download_file_from_google_drive(file_id='1DOBX07xvzdiKl0MUZJ_odLezu0dVy3Ou',
                                    dest_path='./dataVision/MR_data_batch1.zip',
                                    unzip=True)


start_time=tm.time()
os.system('rm -rf auxiliarFolder')
os.system('mkdir auxiliarFolder')

base=('dataVision/MR_data_batch1')
patients=[1,2,3,5,8,10,13,15,19,20]

index=1
totalImages=6
fs=5
for n in range(0,totalImages):
    idx=(np.random.randint(0,len(patients)-1))
    pat=str(patients[idx])

    # Selects a type pf MRI
    sel=np.random.randint(0,1)
    if sel==1:
        typeMR=('T1DUAL')
    else:
        typeMR=('T2SPIR')

    
    patiType=os.path.join(base,pat,typeMR)
    imgs=os.path.join(patiType,'DICOM_anon')
    ground=os.path.join(patiType,'Ground')
    names=os.listdir(imgs)
    names=np.sort(names)
    names_g=os.listdir(ground)
    names_g=np.sort(names_g)

    if sel==1:
        idx=np.random.randint(0,len(names)-1)
        f=names[idx]   
        if (idx==1 or idx==0):
                idx=0
        elif idx % 2==0:
                idx=idx/2
        else:
                idx=1+((idx-1)/2)            
                time=names[idx]
                ann=names_g[idx] 

    else:
        idx=np.random.randint(0,len(names)-1)
        f=names[idx]
        time=names[idx]
        ann=names_g[idx]



    OGimagePath=os.path.join(imgs,time)
    ann_path=os.path.join(ground,ann)
    annotation=imm.imread(ann_path)
    heigth, width=annotation.shape
    
    if (heigth!=256 or width!=256):
        annotation=cv2.resize(annotation,(256,256))


    ds = pydicom.read_file(OGimagePath)
    mapp = ds.pixel_array 
    
    heigth, width=mapp.shape
    if (heigth!=256 or width!=256):
        mapp=cv2.resize(mapp,(256,256))
    
    mapp=255*(mapp/np.amax(mapp))
    mapp=(mapp.astype(int))
            
    cv2.imwrite(f.replace('.dcm','.png'),mapp) 
    os.system('mv '+f.replace('.dcm','.png')+' auxiliarFolder')
    
    
    
    OGpngPath=('./auxiliarFolder/' + f.replace('.dcm','.png'))
    OGpng=Image.open(OGpngPath)



 
    plt.figure(1)
    plt.subplot(totalImages,2,index)
    plt.imshow(OGpng)
    plt.axis('off')
    plt.title(time,fontsize=fs)
    
    plt.subplot(totalImages,2,index+1)
    plt.imshow(annotation)
    plt.axis('off')
    plt.title(ann,fontsize=fs)
    
    index=index+2
os.system('rm -rf auxiliarFolder')    
print("--- %s seconds ---"%(tm.time()-start_time))    
plt.show()

