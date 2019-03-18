#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:22:28 2019

@author: santiago
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:00:59 2019

@author: santiago
"""

from Segment import segmentByClustering
from skimage import io 
import scipy.io as sio
import numpy as np
import os

os.system('rm -rf ownSegmentation')
os.system('mkdir ownSegmentation')

relevant_path = "BSDS500FastBench /BSR/BSDS500/data/images/test"
included_extensions = ['jpg']
fileName_img=[fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]


fileName_img=np.sort(fileName_img)


for idx in range(len(fileName_img)):
    vect=[]
    for k in range(2,202,10):
        img=io.imread('BSDS500FastBench /BSR/BSDS500/data/images/test/'+fileName_img[idx])
        seg=segmentByClustering(img,'rgb+xy','kmeans',k)
        vect.append(seg)
        
        FrameStack = np.empty((len(vect),), dtype=np.object)
        for jdx in range(len(vect)):
            FrameStack[jdx] = vect[jdx]
            
        sio.savemat(('ownSegmentation/'+(fileName_img[idx].replace('jpg','mat'))), {"segs":FrameStack})
  
