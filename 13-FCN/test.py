#!/usr/bin/env python
# coding: utf-8

# # Test FCN32s
# 
# ![image.png](imgs/1.png)

# In[ ]:


import os
import os.path as osp
import datetime
import shlex
import subprocess
import pickle
import pydicom
import cv2
from skimage import io
from google_drive_downloader import GoogleDriveDownloader as gdd
    
import pytz
import torch
import yaml

import warnings
warnings.filterwarnings('ignore')

os.system('mkdir testImages')


if os.path.isdir('CHAOS_Train_Sets')==False:
    gdd.download_file_from_google_drive(file_id='1N3hva6J05q5OgPKzCcr9xmEsP47j0u9W',
                                    dest_path='./CHAOS_Train_Sets/CHAOS_Train_Sets.zip',
                                    unzip=True)  


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}




# In[ ]:


cfg = configurations[1]


# In[ ]:


gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
cuda = torch.cuda.is_available()


# ## PascalVOC Dataset

# In[ ]:


import numpy as np

class_names = np.array([
        'background',
        'liver',
        'Right Kidney',
        'Left Kidney',
        'Spleen'
    ])


# ## FCN - Model

# In[ ]:


def getChaosDataset():
        base='CHAOS_Train_Sets/Train_Sets/MR'
        patients=os.listdir('CHAOS_Train_Sets/Train_Sets/MR')
        strucs=[]
        labels=[]
        liver=[55,70]
        Rkidney=[110,135]#,126]
        Lkidney=[175,200]#,189]
        spleen=[240,255]#,252]
        #BR=0
        for idx in range(len(patients)):
            
            pat=str(patients[idx])
            
            typeMR2=('T2SPIR')
            
            patiType=os.path.join(base,pat,typeMR2)
            imgs=os.path.join(patiType,'DICOM_anon')
            ground=os.path.join(patiType,'Ground')
            names=os.listdir(imgs)
            names=np.sort(names)
            names_g=os.listdir(ground)
            names_g=np.sort(names_g)
            
            struct=np.zeros((len(names),256,256))
            label=np.zeros((len(names),256,256))
            
            for jdx in range(len(names)):
                time=names[jdx]
                ann=names_g[jdx]
                
                OGimagePath=os.path.join(imgs,time)
                ann_path=os.path.join(ground,ann)
                annotation=io.imread(ann_path)
                            
                
                heigth, width=annotation.shape
                if (heigth!=256 or width!=256):
                    annotation=cv2.resize(annotation,(256,256))
                transformedAnnotation=np.zeros((256,256))
                for kdx in range(256):
                    for ldx in range(256):
                        if (annotation[kdx,ldx]>=liver[0] and annotation[kdx,ldx]<=liver[1]):
                            transformedAnnotation[kdx,ldx]=1
                        elif (annotation[kdx,ldx]>=Rkidney[0] and annotation[kdx,ldx]<=Rkidney[1]):
                            transformedAnnotation[kdx,ldx]=2
                        elif (annotation[kdx,ldx]>=Lkidney[0] and annotation[kdx,ldx]<=Lkidney[1]):
                            transformedAnnotation[kdx,ldx]=3     
                        elif (annotation[kdx,ldx]>=spleen[0] and annotation[kdx,ldx]<=spleen[1]):
                            transformedAnnotation[kdx,ldx]=4
                        else:
                            transformedAnnotation[kdx,ldx]=0
                
                heigth, width=transformedAnnotation.shape
                
                    
                    
                ds = pydicom.read_file(OGimagePath)
                mapp = ds.pixel_array 
                
                heigth, width=mapp.shape
                if (heigth!=256 or width!=256):
                    mapp=cv2.resize(mapp,(256,256))
                    
                mapp=(mapp)
                
                struct[jdx,:,:]=mapp.astype('int8')
                label[jdx,:,:]=transformedAnnotation
                
                strucs.append(struct)
                labels.append(label)
        with open('dataSet_network.pickle', 'wb') as handle:
                pickle.dump([strucs,labels], handle, protocol=pickle.HIGHEST_PROTOCOL)




try:
                  
    with open('dataSet_network.pickle','rb') as pickleFile:
        dataSet=pickle.load(pickleFile)
    strucs=dataSet[0]
    labels=dataSet[1]
    print('Dataset Found') 
    print('Variables created')   
except FileNotFoundError:    
    print('Dataset not found')
    print('Creating dataset...')
    getChaosDataset()
    print('Dataset created')
    with open('dataSet_network.pickle','rb') as pickleFile:
        dataSet=pickle.load(pickleFile)
    strucs=dataSet[0]
    labels=dataSet[1]
    print('Variables created')
    
    
val=int(np.round((len(strucs))/5))
sett='val'
numSlices=0
images=[]
annotations=[]
if sett=='train':
    for jdx in range(val,len(labels)):
        numSlices=numSlices+labels[jdx].shape[0]
        for kdx in range(labels[jdx].shape[0]):
            images.append(strucs[jdx][kdx])
            annotations.append(labels[jdx][kdx])
            
elif sett=='val':
    for jdx in range(0,val):
        numSlices=numSlices+labels[jdx].shape[0]
        for kdx in range(labels[jdx].shape[0]):
            images.append(strucs[jdx][kdx])
            annotations.append(labels[jdx][kdx])



import numpy as np
import torch.nn as nn

class organNet32s(nn.Module):

    def __init__(self, n_class=5):
        super(organNet32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

    def forward(self, x, debug = False):
        h = x
        if debug: print(h.data.shape)
        h = self.relu1_1(self.conv1_1(h))
        if debug: print(h.data.shape)
        h = self.relu1_2(self.conv1_2(h))
        if debug: print(h.data.shape)
        h = self.pool1(h)
        if debug: print(h.data.shape)

        h = self.relu2_1(self.conv2_1(h))
        if debug: print(h.data.shape)
        h = self.relu2_2(self.conv2_2(h))
        if debug: print(h.data.shape)
        h = self.pool2(h)
        if debug: print(h.data.shape)

        h = self.relu3_1(self.conv3_1(h))
        if debug: print(h.data.shape)
        h = self.relu3_2(self.conv3_2(h))
        if debug: print(h.data.shape)
        h = self.relu3_3(self.conv3_3(h))
        if debug: print(h.data.shape)
        h = self.pool3(h)
        if debug: print(h.data.shape)

        h = self.relu4_1(self.conv4_1(h))
        if debug: print(h.data.shape)
        h = self.relu4_2(self.conv4_2(h))
        if debug: print(h.data.shape)
        h = self.relu4_3(self.conv4_3(h))
        if debug: print(h.data.shape)
        h = self.pool4(h)
        if debug: print(h.data.shape)

        h = self.relu5_1(self.conv5_1(h))
        if debug: print(h.data.shape)
        h = self.relu5_2(self.conv5_2(h))
        if debug: print(h.data.shape)
        h = self.relu5_3(self.conv5_3(h))
        if debug: print(h.data.shape)
        h = self.pool5(h)
        if debug: print(h.data.shape)

        h = self.relu6(self.fc6(h))
        if debug: print(h.data.shape)
        h = self.drop6(h)
        if debug: print(h.data.shape)

        h = self.relu7(self.fc7(h))
        if debug: print(h.data.shape)
        h = self.drop7(h)
        if debug: print(h.data.shape)

        h = self.score_fr(h)
        if debug: print(h.data.shape)

        h = self.upscore(h)
        if debug: print(h.data.shape)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        if debug: print(h.data.shape)
            
        return h


# ## Loading model

# In[ ]:


model = organNet32s(n_class=5)
if cuda: model.to('cuda')
model.eval()


# In[ ]:


resume = 'logs/MODEL-organNet32s_CFG-001_MAX_ITERATION-100000_LR-1e-10_MOMENTUM-0.99_WEIGHT_DECAY-0.0005_INTERVAL_VALIDATE-4000_TIME-20190517-164035/checkpoint.pth.tar'
print('Loading checkpoint from: '+resume)
model.load_state_dict(torch.load(resume)['model_state_dict'])
#model.load_state_dict(torch.load(resume))


# ## Running model

# In[1]:


#import PIL.Image
import torch
import numpy as np

#mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

def fileimg2model(idx):
    img = images[idx]
#    lbl = annotations[idx]
    img = np.array(img, dtype=np.uint8)
    return transform(img)

def transform(image):
    image = image.astype(np.float64)
    img = np.zeros((image.shape[0],image.shape[1],3))   
    img[:,:,0]=image
    img[:,:,1]=image
    img[:,:,2]=image
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def untransform(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
#    lbl = lbl.numpy()
    return img



# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from torch.autograd import Variable

def imshow_label(label_show, alpha=None):
    import matplotlib
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0.0,0.0,0.0,1.0)
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.arange(0,len(class_names))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(label_show, cmap=cmap, norm=norm, alpha=alpha)
    if alpha is None:
        plt.title(str([class_names[i] for i in np.unique(label_show) if i!=0]))
        cbar = plt.colorbar(ticks=bounds)
        cbar.ax.set_yticklabels(class_names)

def run_fromfile(idx,name):
    img_torch = torch.unsqueeze(fileimg2model(idx), 0)
    if cuda: img_torch = img_torch.to('cuda')
    with torch.no_grad():
        plt.imshow((images[idx]))
        plt.savefig(('testImages/image_'+name+'.png'))
        #plt.show()

        score = model(img_torch)
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]        
        plt.imshow((images[idx]), alpha=.9)
        imshow_label(lbl_pred[0], alpha=0.5)
        plt.savefig(('testImages/label_asd_'+name+'.png'))
        #plt.show()      

        imshow_label(lbl_pred[0])
        plt.savefig(('testImages/label_'+name+'.png'))
        #plt.show()
    return lbl_pred

def run_simple(idx):
    img_torch = torch.unsqueeze(fileimg2model(idx), 0)
    if cuda: img_torch = img_torch.to('cuda')
    with torch.no_grad():
        OG=images[idx]

        score = model(img_torch)
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

        lbl = annotations[idx]        
        
    return OG, lbl_pred, lbl

# In[3]:


img_file = 100
pred=run_fromfile(img_file,'demo')
lbl=annotations[img_file]
plt.imshow(lbl)
plt.savefig('testImages/ActualLabel.png')
#
#
## In[4]:
#
img_file = 200
pred=run_fromfile(img_file,'demo_1')
lbl=annotations[img_file]
plt.imshow(lbl)
plt.savefig('testImages/ActualLabel_1.png')
#
#
## In[5]:
#
#
#img_file = 'imgs/demo2.JPG'
#run_fromfile(img_file,'demo2')
#
#
## In[6]:
#
#
#img_file = 'imgs/demo3.jpg'
#run_fromfile(img_file,'demo3')


# In[ ]:





