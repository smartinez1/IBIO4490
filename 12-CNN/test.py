#!/home/afromero/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:09:05 2019

@author: santiago
"""

from __future__ import print_function, division
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils   
import torch.nn as nn
import tqdm
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


def showAtributes(image, labels):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.title('Example')
    leg=''
    if labels[0]==1:
        leg=leg +'Eyeglasses, '
    if labels[1]==1:
        leg=leg +'Bangs, '
    if labels[2]==1:
        leg=leg +'Black Hair, '
    if labels[3]==1:
        leg=leg +'Blond Hair, '
    if labels[4]==1:
        leg=leg +'Brown Hair, '
    if labels[5]==1:
        leg=leg +'Gray Hair, '
    if labels[6]==1:
        leg=leg +'Male, '
    if labels[7]==1:
        leg=leg +'Pale Skin, '
    if labels[8]==1:
        leg=leg +'Smiling, '
    if labels[9]==1:
        leg=leg +'Young, '
    
    plt.axis('off')
    #plt.text(0.5,0.5,leg)
    print(leg)


class CelebADataset(Dataset):
    
    def __init__(self, csv_file, root_dir,csv_file_part,sett,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            csv_file_part (string): Path to the csv file with partitions.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sett=sett
        # Partition (1=train,2=test,3=val)
        self.partition=pd.read_csv(csv_file_part).partition
        self.imgDir=root_dir
        labels=pd.read_csv(csv_file)
 
        idx=0
        jdx=162770
        kdx=182637
        ldx=202598+1
        #self.idx=idx
        self.jdx=jdx
        self.kdx=kdx
        #self.ldx=ldx
        #images ID
        imgs=(labels.image_id[idx:jdx])
        #Labels
        lab_1=(labels.Eyeglasses[idx:jdx])
        lab_2=(labels.Bangs[idx:jdx])
        lab_3=(labels.Black_Hair[idx:jdx])
        lab_4=(labels.Blond_Hair[idx:jdx])
        lab_5=(labels.Brown_Hair[idx:jdx])
        lab_6=(labels.Gray_Hair[idx:jdx])
        lab_7=(labels.Male[idx:jdx])
        lab_8=(labels.Pale_Skin[idx:jdx])
        lab_9=(labels.Smiling[idx:jdx])
        lab_10=(labels.Young[idx:jdx])
          
        #images ID
        imgs_test=(labels.image_id[jdx:kdx])
        #Labels
        lab_1_test=(labels.Eyeglasses[jdx:kdx])
        lab_2_test=(labels.Bangs[jdx:kdx])
        lab_3_test=(labels.Black_Hair[jdx:kdx])
        lab_4_test=(labels.Blond_Hair[jdx:kdx])
        lab_5_test=(labels.Brown_Hair[jdx:kdx])
        lab_6_test=(labels.Gray_Hair[jdx:kdx])
        lab_7_test=(labels.Male[jdx:kdx])
        lab_8_test=(labels.Pale_Skin[jdx:kdx])
        lab_9_test=(labels.Smiling[jdx:kdx])
        lab_10_test=(labels.Young[jdx:kdx])
          
        #images ID
        imgs_val=(labels.image_id[kdx:ldx])
        #Labels
        lab_1_val=(labels.Eyeglasses[kdx:ldx])
        lab_2_val=(labels.Bangs[kdx:ldx])
        lab_3_val=(labels.Black_Hair[kdx:ldx])
        lab_4_val=(labels.Blond_Hair[kdx:ldx])
        lab_5_val=(labels.Brown_Hair[kdx:ldx])
        lab_6_val=(labels.Gray_Hair[kdx:ldx])
        lab_7_val=(labels.Male[kdx:ldx])
        lab_8_val=(labels.Pale_Skin[kdx:ldx])
        lab_9_val=(labels.Smiling[kdx:ldx])
        lab_10_val=(labels.Young[kdx:ldx])

        #Assign Values Train       
        self.imgs=imgs
        self.lab_1=lab_1
        self.lab_2=lab_2
        self.lab_3=lab_3
        self.lab_4=lab_4
        self.lab_5=lab_5
        self.lab_6=lab_6
        self.lab_7=lab_7
        self.lab_8=lab_8
        self.lab_9=lab_9
        self.lab_10=lab_10
        #Assign Values Test       
        self.imgs_test=imgs_test
        self.lab_1_test=lab_1_test
        self.lab_2_test=lab_2_test
        self.lab_3_test=lab_3_test
        self.lab_4_test=lab_4_test
        self.lab_5_test=lab_5_test
        self.lab_6_test=lab_6_test
        self.lab_7_test=lab_7_test
        self.lab_8_test=lab_8_test
        self.lab_9_test=lab_9_test
        self.lab_10_test=lab_10_test
        #Assign Values Val
        self.imgs_val=imgs_val
        self.lab_1_val=lab_1_val
        self.lab_2_val=lab_2_val
        self.lab_3_val=lab_3_val
        self.lab_4_val=lab_4_val
        self.lab_5_val=lab_5_val
        self.lab_6_val=lab_6_val
        self.lab_7_val=lab_7_val
        self.lab_8_val=lab_8_val
        self.lab_9_val=lab_9_val
        self.lab_10_val=lab_10_val        
        # Transform
        self.transform=transform
    def __len__(self):
        if self.sett=='train':
            return len(self.imgs)
        elif self.sett=='test':
            return len(self.imgs_val)
        elif self.sett=='val':
            return len(self.imgs_test)
        
    def __getitem__(self, idx):
        if self.sett=='train':
            
            #Train
            img_name = os.path.join(self.imgDir,self.imgs[idx])
            image = io.imread(img_name)
            labs= [self.lab_1[idx],self.lab_2[idx],self.lab_3[idx],
                   self.lab_4[idx],self.lab_5[idx],self.lab_6[idx],
                   self.lab_7[idx],self.lab_8[idx],self.lab_9[idx],
                   self.lab_10[idx]]
            labs=np.asarray(labs)
        elif self.sett=='val':
            idx=idx+self.jdx
            #Test
            img_name = os.path.join(self.imgDir,self.imgs_test[idx])
            image= io.imread(img_name)
            labs= [self.lab_1_test[idx],self.lab_2_test[idx],self.lab_3_test[idx],
                        self.lab_4_test[idx],self.lab_5_test[idx],self.lab_6_test[idx],
                        self.lab_7_test[idx],self.lab_8_test[idx],self.lab_9_test[idx],
                        self.lab_10_test[idx]]
            labs=np.asarray(labs)
        elif self.sett=='test':
            idx=idx+self.kdx
            #Val
            img_name = os.path.join(self.imgDir,self.imgs_val[idx])
            image = io.imread(img_name)
            labs= [self.lab_1_val[idx],self.lab_2_val[idx],self.lab_3_val[idx],
                   self.lab_4_val[idx],self.lab_5_val[idx],self.lab_6_val[idx],
                   self.lab_7_val[idx],self.lab_8_val[idx],self.lab_9_val[idx],
                   self.lab_10_val[idx]]
            labs=np.asarray(labs)
        
        
        
        sample = {'image': image, 'labels': labs}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels= sample['image'], sample['labels']
        img = transform.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'labels': labels}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labs= sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = (np.round(image*255))
        image = image.astype(int)
           
        return {'image': torch.FloatTensor(image),
                'labels': torch.FloatTensor(labs)
                }
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #layer with 64 2d convolutional filter of size 3x3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) #Channels input: 1, c output: 63, filter of size 3
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc = nn.Linear(32, 10)    
    
    def forward(self, x, verbose=False):
        if verbose: "Output Layer by layer"
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #Perform a Maximum pooling operation over the nonlinear responses of the convolutional layer
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)#Try to control overfit on the network, by randomly excluding 25% of neurons on the last #layer during each iteration
        if verbose: print(x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        if verbose: print(x.size())
        x = F.dropout(x, 0.25, training=self.training)
        if verbose: print(x.size())
        #ipdb.set_trace()
        x = x.view(-1, 32)
        if verbose: print(x.size())
        x = self.fc(x)
        if verbose: print(x.size())
        return x

    def training_params(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        self.Loss = nn.BCEWithLogitsLoss()
  

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params)) 


def test(data_loader, model, epoch):
    model.eval()
    loss_cum = []
#    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        Data= next(iter(train_loader))
        data=Data['image']
        target=Data['labels']
        data = data.to(device).requires_grad_(False)
        target = target.to(device).requires_grad_(False)

        output = model(data)
        loss = model.Loss(output,target)   
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        #Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss Test: %0.3f "%(np.array(loss_cum).mean()))#, float(Acc*100)/len(data_loader.dataset)))
    
    
#---------------------------------------------------------#
#---------------------------------------------------------#
#---------------------------------------------------------#
        
if __name__=='__main__':   
    
    if os.path.isdir('celeba-dataset')==False:
        gdd.download_file_from_google_drive(file_id='1ybgNi6RGDKjem13N66ur1umXfR1RAiDH',
                                            dest_path='./celeba-dataset.zip',
                                            unzip=True)
        os.system('mkdir celeba-dataset')
        os.system('unzip img_align_celeba.zip')
        os.system('rm -rf img_align_celeba.zip')
        os.system('rm -rf celeba-dataset.zip ')
        os.system('mv img_align_celeba celeba-dataset')
        os.system('mv list_attr_celeba.csv celeba-dataset')
        os.system('mv list_bbox_celeba.csv celeba-dataset')
        os.system('mv list_eval_partition.csv celeba-dataset')
        os.system('mv list_landmarks_align_celeba.csv celeba-dataset')




    csv_file='celeba-dataset/list_attr_celeba.csv'
    root_dir='celeba-dataset/img_align_celeba'
    csv_file_part='celeba-dataset/list_eval_partition.csv'
    transform_train = transforms.Compose([Rescale(224),ToTensor()])
    #
    transformed_dataset_train = CelebADataset(csv_file=csv_file,
                                           root_dir=root_dir,
                                           csv_file_part=csv_file_part,sett='train',
                                           transform=transform_train)
                
    transformed_dataset_test = CelebADataset(csv_file=csv_file,
                                           root_dir=root_dir,
                                           csv_file_part=csv_file_part,sett='test',
                                           transform=transform_train)
                
    transformed_dataset_val = CelebADataset(csv_file=csv_file,
                                           root_dir=root_dir,
                                           csv_file_part=csv_file_part,sett='val',
                                           transform=transform_train)
                 
    
    
    
    plot=False
    if plot:
        fig = plt.figure()
        for i in range(len(transformed_dataset_val)):
            sample = transformed_dataset_val[i]
            
            print(i, sample['image'].shape, sample['labels'].shape)
            
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.axis('off') 
            showAtributes(**sample)
            
            if i == 3:
                plt.show()
                break
    

#    for i in range(len(transformed_dataset_val)):
#        sample = transformed_dataset_val[i]
#
#        print(i, sample['image'].size(), sample['labels'].size())
#
#        if i == 3:
#            break
##############################################################################
            
    epochs=10
    TEST=False
    workers=2
    batchS=64
    train_loader = DataLoader(transformed_dataset_train, batch_size=batchS,
                        shuffle=False, num_workers=workers)
    val_loader = DataLoader(transformed_dataset_val, batch_size=batchS,
                        shuffle=False, num_workers=workers)  
    test_loader = DataLoader(transformed_dataset_test, batch_size=batchS,
                        shuffle=False, num_workers=workers)
    
    model =torch.load('model.pt')
    model.eval()
    model.to(device)

    model.training_params()
    print_network(model, 'Conv network')  
    
    #Exploring model
    Data= next(iter(train_loader))
    img=Data['image']
    _ = model(img.to(device).requires_grad_(False))

    for epoch in range(epochs):
        test(test_loader, model, epoch)
    


                  

