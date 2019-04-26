#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:32:05 2019

@author: santiago
"""
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
import tqdm

def train(data_loader, model, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        model.optimizer.zero_grad()
        loss = model.Loss(output,target)   
        loss.backward()
        model.optimizer.step()
        loss_cum.append(loss.item())
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    
class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
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
    
    
    
if __name__=='__main__':   
    ngpu=1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
        
    model = Net()
    model.to(device)

    model.training_params()
    print_network(model, 'Conv network')    
