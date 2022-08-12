# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:33:01 2022

@author: ahmed
"""


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class convBlock(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1):
        super(convBlock, self).__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.convb = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                  stride=1,padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                  stride=1, padding=1, kernel_size=3, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
        )
        
    def forward(self, X):
        X = self.convb(X)
        return X
    

class Unet(nn.Module):
    
    def __init__(self, features, in_feature, out_feature):
        super(Unet,self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.maxpools = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convups = nn.Sequential()
        self.in_feature = in_feature
        self.features = features
        
        for feature in features:
            self.downs.append(convBlock(in_feature, feature))
            in_feature = feature
            
        for feature in reversed(features[1:]):
            self.convups.append(nn.ConvTranspose2d(in_channels=feature, out_channels=feature//2, 
                                                   kernel_size=2, stride=2))
            self.ups.append(convBlock(feature, feature//2))
            
        self.finallayer = nn.Conv2d(features[0], out_feature, kernel_size=1)
        
        
    def forward(self, OUT):
        
        Skips = []
        for down in self.downs[:-1]:
            OUT = down(OUT)
            Skips.append(OUT) 
            OUT = self.maxpools(OUT)
            #print(OUT.shape)
        OUT = self.downs[-1](OUT) 
        #print(OUT.shape)
        Skips_rev = Skips[::-1]
        s = 0
        for i in range(0,(len(self.ups)*2)):
            
            if i % 2 == 0:
                OUT =  self.convups[i//2](OUT)
                OUT = TF.resize(img=OUT, size=Skips_rev[i//2].shape[2:])
                C_OUT = torch.cat((Skips_rev[i//2],OUT),dim=1)
            else: 
                OUT = self.ups[s](C_OUT)
                s +=1
            #print(OUT.shape)
        return self.finallayer(OUT)
            
            
            
def test():
    some_random_image = torch.rand((2,3,180,180)) 
    UNETmodel = Unet([64, 128, 256, 512, 1024], 3,1)
    #print(UNETmodel)           
    preds = UNETmodel(some_random_image)   
    print(preds.shape)     
    print(some_random_image.shape)    
    if preds.shape == some_random_image.shape:
        print('Hurraaaay')        
   
test()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            