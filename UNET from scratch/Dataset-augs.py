# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:38:06 2022

@author: ahmed
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import albumentations as A
import os 
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2



IMGDIR = 'C://Users//ahmed//Desktop//Water Marked//Original//'
IMAGE_NAMES = os.listdir(IMGDIR)

IMAGE = []
for names in IMAGE_NAMES:
    IMAGE.append(os.path.join(IMGDIR, names))


image = cv2.imread(IMAGE[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image)
plt.imshow(image)

MSKDIR = 'C:/Users/ahmed/Desktop/Water Marked/Masked/'
MSK_NAMES = os.listdir(MSKDIR)
MSK = []
for i,_ in enumerate(MSK_NAMES):
    MSK.append(MSKDIR + MSK_NAMES[i])
    
msk = cv2.imread(MSK[0],0)
# msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
# msk = msk[:,:,1]
plt.imshow(msk)

# IMAGE Parameters
HEIGHT = 300
WIDTH = 300
# HYPERparameters


#Augs if needed

train_aug = A.Compose([
    A.Resize(HEIGHT, WIDTH),
    A.Rotate(),
    A.HorizontalFlip(p=0.1)
    ])

validation_aug = A.Compose([
    A.Resize(HEIGHT, WIDTH)
    ])

# aug_image = train_aug(image=image)


transformed = train_aug(image=image, mask=msk)
transformed_image = transformed['image']
transformed_mask = transformed['mask']
plt.imshow(transformed_image)
plt.imshow(transformed_mask)




class UNETDataset(Dataset):
    
    def __init__(self, images, masks, augmentations = None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        self.images[index]
        self.masks[index]
        
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        msk = cv2.imread(self.masks[index],0)
        # msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        
        
        if self.augmentations:
            
            datawithaugs = self.augmentations(image=image, mask=msk)
            image = datawithaugs['image']
            msk = datawithaugs['mask'] 
            image = torch.from_numpy(image).permute(2,0,1)/255
            msk = torch.from_numpy(msk).unsqueeze(0)/255

        return image, msk




train_data1 = UNETDataset(IMAGE, MSK, train_aug)

exampim, exampmsk = train_data1[1]
plt.imshow(exampim.permute(1,2,0).numpy())
plt.imshow(exampmsk.squeeze(0).numpy())












