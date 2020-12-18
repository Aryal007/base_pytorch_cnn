#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:59:35 2020

@author: mibook
"""
import torch
from PIL import Image
import os
import numpy as np
import random

class Cifar10Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, data_size = 0, transforms = None):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir,x) for x in files]
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.transforms = transforms
        
        with open(os.path.dirname(os.path.dirname(data_dir))+"/labels.txt") as label_file:
            labels = label_file.read().split()
            label_mapping = dict(zip(labels, list(range(len(labels)))))
        
        self.label_mapping = label_mapping
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = self.preprocess(image)
        label_name = image_address[:-4].split("_")[-1]
        label = self.label_mapping[label_name]
        
        image = image.astype(np.float32)
        
        if self.transforms:
            image = self.transforms(image)

        return image, label
    
    def preprocess(self, image):
        image = np.array(image)
        cifar_mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1,1,-1)
        cifar_std  = np.array([0.2023, 0.1994, 0.2010]).reshape(1,1,-1)
        image = (image - cifar_mean) / cifar_std
        image = image.transpose(2,1,0)
        return image

class DataLoader():
    
    def __init__(self, data, filepath="./", batch_size = 64, data_size = 0, shuffle = True, num_workers=3):
        data_path = filepath+data+"/"
        
        dataset = Cifar10Dataset(data_dir = data_path, data_size = data_size, transforms=None)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    def get(self):
        return self.dataloader