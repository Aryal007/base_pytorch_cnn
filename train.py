#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:59:35 2020

@author: mibook
"""

from utils.dataloader import DataLoader
from utils.frame import Framework
import torch 
import numpy as np
import random


random_seed = 7
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic=True

epochs = 25
train_losses = []
val_losses = []
train_accuracys = []
val_accuracys = []

trainloader = DataLoader('train', filepath="./cifar/")
trainloader = trainloader.get()
valloader = DataLoader('test', filepath="./cifar/")
valloader = valloader.get()

frame = Framework("SimpleNet")

for epoch in range(1, epochs+1):
    train_loss = frame.forward_pass("train", trainloader)
    train_accuracy = frame.get_metrics("train")
    val_loss = frame.forward_pass("val", valloader, update=1.0001)
    val_accuracy = frame.get_metrics("val")
    print(f"Iteration {epoch}, T_Loss: {train_loss:.5f}, V_Loss: {val_loss:.5f}, T_Acc: {train_accuracy:.4f}, V_Acc: {val_accuracy:.4f}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracys.append(train_accuracy)
    val_accuracys.append(val_accuracy)

'''
#frame.load("test")
import matplotlib.pyplot as plt
import numpy as np

plt.title("Epoch vs loss (1.0001 Update)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(np.arange(1,epochs+1), train_losses, 'go--', label="Train Loss")
plt.plot(np.arange(1,epochs+1), val_losses, 'bo--', label="Val Loss")
plt.grid()
plt.legend()
plt.show()

plt.title("Accuracy vs loss (1.0001 Update)")
plt.xlabel("Acc")
plt.ylabel("Loss")
plt.plot(np.arange(1,epochs+1), train_accuracys, 'go--', label="Train Acc")
plt.plot(np.arange(1,epochs+1), val_accuracys, 'bo--', label="Val Acc")
plt.grid()
plt.legend()
plt.show()
'''