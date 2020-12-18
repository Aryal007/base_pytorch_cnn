#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:28:40 2020

@author: mibook
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .models import SimpleNet
import os

class Framework():
    
    def __init__(self, network="SimpleNet"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #Check whether a GPU is present.
        if network == "SimpleNet":
           self.clf = SimpleNet()
        else:
            raise Exception("Network Architecture unavailable")
        self.clf.to(self.device)   #Put the network on GPU if present
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.clf.parameters(), lr=0.001, momentum=0.9)
        self.loss = None
        
    def train_batch(self, batch):
        self.optimizer.zero_grad()
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        y_hat = self.clf(X)
        self.loss = self.criterion(y_hat, y)
        self.loss.backward()
        self.optimizer.step()
        correct = torch.eq(torch.max(F.softmax(y_hat), dim=1)[1], y).view(-1)
        self.train_num_correct += torch.sum(correct).item()
        self.train_num_samples += correct.shape[0] 
        return self.loss.data.item()
    
    def eval_batch(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        y_hat = self.clf(X)
        y_hat = y_hat.to(self.device)
        self.loss = self.criterion(y_hat, y)
        correct = torch.eq(torch.max(F.softmax(y_hat), dim=1)[1], y).view(-1)
        self.val_num_correct += torch.sum(correct).item()
        self.val_num_samples += correct.shape[0] 
        return self.loss.data.item()
    
    def get_metrics(self, data):
        if data == "train":
            return self.train_num_correct/self.train_num_samples
        elif data == "val":
            return self.val_num_correct/self.val_num_samples
        else:
            raise Exception("Incorrect parameter for metrics")
    
    def forward_pass(self, name, dataloader):
        loss = 0
        if name == "train":
            self.train_num_correct = 0
            self.train_num_samples = 0
        else:
            self.val_num_correct = 0
            self.val_num_samples = 0
        for batch in dataloader:
            if name == "train":
                loss += self.train_batch(batch)
            elif name == "val":
                loss += self.eval_batch(batch)
            else:
                raise Exception("Incorrect forward pass")
        loss /= len(dataloader)
        return loss
    
    def save(self, name="test"):
        if not os.path.exists("./tmp"):
            os.makedirs(os.path.dirname("./tmp"))
        torch.save(self.clf.state_dict(), "./tmp/"+name)
        
    def load(self, name="test"):
        tmp = torch.load("./tmp/"+name)
        self.clf.load_state_dict(tmp)
