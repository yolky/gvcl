import sys
import torch
import numpy as np

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.c1=torch.nn.Conv2d(ncha,16,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(size,3, padding = 1)
        s=s//2
        self.c2=torch.nn.Conv2d(16,32,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(s,3, padding = 1)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.0)
        self.drop2=torch.nn.Dropout(0.0)
        self.fc1=torch.nn.Linear(32*self.smid*self.smid,100)
        self.last=torch.nn.ModuleList()
        
        self.scale1=torch.nn.Embedding(len(self.taskcla),16)
        self.scale2=torch.nn.Embedding(len(self.taskcla),32)
        self.scale3=torch.nn.Embedding(len(self.taskcla),100)
        
        self.shift1=torch.nn.Embedding(len(self.taskcla),16)
        self.shift2=torch.nn.Embedding(len(self.taskcla),32)
        self.shift3=torch.nn.Embedding(len(self.taskcla),100)
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(100,n))
            
        self.init_film()

        return
    
    def init_film(self):
        torch.nn.init.constant_(self.scale1.weight.data, 1.)
        torch.nn.init.constant_(self.scale2.weight.data, 1.)
        torch.nn.init.constant_(self.scale3.weight.data, 1.)
        
        torch.nn.init.constant_(self.shift1.weight.data, 0.)
        torch.nn.init.constant_(self.shift2.weight.data, 0.)
        torch.nn.init.constant_(self.shift3.weight.data, 0.)

    def forward(self,t, x):
        task_labels = t * torch.ones(x.shape[0]).long().to(x.get_device())
        h=self.maxpool(self.relu(self.scale1(task_labels)[:, :, None, None] * self.c1(x) + self.shift1(task_labels)[:, :, None, None]))
        h=self.drop2(self.maxpool(self.scale2(task_labels)[:, :, None, None] * self.c2(h) + self.shift2(task_labels)[:, :, None, None]))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.scale3(task_labels) * self.fc1(h) + self.shift3(task_labels)))
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y
