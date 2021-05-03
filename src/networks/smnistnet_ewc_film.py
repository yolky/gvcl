import sys
import torch
import numpy as np

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha * size * size,256)
        self.fc2=torch.nn.Linear(256, 256)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))

        self.scale1=torch.nn.Embedding(len(self.taskcla),256)
        self.scale2=torch.nn.Embedding(len(self.taskcla),256)
        
        self.shift1=torch.nn.Embedding(len(self.taskcla),256)
        self.shift2=torch.nn.Embedding(len(self.taskcla),256)
    
        self.init_film()

        return
    
    def init_film(self):
        torch.nn.init.constant_(self.scale1.weight.data, 1.)
        torch.nn.init.constant_(self.scale2.weight.data, 1.)
        
        torch.nn.init.constant_(self.shift1.weight.data, 0.)
        torch.nn.init.constant_(self.shift2.weight.data, 0.)

    def forward(self,t, x):
        task_labels = t * torch.ones(x.shape[0]).long().to(x.get_device())
        h=x.view(x.size(0),-1)
        h=self.drop1(self.relu(self.scale1(task_labels) * self.fc1(h) + self.shift1(task_labels)))
                
        h=self.drop2(self.relu(self.scale2(task_labels) * self.fc2(h) + self.shift2(task_labels)))

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y