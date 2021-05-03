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

        return

    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=self.drop1(self.relu(self.fc1(h)))
                
        h=self.drop2(self.relu(self.fc2(h)))

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y, h