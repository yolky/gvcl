import sys
import torch
import numpy as np

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.c1=torch.nn.Conv2d(ncha,32,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(size,3, padding = 1)
#         s=s//2
        self.c2=torch.nn.Conv2d(32,32,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(s,3, padding = 1)
        s=s//2
        self.c3=torch.nn.Conv2d(32,64,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(s,3, padding = 1)
#         s=s//2
        self.c4=torch.nn.Conv2d(64,64,kernel_size=3, padding = 1)
        s=utils.compute_conv_output_size(s,3, padding = 1)
        s = s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(64*self.smid*self.smid,512)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(512,n))

        return

    def forward(self,x):
        h=self.relu(self.c1(x))
        h=self.drop1(self.maxpool(self.relu(self.c2(h))))
        h=self.relu(self.c3(h))
        
        h=self.drop2(self.maxpool(self.relu(self.c4(h))))
        
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y