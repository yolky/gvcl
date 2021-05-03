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

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(32*self.smid*self.smid,100)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(100,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),16)
        self.ec2=torch.nn.Embedding(len(self.taskcla),32)
        self.efc1=torch.nn.Embedding(len(self.taskcla),100)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gfc1=masks
        # Gated
        h=self.maxpool(self.relu(self.c1(x)))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.drop2(self.maxpool(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gfc1=self.gate(s*self.efc1(t))
        return [gc1,gc2,gfc1]

    def get_view_for(self,n,masks):
        gc1,gc2,gfc1=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc2.data.view(-1,1,1).expand((self.ec2.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)

        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        return None
