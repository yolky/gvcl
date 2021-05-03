import sys
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))
        
        self.scale1=torch.nn.Embedding(len(self.taskcla),64)
        self.scale2=torch.nn.Embedding(len(self.taskcla),128)
        self.scale3=torch.nn.Embedding(len(self.taskcla),256)
        self.scale4=torch.nn.Embedding(len(self.taskcla),2048)
        self.scale5=torch.nn.Embedding(len(self.taskcla),2048)
        
        self.shift1=torch.nn.Embedding(len(self.taskcla),64)
        self.shift2=torch.nn.Embedding(len(self.taskcla),128)
        self.shift3=torch.nn.Embedding(len(self.taskcla),256)
        self.shift4=torch.nn.Embedding(len(self.taskcla),2048)
        self.shift5=torch.nn.Embedding(len(self.taskcla),2048)
        
        self.init_film();
        
        return
    
    def init_film(self):
        torch.nn.init.constant_(self.scale1.weight.data, 1.)
        torch.nn.init.constant_(self.scale2.weight.data, 1.)
        torch.nn.init.constant_(self.scale3.weight.data, 1.)
        torch.nn.init.constant_(self.scale4.weight.data, 1.)
        torch.nn.init.constant_(self.scale5.weight.data, 1.)
        
        torch.nn.init.constant_(self.shift1.weight.data, 0.)
        torch.nn.init.constant_(self.shift2.weight.data, 0.)
        torch.nn.init.constant_(self.shift3.weight.data, 0.)
        torch.nn.init.constant_(self.shift4.weight.data, 0.)
        torch.nn.init.constant_(self.shift5.weight.data, 0.)
        
        
    def forward(self,t, x):
        task_labels = t * torch.ones(x.shape[0]).long().to(x.get_device())
        h=self.maxpool(self.drop1(self.relu(self.scale1(task_labels)[:, :, None, None] * self.conv1(x) + self.shift1(task_labels)[:, :, None, None])))
        h=self.maxpool(self.drop1(self.relu(self.scale2(task_labels)[:, :, None, None] * self.conv2(h) + self.shift2(task_labels)[:, :, None, None])))
        h=self.maxpool(self.drop2(self.relu(self.scale3(task_labels)[:, :, None, None] * self.conv3(h) + self.shift3(task_labels)[:, :, None, None])))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.scale4(task_labels) * self.fc1(h) + self.shift4(task_labels)))
        h=self.drop2(self.relu(self.scale4(task_labels) * self.fc2(h) + self.shift4(task_labels)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y
