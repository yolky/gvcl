import sys
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.ntasks = len(self.taskcla)

        expand_factor = 1.117 #increases the number of parameters

        #init task columns subnets
        self.conv1=torch.nn.ModuleList()
        self.sizec1 = int(expand_factor*32/self.ntasks)
        
        self.conv2=torch.nn.ModuleList()
        self.V2scale=torch.nn.ModuleList()
        #for conv layers the dimensionality reduction in the adapters is performed by 1x1 convolutions
        self.V2x1=torch.nn.ModuleList()
        self.U2=torch.nn.ModuleList()
        self.sizec2 = int(expand_factor*32/self.ntasks)
        
        self.conv3=torch.nn.ModuleList()
        self.V3scale=torch.nn.ModuleList()
        self.V3x1=torch.nn.ModuleList()
        self.U3=torch.nn.ModuleList()
        self.sizec3 = int(expand_factor*64/self.ntasks)
        
        self.conv4=torch.nn.ModuleList()
        self.V4scale=torch.nn.ModuleList()
        self.V4x1=torch.nn.ModuleList()
        self.U4=torch.nn.ModuleList()
        self.sizec4 = int(expand_factor*64/self.ntasks)
        
        self.fc1=torch.nn.ModuleList()
        self.Vf1scale=torch.nn.ModuleList()
        self.Vf1=torch.nn.ModuleList()
        self.Uf1=torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor*512/self.ntasks)

        self.last=torch.nn.ModuleList()
        self.Vflscale=torch.nn.ModuleList()
        self.Vfl=torch.nn.ModuleList()
        self.Ufl=torch.nn.ModuleList()

        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        #declare task columns subnets
        for t,n in self.taskcla:
            self.c1=self.conv1.append(torch.nn.Conv2d(ncha,self.sizec1,kernel_size=3, padding = 1))
            s=utils.compute_conv_output_size(size,3, padding = 1)

            self.c2=self.conv2.append(torch.nn.Conv2d(self.sizec1,self.sizec2,kernel_size=3, padding = 1))
            s=utils.compute_conv_output_size(s,3, padding = 1)
            s=s//2
            self.c3=self.conv3.append(torch.nn.Conv2d(self.sizec2,self.sizec3,kernel_size=3, padding = 1))
            s=utils.compute_conv_output_size(s,3, padding = 1)
            
            self.c4=self.conv4.append(torch.nn.Conv2d(self.sizec3,self.sizec4,kernel_size=3, padding = 1))
            s=utils.compute_conv_output_size(s,3, padding = 1)
            s = s//2

            self.fc1.append(torch.nn.Linear(self.sizec4*s*s,self.sizefc1))
            self.last.append(torch.nn.Linear(self.sizefc1,n))

            if t>0:
                #lateral connections with previous columns
                self.V2scale.append(torch.nn.Embedding(1,t))
                self.V2x1.append(torch.nn.Conv2d(t*self.sizec1, self.sizec1, kernel_size=1, stride=1))
                self.U2.append(torch.nn.Conv2d(self.sizec1,self.sizec2,kernel_size = 3, padding = 1))

                self.V3scale.append(torch.nn.Embedding(1,t))
                self.V3x1.append(torch.nn.Conv2d(t*self.sizec2,self.sizec2, kernel_size=1, stride=1))
                self.U3.append(torch.nn.Conv2d(self.sizec2,self.sizec3,kernel_size=3, padding = 1))
                
                self.V4scale.append(torch.nn.Embedding(1,t))
                self.V4x1.append(torch.nn.Conv2d(t*self.sizec3,self.sizec3, kernel_size=1, stride=1))
                self.U4.append(torch.nn.Conv2d(self.sizec3,self.sizec4,kernel_size=3, padding = 1))

                self.Vf1scale.append(torch.nn.Embedding(1,t))
                self.Vf1.append(torch.nn.Linear(t*self.sizec4*s*s, self.sizec4*s*s))
                self.Uf1.append(torch.nn.Linear(self.sizec4*s*s,self.sizefc1))

                self.Vflscale.append(torch.nn.Embedding(1,t))
                self.Vfl.append(torch.nn.Linear(t*self.sizefc1,self.sizefc1))
                self.Ufl.append(torch.nn.Linear(self.sizefc1,n))

        return

    def forward(self,x,t):

        h=self.relu(self.conv1[t](x))
        if t>0: #compute activations for previous columns/tasks
            h_prev1 = []
            for j in range(t):
                h_prev1.append(self.relu(self.conv1[j](x)))

        h_pre = self.conv2[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            h_prev2 = [self.drop1(self.maxpool(self.relu(self.conv2[j](h_prev1[j])))) for j in range(t)]
            h_pre = h_pre + self.U2[t-1](self.relu(self.V2x1[t-1](torch.cat([self.V2scale[t-1].weight[0][j] * h_prev1[j] for j in range(t)],1))))
        h=self.drop1(self.maxpool(self.relu(h_pre)))

        h_pre = self.conv3[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            h_prev3 = [self.relu(self.conv3[j](h_prev2[j])) for j in range(t)]
            h_pre = h_pre + self.U3[t-1](self.relu(self.V3x1[t-1](torch.cat([self.V3scale[t-1].weight[0][j] * h_prev2[j] for j in range(t)],1))))
        h=self.relu(h_pre)
        
        h_pre = self.conv4[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            h_prev4 = [self.drop2(self.maxpool(self.relu(self.conv4[j](h_prev3[j])))).view(x.size(0),-1) for j in range(t)]
            h_pre = h_pre + self.U4[t-1](self.relu(self.V4x1[t-1](torch.cat([self.V4scale[t-1].weight[0][j] * h_prev3[j] for j in range(t)],1))))
        h=self.drop2(self.maxpool(self.relu(h_pre)))
        
        h=h.view(x.size(0),-1)

        h_pre = self.fc1[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            hf_prev1 = [self.drop2(self.relu(self.fc1[j](h_prev4[j]))) for j in range(t)]
            h_pre = h_pre + self.Uf1[t-1](self.relu(self.Vf1[t-1](torch.cat([self.Vf1scale[t-1].weight[0][j] * h_prev4[j] for j in range(t)],1))))
        h=self.drop2(self.relu(h_pre))

        y=[]
        for tid,i in self.taskcla:
            if t>0 and tid<t:
                h_pre = self.last[tid](hf_prev1[tid]) #current column/task
                if tid>0:
                    #sum laterals, no non-linearity for last layer
                    h_pre = h_pre + self.Ufl[tid-1](self.Vfl[tid-1](torch.cat([self.Vflscale[tid-1].weight[0][j] * hf_prev1[j] for j in range(tid)],1)))
                y.append(h_pre)
            else:
                y.append(self.last[tid](h))
        return y

    #train only the current column subnet
    def unfreeze_column(self,t):
        utils.set_req_grad(self.conv1[t],True)
        utils.set_req_grad(self.conv2[t],True)
        utils.set_req_grad(self.conv3[t],True)
        utils.set_req_grad(self.conv4[t],True)
        utils.set_req_grad(self.fc1[t],True)
        utils.set_req_grad(self.last[t],True)
        if t>0:
            utils.set_req_grad(self.V2scale[t-1],True)
            utils.set_req_grad(self.V2x1[t-1],True)
            utils.set_req_grad(self.U2[t-1],True)
            
            utils.set_req_grad(self.V3scale[t-1],True)
            utils.set_req_grad(self.V3x1[t-1],True)
            utils.set_req_grad(self.U3[t-1],True)
            
            utils.set_req_grad(self.V4scale[t-1],True)
            utils.set_req_grad(self.V4x1[t-1],True)
            utils.set_req_grad(self.U4[t-1],True)
            
            utils.set_req_grad(self.Vf1scale[t-1],True)
            utils.set_req_grad(self.Vf1[t-1],True)
            utils.set_req_grad(self.Uf1[t-1],True)
            
            utils.set_req_grad(self.Vflscale[t-1],True)
            utils.set_req_grad(self.Vfl[t-1],True)
            utils.set_req_grad(self.Ufl[t-1],True)

        #freeze other columns
        for i in range(self.ntasks):
            if i!=t:
                utils.set_req_grad(self.conv1[i],False)
                utils.set_req_grad(self.conv2[i],False)
                utils.set_req_grad(self.conv3[i],False)
                utils.set_req_grad(self.conv4[i],False)
                utils.set_req_grad(self.fc1[i],False)
                utils.set_req_grad(self.last[i],False)
                if i>0:
                    utils.set_req_grad(self.V2scale[i-1],False)
                    utils.set_req_grad(self.V2x1[i-1],False)
                    utils.set_req_grad(self.U2[i-1],False)
                    
                    utils.set_req_grad(self.V3scale[i-1],False)
                    utils.set_req_grad(self.V3x1[i-1],False)
                    utils.set_req_grad(self.U3[i-1],False)
                    
                    utils.set_req_grad(self.V4scale[t-1],False)
                    utils.set_req_grad(self.V4x1[t-1],False)
                    utils.set_req_grad(self.U4[t-1],False)
                    
                    utils.set_req_grad(self.Vf1scale[i-1],False)
                    utils.set_req_grad(self.Vf1[i-1],False)
                    utils.set_req_grad(self.Uf1[i-1],False)
                    
                    utils.set_req_grad(self.Vflscale[i-1],False)
                    utils.set_req_grad(self.Vfl[i-1],False)
                    utils.set_req_grad(self.Ufl[i-1],False)
        return

