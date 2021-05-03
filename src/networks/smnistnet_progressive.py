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

        self.fc1=torch.nn.ModuleList()
        self.Vf1scale=torch.nn.ModuleList()
        self.Vf1=torch.nn.ModuleList()
        self.Uf1=torch.nn.ModuleList()
        self.sizefc1 = int(expand_factor*256/self.ntasks)
        self.fc2=torch.nn.ModuleList()
        self.Vf2scale=torch.nn.ModuleList()
        self.Vf2=torch.nn.ModuleList()
        self.Uf2=torch.nn.ModuleList()
        self.sizefc2 = int(expand_factor*256/self.ntasks)
        self.last=torch.nn.ModuleList()
        self.Vflscale=torch.nn.ModuleList()
        self.Vfl=torch.nn.ModuleList()
        self.Ufl=torch.nn.ModuleList()

        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        #declare task columns subnets
        for t,n in self.taskcla:
            self.fc1.append(torch.nn.Linear(ncha*size*size,self.sizefc1))
            self.fc2.append(torch.nn.Linear(self.sizefc1,self.sizefc2))
            self.last.append(torch.nn.Linear(self.sizefc2,n))

            if t>0:
                self.Vf2scale.append(torch.nn.Embedding(1,t))
                self.Vf2.append(torch.nn.Linear(t*self.sizefc1,self.sizefc1))
                self.Uf2.append(torch.nn.Linear(self.sizefc1,self.sizefc2))

                self.Vflscale.append(torch.nn.Embedding(1,t))
                self.Vfl.append(torch.nn.Linear(t*self.sizefc2,self.sizefc2))
                self.Ufl.append(torch.nn.Linear(self.sizefc2,n))

        return

    def forward(self,x,t):
        x = x.view(x.size(0), -1)
        h=self.drop1(self.relu(self.fc1[t](x)))
        if t>0: #compute activations for previous columns/tasks
            hf_prev1 = []
            for j in range(t):
                hf_prev1.append(self.drop1(self.relu(self.fc1[j](x))))

        h_pre = self.fc2[t](h) #current column/task
        if t>0: #compute activations for previous columns/tasks & sum laterals
            hf_prev2 = [self.drop2(self.relu(self.fc2[j](hf_prev1[j]))) for j in range(t)]
            h_pre = h_pre + self.Uf2[t-1](self.relu(self.Vf2[t-1](torch.cat([self.Vf2scale[t-1].weight[0][j] * hf_prev1[j] for j in range(t)],1))))
        h=self.drop2(self.relu(h_pre))

        y=[]
        for tid,i in self.taskcla:
            if t>0 and tid<t:
                h_pre = self.last[tid](hf_prev2[tid]) #current column/task
                if tid>0:
                    #sum laterals, no non-linearity for last layer
                    h_pre = h_pre + self.Ufl[tid-1](self.Vfl[tid-1](torch.cat([self.Vflscale[tid-1].weight[0][j] * hf_prev2[j] for j in range(tid)],1)))
                y.append(h_pre)
            else:
                y.append(self.last[tid](h))
        return y

    #train only the current column subnet
    def unfreeze_column(self,t):
        utils.set_req_grad(self.fc1[t],True)
        utils.set_req_grad(self.fc2[t],True)
        utils.set_req_grad(self.last[t],True)
        if t>0:
            utils.set_req_grad(self.Vf2scale[t-1],True)
            utils.set_req_grad(self.Vf2[t-1],True)
            utils.set_req_grad(self.Uf2[t-1],True)
            utils.set_req_grad(self.Vflscale[t-1],True)
            utils.set_req_grad(self.Vfl[t-1],True)
            utils.set_req_grad(self.Ufl[t-1],True)

        #freeze other columns
        for i in range(self.ntasks):
            if i!=t:
                utils.set_req_grad(self.fc1[i],False)
                utils.set_req_grad(self.fc2[i],False)
                utils.set_req_grad(self.last[i],False)
                if i>0:
                    utils.set_req_grad(self.Vf2scale[i-1],False)
                    utils.set_req_grad(self.Vf2[i-1],False)
                    utils.set_req_grad(self.Uf2[i-1],False)
                    utils.set_req_grad(self.Vflscale[i-1],False)
                    utils.set_req_grad(self.Vfl[i-1],False)
                    utils.set_req_grad(self.Ufl[i-1],False)
        return

