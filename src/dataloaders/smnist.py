import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle

########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0.1):
    data={}
    taskcla=[]
    size=[1,28,28]

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.MNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.MNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    for t in range(5):
        data[t]={}
        data[t]['name']='mnist-{}-{}'.format(2*t, 2*t + 1)
        data[t]['ncla']=2
        data[t]['train']={'x': [],'y': []}
        data[t]['test']={'x': [],'y': []}
        
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)

        for image,target in loader:
            label=target.numpy()[0]
            data[label//2][s]['x'].append(image)
            data[label//2][s]['y'].append(label%2)
    
    mean=(0.1918,)
    std=(0.3483,)
    dat={}
    dat['train']=datasets.KMNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.KMNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    for t in range(5):
        
        data[t+5]={}
        data[t+5]['name']='kmnist-{}-{}'.format(2*t, 2*t + 1)
        data[t+5]['ncla']=2
        data[t+5]['train']={'x': [],'y': []}
        data[t+5]['test']={'x': [],'y': []}
        
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)

        for image,target in loader:
            label=target.numpy()[0]
            data[5 + label//2][s]['x'].append(image)
            data[5 + label//2][s]['y'].append(label%2)
                
    # "Unify" and save
    for n in range(10):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
            

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
        
        print("SLDKJFLSDJL")

    # Others
    n=0
    for t in range(10):
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    
    for t in range(10):
        print(data[t]['train']['y'])

    return data,taskcla,size

########################################################################################################################