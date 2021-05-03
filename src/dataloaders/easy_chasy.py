import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from dataloaders.hasy_utils import get_hasy_datasets_even_split

def get(seed = 0, pc_valid = 0.10):
    data = {}
    taskcla = []
    size = [1,32,32]
    
    tasks = ['hasy_even101-133-84-123-126-170-105-74-180-44-111-96-131-31-106-42-198-13-52-73', 
    'hasy_even137-118-49-93-27-158-108-119-51-23-83-79-110-95-86-179-6-1-148-3', 
    'hasy_even134-63-33-189-43-41-38-146-15-135-155-21-151-104-18-103-87-24',
     'hasy_even112-36-19-159-85-168-80-107-60-55-8-47-124-150-164-115-120-173',
     'hasy_even130-184-10-186-102-128-191-82-70-90-185-48-4-172-99-129-64-161',
      'hasy_even0-50-154-181-56-77-91-5-37-113-145-62-25-196-169-68-2', 
     'hasy_even127-122-65-153-190-116-147-121-39-28-76-176-132-59-149-88', 
     'hasy_even58-183-109-7-160-178-144-92-69-72-188-35-40-94-117-11', 
     'hasy_even97-140-174-17-26-177-138-199-32-81-125-46-187', 
     'hasy_even34-67-114-175-192-157-16-53-143-22-162']
    heads = [20, 20, 18, 18, 18, 17, 16, 16, 13, 11]
    test_sizes = [182, 163, 173, 157, 175, 180, 174, 172, 183, 176]
    
    if not os.path.isdir('../dat/easy_chasy/'):
        os.makedirs('../dat/easy_chasy/')
        
    transform = transforms.Compose([transforms.Resize([32,32]), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        
    for t in range(10):
        data[t] = {}
        data[t]['name'] = 'easy_chasy_{}'.format(t)
        data[t]['ncla'] = heads[t]
        data[t]['train']={'x': [],'y': []}
        data[t]['test']={'x': [],'y': []}
        data[t]['valid']={'x': [],'y': []}
        
        numbers = list(map(int, tasks[t][len('hasy_even'):].split('-')))
        
        data_train, data_test, data_val = get_hasy_datasets_even_split('../dat/HASYv2', train_samples_per_class = 16, test_samples_per_class = test_sizes[t]-8, val_samples_per_class = 8, seed = seed, transform = transform, classes = numbers)
        
        for s, dataset in zip(['train', 'test', 'valid'], [data_train, data_test, data_val]):
            loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
            for image,target in loader:
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(target.numpy()[0])
                    
    for key in data.keys():
        for s in ['train','test', 'valid']:
            data[key][s]['x']=torch.stack(data[key][s]['x']).view(-1,size[0],size[1],size[2])
            data[key][s]['y']=torch.LongTensor(np.array(data[key][s]['y'],dtype=int)).view(-1)
            torch.save(data[key][s]['x'], os.path.join(os.path.expanduser('../dat/easy_chasy'),'data'+str(key)+s+'x.bin'))
            torch.save(data[key][s]['y'], os.path.join(os.path.expanduser('../dat/easy_chasy'),'data'+str(key)+s+'y.bin'))
        
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    
    return data,taskcla,size