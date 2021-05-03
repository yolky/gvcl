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
    
    tasks = ['hasy_even123-93-33-19-10-154-65-109-140-67-156-141-171-61-54-163-57-30', 
    'hasy_even170-158-43-85-102-56-190-160-174-114-194-89-12-139-14-195-29-193', 
    'hasy_even111-83-135-55-90-37-39-69-138-16-9-66-142-182-166-71',
     'hasy_even101-137-134-112-130-0-127-58-97-34-167-165-100-197-75'
     , 'hasy_even131-110-21-47-48-145-76-188-32-53-20-98',
      'hasy_even74-119-38-80-191-77-116-178-17-175-78',
       'hasy_even44-23-15-60-70-5-121-92-177-157-152',
        'hasy_even31-95-151-124-4-62-176-35-81-143-136',
         'hasy_even73-3-24-173-161-2-88-11-187-162-45', 
    'hasy_even180-51-146-107-82-91-147-144-26-192']
    heads = [18, 18, 16, 15, 12, 11, 11, 11, 11, 10]
    test_sizes = [155,178,166,181,172,175,161,160,171,176]
    
    if not os.path.isdir('../dat/hard_chasy/'):
        os.makedirs('../dat/hard_chasy/')
        
    transform = transforms.Compose([transforms.Resize([32,32]), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        
    for t in range(10):
        data[t] = {}
        data[t]['name'] = 'hard_chasy_{}'.format(t)
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
            torch.save(data[key][s]['x'], os.path.join(os.path.expanduser('../dat/hard_chasy'),'data'+str(key)+s+'x.bin'))
            torch.save(data[key][s]['y'], os.path.join(os.path.expanduser('../dat/hard_chasy'),'data'+str(key)+s+'y.bin'))
        
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    
    return data,taskcla,size