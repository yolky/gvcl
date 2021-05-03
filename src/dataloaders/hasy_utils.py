from PIL import Image, ImageDraw
import csv
import json
import os
import sys
from math import log
import shutil
import urllib.request
import tarfile
import time
from functools import partial
import torchvision.datasets
from torchvision import transforms
import subprocess
import torch 
import numpy as np
import urllib3

def load_csv(filepath):
    """Read a CSV file."""
    data = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    return data

def get_labels_sorted_by_samples(base_dir):
    base_dir = os.path.expanduser(base_dir)
    data = load_csv(base_dir + '/symbols.csv')
    for row in data:
        row['samples'] = int(row['training_samples']) + int(row['test_samples'])
    sorted_labels = [int(row['symbol_id']) for row in sorted(data, key = lambda x: -x['samples'])]
    lmao = sorted(data, key = lambda x: -x['samples'])
    for i in range(37):
        sample_counts = [lmao[10 * i + j]['samples'] for j in range(10) if 10 * i + j < len(lmao)]
    
    return sorted_labels

def get_sample_counts(base_dir):
    base_dir = os.path.expanduser(base_dir)
    data = load_csv(base_dir + '/symbols.csv')
    samples = []
    for row in data:
        samples.append(int(row['training_samples']) + int(row['test_samples']))
    
    return sorted(samples, reverse = True)

def load_image_path_set(filepath):
    """Read a CSV file."""
    path_set = set()
    labels = set()
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            path_set.add(row['path'].split('/')[-1])
            labels.add(int(row['symbol_id']))
    return path_set, sorted(labels)

def get_filenames_for_symbols(symbol_ids, path):
    symbol_id_to_filenames = {}
    for symbol_id in symbol_ids:
        symbol_id_to_filenames[symbol_id] = os.listdir(path + '/by_class/' + str(symbol_id))
    return symbol_id_to_filenames
        

def get_train_file_set(seed, train_samples_per_class, test_samples_per_class, val_samples_per_class, symbol_ids, path):
    symbol_id_to_filenames = get_filenames_for_symbols(symbol_ids, path)
    train_file_set = set()
    val_file_set = set()
    test_file_set = set()
    for symbol_id in symbol_id_to_filenames:
        np.random.seed(seed)
        subset = np.random.choice(symbol_id_to_filenames[symbol_id], train_samples_per_class + val_samples_per_class + test_samples_per_class, replace = False)
        train_file_set.update(subset[:train_samples_per_class])
        val_file_set.update(subset[train_samples_per_class:train_samples_per_class + val_samples_per_class])
        test_file_set.update(subset[train_samples_per_class + val_samples_per_class:])
    return train_file_set, test_file_set, val_file_set

def load_dataset(base_dir, replace = False):
    #first load csv
    #second reconstruct files
    base_dir = os.path.expanduser(base_dir)
    target_dir = os.path.expanduser(base_dir) + '/by_class/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    elif not replace:
        print("dataset already loaded")
        return
    else:
        shutil.rmtree(target_dir)
    labels = load_csv(base_dir + '/hasy-data-labels.csv')
    symbol_ids = set()
    
    for i, el, in enumerate(labels):
        if el['symbol_id'] not in symbol_ids:
            os.makedirs(target_dir + '/' + el['symbol_id'])
            symbol_ids.add(el['symbol_id'])
    print("copying")
    import multiprocessing as mp
    pool = mp.Pool(8)
    pool.map(partial(copy_image, base_dir, target_dir), labels)
    pool.close()

def is_valid_file_getter(file_set, valid_symbol_ids):
    def is_valid_file(path):
        path = path.replace('\\', '/')
        split = path.split('/')
        filename = split[-1]
        class_label = int(split[-2])
        return (filename in file_set) and ((valid_symbol_ids is None) or (class_label in valid_symbol_ids))
    return is_valid_file

def make_datasets(base_dir, fold = 0, transform = None, classes = None):
    base_dir = os.path.expanduser(base_dir)
    class_list = get_labels_sorted_by_samples(base_dir)
    train_files, _ = load_image_path_set(base_dir + '/classification-task/fold-{}/train.csv'.format(fold + 1))
    
    if classes is None:
        valid_symbol_ids = None
    else:
        symbol_ids = [class_list[i] for i in classes]
        valid_symbol_ids = set(symbol_ids)
        inverse_symbol_list = {k: v for v, k in enumerate(symbol_ids)}
    train_dataset = torchvision.datasets.ImageFolder(base_dir + '/by_class/', transform=transform, is_valid_file=is_valid_file_getter(train_files, valid_symbol_ids))

    test_files, _ = load_image_path_set(base_dir + '/classification-task/fold-{}/test.csv'.format(fold + 1))
    test_dataset = torchvision.datasets.ImageFolder(base_dir + '/by_class/', transform=transform, is_valid_file=is_valid_file_getter(test_files, valid_symbol_ids))

    if classes is not None:
        train_dataset.targets = torch.tensor([inverse_symbol_list[int(train_dataset.classes[target])] for target in train_dataset.targets])
        train_dataset.samples = [(sample[0], inverse_symbol_list[int(train_dataset.classes[sample[1]])]) for sample in train_dataset.samples]
        
        test_dataset.targets = torch.tensor([inverse_symbol_list[int(test_dataset.classes[target])] for target in test_dataset.targets])
        test_dataset.samples = [(sample[0], inverse_symbol_list[int(test_dataset.classes[sample[1]])]) for sample in test_dataset.samples]
    return train_dataset, test_dataset


def make_datasets_even_split(base_dir, seed, train_samples_per_class, test_samples_per_class, val_samples_per_class, transform = None, classes = None):
    base_dir = os.path.expanduser(base_dir)
    class_list = get_labels_sorted_by_samples(base_dir)
    
    
    if classes is None:
        valid_symbol_ids = None
    else:
        symbol_ids = [class_list[i] for i in classes]
        valid_symbol_ids = set(symbol_ids)
        inverse_symbol_list = {k: v for v, k in enumerate(symbol_ids)}

    train_file_set, test_file_set, val_file_set = get_train_file_set(seed, train_samples_per_class, test_samples_per_class, val_samples_per_class, valid_symbol_ids, base_dir)

    train_dataset = torchvision.datasets.ImageFolder(base_dir + '/by_class/', transform=transform, is_valid_file=is_valid_file_getter(train_file_set, valid_symbol_ids))
    test_dataset = torchvision.datasets.ImageFolder(base_dir + '/by_class/', transform=transform, is_valid_file=is_valid_file_getter(test_file_set, valid_symbol_ids))
    if len(val_file_set) > 0:
        val_dataset = torchvision.datasets.ImageFolder(base_dir + '/by_class/', transform=transform, is_valid_file=is_valid_file_getter(val_file_set, valid_symbol_ids))
    else:
        val_dataset = None

    if classes is not None:
        train_dataset.targets = torch.tensor([inverse_symbol_list[int(train_dataset.classes[target])] for target in train_dataset.targets])
        train_dataset.samples = [(sample[0], inverse_symbol_list[int(train_dataset.classes[sample[1]])]) for sample in train_dataset.samples]
        
        test_dataset.targets = torch.tensor([inverse_symbol_list[int(test_dataset.classes[target])] for target in test_dataset.targets])
        test_dataset.samples = [(sample[0], inverse_symbol_list[int(test_dataset.classes[sample[1]])]) for sample in test_dataset.samples]

        if len(val_file_set) > 0:
            val_dataset.targets = torch.tensor([inverse_symbol_list[int(val_dataset.classes[target])] for target in val_dataset.targets])
            val_dataset.samples = [(sample[0], inverse_symbol_list[int(val_dataset.classes[sample[1]])]) for sample in val_dataset.samples]
        else:
            val_dataset = None
    return train_dataset, test_dataset, val_dataset

def copy_image(base_dir, target_dir, el):
    file_name = el['path'].split('/')[-1]
    target_file_dir = target_dir + '/' + el['symbol_id'] + '/'
    if(os.path.isfile(base_dir + '/' + el['path'])):
        shutil.move(base_dir + '/' + el['path'], target_file_dir + file_name)

def download_and_extract_hasy(path, replace = False):
    path = os.path.expanduser(path)
    download_url = 'https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1'
    target_archive_name = path + "HASYv2.tar.bz2"
    
    if not replace and os.path.exists(path):
        return

    if os.path.exists(path):
        print(os.listdir(path + '/../'))
        shutil.rmtree(path)
    os.makedirs(path)

    print("downloading")
    # urllib.request.urlretrieve(download_url, target_archive_name, reporthook)
    

    http = urllib3.PoolManager()
    with open(target_archive_name, 'wb') as out:
        r = http.request('GET', download_url, preload_content=False)
        shutil.copyfileobj(r, out)

    print("downloaded")

    # subprocess.run(["tar", "-xf", target_archive_name, "--use-compress-prog=lbzip2", "--directory " + path])
    tar = tarfile.open(target_archive_name, 'r:bz2')
    tar.extractall(path)
    tar.close()

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    if(sys.stdout is not None):
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

def get_hasy_datasets(path, replace = False, fold = 0, transform = None, classes = None):
    download_and_extract_hasy(path, replace = replace)
    load_dataset(path, replace= replace)
    return make_datasets(path, fold, transform, classes = classes)

def get_hasy_datasets_even_split(path, train_samples_per_class = 64, test_samples_per_class = 64, val_samples_per_class = 64, replace = False, seed = 0, transform = None, classes = None):
    download_and_extract_hasy(path, replace = replace)
    load_dataset(path, replace= replace)
    return make_datasets_even_split(path, seed, train_samples_per_class, test_samples_per_class, val_samples_per_class, transform, classes = classes)


if __name__ == '__main__':
    download_and_extract_hasy("../dat/HASYv2")
    load_dataset("../dat/HASYv2", replace = True)
    # make_datasets("~/../notebooks/storage/datasets/HASYv2/")
    # get_labels_sorted_by_samples("~/../notebooks/storage/datasets/HASYv2/")