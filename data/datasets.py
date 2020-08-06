import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import io
import cv2
from PIL import Image
import torchvision.transforms as transforms
try:
    import mc
except ImportError:
    pass

import pdb

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

def cv2_loader(path):
    img =cv2.imread(path)
    return img


class GivenSizeSampler(Sampler):
    '''
    Sampler with given total size
    '''
    def __init__(self, dataset, total_size=None, rand_seed=None, sequential=False, silent=False):
        self.rand_seed = rand_seed if rand_seed is not None else 0
        self.dataset = dataset
        self.epoch = 0
        self.sequential = sequential
        self.silent = silent
        self.total_size = total_size if total_size is not None else len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.sequential:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rand_seed)
            origin_indices = list(torch.randperm(len(self.dataset), generator=g))
        else:
            origin_indices = list(range(len(self.dataset)))
        indices = origin_indices[:]

        # add extra samples to meet self.total_size
        extra = self.total_size - len(origin_indices)
        if not self.silent:
            print('Origin Size: {}\tAligned Size: {}'.format(len(origin_indices), self.total_size))
        if extra < 0:
            indices = indices[:self.total_size]
        while extra > 0:
            intake = min(len(origin_indices), extra)
            indices += origin_indices[:intake]
            extra -= intake
        assert len(indices) == self.total_size, "{} vs {}".format(len(indices), self.total_size)

        return iter(indices)

    def __len__(self):
        return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch




def build_labeled_dataset(filelist, prefix):
    img_lst = []
    lb_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            n = x.split(' ')[0]
            lb = x.split(' ')[1]
            lb = int(lb)
            img_lst.append('{}/{}'.format(prefix,n))
            lb_lst.append(lb)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst

def build_unlabeled_dataset(filelist, prefix):
    img_lst = []
    with open(filelist) as f:
        for x in f.readlines():
            img_lst.append(os.path.join(prefix, x.strip().split(' ')[0]))
    return img_lst


class FileListLabeledDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst, self.lb_lst = build_labeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.num_class = max(self.lb_lst) + 1
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = cv2_loader(value_str)
            else:
                img = cv2_loader(fn)

            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img, lb = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb

class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None, memcached=False, memcached_client=''):
        self.img_lst = build_unlabeled_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform
        self.initialized = False
        self.memcached = memcached
        self.memcached_client = memcached_client

    def __len__(self):
        return self.num

    def __init_memcached(self):
        if not self.initialized:
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _read(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num)
        fn = self.img_lst[idx]
        try:
            #img = pil_loader(open(fn, 'rb').read())
            if self.memcached:
                value = mc.pyvector()
                self.mclient.Get(fn, value)
                value_str = mc.ConvertBuffer(value)
                img = pil_loader(value_str)
            else:
                img = pil_loader(open(fn, 'rb').read())
            return img
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        if self.memcached:
            self.__init_memcached()
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img
