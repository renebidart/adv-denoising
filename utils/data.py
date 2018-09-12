import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from PIL import Image


class FilesDFImageDataset(Dataset):
    def __init__(self, files_df, transforms=None, path_colname='path', adv_path_colname=None, return_loc=False):
        """
        files_df: Pandas Dataframe containing the class and path of an image
        transforms: result of transforms.Compose()
        return_loc: return location as well as the image and class
        path_colname: Name of colum containing locations
        """
        self.files = files_df
        self.transforms = transforms
        self.path_colname = path_colname
        self.adv_path_colname = adv_path_colname
        self.return_loc = return_loc


    def __getitem__(self, index):
        img = Image.open(self.files[self.path_colname].iloc[index]).convert('RGB') # incase of greyscale
        label =  self.files['class'].iloc[index]
        if self.transforms is not None:
            img = self.transforms(img)

        if self.adv_path_colname and not self.return_loc:
            adv_img = Image.open(self.files[self.adv_path_colname].iloc[index]).convert('RGB') # incase of greyscale
            if self.transforms is not None:
                adv_img = self.transforms(adv_img)
            return img, adv_img, label
        elif self.adv_path_colname and self.return_loc:
            loc = (self.files[self.path_colname].iloc[index], self.files[self.adv_path_colname].iloc[index])
            return img, adv_img, label, loc
        elif not self.adv_path_colname and not self.return_loc:
            return img, label
        elif not self.adv_path_colname and self.return_loc:
            loc = self.files[self.path_colname].iloc[index]
            return img, label, loc

    def __len__(self):
        return len(self.files)


def make_generators_DF_cifar(files_df, batch_size, num_workers, size=32, 
                          path_colname='path', adv_path_colname=None, return_loc=False):
    """
    files_df: Dict containing train and val Pandas Dataframes
    Uses standard cifar augmentation and nomalization.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(int(size), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
    }
    datasets = {}
    dataloaders = {}

    datasets = {x: FilesDFImageDataset(files_df[x], data_transforms[x], path_colname=path_colname, 
                                        adv_path_colname=adv_path_colname, return_loc=return_loc)
                                        for x in list(data_transforms.keys())}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                                                    for x in list(data_transforms.keys())}
    return dataloaders

def make_gen_std_cifar(PATH, batch_size, num_workers):
    """ Make standard pytorch cifiar generators"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]),
    }
    datasets = {}
    dataloaders = {}

    datasets['train'] = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True, transform=data_transforms['train'])
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)

    datasets['val'] = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=data_transforms['val'])
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloaders
