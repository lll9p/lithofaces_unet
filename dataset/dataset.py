import os
import pathlib
from functools import partial

import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import h5py


def get_datasets(file_path, modes=['train', 'val', 'test']):
    # Read datasets to memory
    print("Reading whole dataset to memory.")
    datasets = dict()
    with h5py.File(file_path, "r") as file:
        for mode in modes:
            datasets[f'{mode}'] = (file[f'{mode}/images'][()],
                                   file[f'{mode}/masks'][()],
                                   file[f'{mode}/idx'][()])
    return datasets


def get_data(dataset, index):
    return dataset[0][index], dataset[1][index], dataset[2][index]


class Dataset(data.Dataset):
    def __init__(self, dataset, root, mode='train', classes=list(range(11)), labels=None):
        self.root = root
        self.labels = labels
        self.dataset = dataset[mode]
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1),
                    transforms.ColorJitter(contrast=0.1),
                    transforms.ColorJitter(saturation=0.15),
                    transforms.ColorJitter(hue=0.1)
                ], p=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        image, masks, idx = get_data(dataset=self.dataset, index=index)
        if self.transform is not None:
            image = self.transform(image)
        return image, masks, idx

    def __len__(self):
        return len(self.dataset[0])
