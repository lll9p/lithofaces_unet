import os
import random
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


def get_data(dataset, index, labels=None):
    full_labels = ["mAlite", "mBlite",
                   "mPore", "iAlite", "iBlite", "iPore", "edges"]
    label_index = [full_labels.index(label) for label in labels]
    masks = dataset[1][index][label_index]
    masks[-1] = masks[-1]
    # masks selector
    return dataset[0][index], masks, dataset[2][index]


class Dataset(data.Dataset):
    def __init__(self, dataset, root, mode='train', classes=list(range(11)), labels=None):
        self.root = root
        self.labels = labels
        self.dataset = dataset[mode]
        self.mode = mode

    def transforms(self, image, masks):
        def normalize(image):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])(image)
        if self.mode == 'train':
            composed = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1),
                    transforms.ColorJitter(contrast=0.1),
                    transforms.ColorJitter(saturation=0.15),
                    transforms.ColorJitter(hue=0.1)
                ], p=0.1),
            ])
        else:
            composed = transforms.Compose([
                transforms.ToPILImage(),
            ])
        image = composed(image)
        masks = [transforms.functional.to_tensor(mask) for mask in masks]
        if random.random() > .5:
            image = transforms.functional.hflip(image)
            masks = tuple((transforms.functional.hflip(mask)
                           for mask in masks))
        if random.random() > .5:
            image = transforms.functional.vflip(image)
            masks = tuple((transforms.functional.vflip(mask)
                           for mask in masks))
        masks = torch.cat(masks)
        return normalize(image), masks

    def __getitem__(self, index):
        image, masks, idx = get_data(
            dataset=self.dataset, index=index, labels=self.labels)
        masks = masks.astype(np.float32)
        if self.transforms is not None:
            image, masks = self.transforms(image, masks)
        return image, masks, idx.decode()

    def __len__(self):
        return len(self.dataset[0])
