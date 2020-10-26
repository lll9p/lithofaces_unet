import os
import pathlib
from functools import partial

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import transforms


def get_datasets(path=pathlib.Path('/kaggle/input/lithofaces-dataset-generate/data_256/')):
    image_folders = sorted(
        path.iterdir(), key=lambda path_: path_.name.split("-")[0])
    train_folders, test_folders = train_test_split(
        image_folders, test_size=0.1, random_state=42)
    train_folders, val_folders = train_test_split(
        train_folders, test_size=0.1, random_state=42)
    # folder expand
    return dict(train=expand_idx(train_folders),
                val=expand_idx(val_folders),
                test=expand_idx(test_folders))


def expand_idx(folders):
    idxes = []
    for folder in folders:
        for crop_index in range(4):
            for aug_index in range(8):
                idxes.append(f"{folder.name}_{crop_index}_{aug_index}")
    return idxes


def crop(image, crop_index, aug_index):
    """ Method crop:0,1,2,3 expand:0-8"""
    def crop(image, index):
        crop_size = 224
        if index == "0":
            return image[0:crop_size, 0:crop_size, ...]
        if index == "1":
            return image[256-crop_size:, 256-crop_size:, ...]
        if index == "2":
            return image[256-crop_size:, 0:crop_size, ...]
        if index == "3":
            return image[0:crop_size, 256-crop_size:, ...]
    crop_image = crop(image, crop_index)
    aug_funcs = {
        "0": lambda crop_image: crop_image,  # 原图
        "1": partial(np.rot90, k=1),  # 90°
        "2": partial(np.rot90, k=2),  # 180°
        "3": partial(np.rot90, k=3),  # 270°
        "4": partial(np.flip, axis=0),  # 垂直翻转
        "5": partial(np.flip, axis=1),  # 水平翻转
        # 垂直翻转+90°
        "6": lambda crop_image: np.rot90(np.flip(crop_image, 0), 3),
        # 水平翻转+90°
        "7": lambda crop_image: np.rot90(np.flip(crop_image, 1), 3)
    }
    return aug_funcs[aug_index](crop_image)


def get_data(path=None, index='23-44_0_0'):
    image_path, crop_index, aug_index = index.split("_")
    base = str(path/f"{image_path}")
    image = cv2.imread(f"{base}/images/{image_path}.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop(image, crop_index, aug_index)
    masks = []
    minerals = ["Alite", "Blite", "C3A", "fCaO", "Pore"]
    for mask_name in minerals:
        temp = cv2.imread(f"{base}/masks/{mask_name}.png",
                          cv2.IMREAD_UNCHANGED)
        if temp is None:
            temp = np.zeros((256, 256))
        masks.append(crop(temp, crop_index, aug_index))
    for mask_name in minerals:
        temp = cv2.imread(f"{base}/masks/i{mask_name}.png",
                          cv2.IMREAD_UNCHANGED)
        if temp is None:
            temp = np.zeros((256, 256))
        masks.append(crop(temp, crop_index, aug_index))
    edges = cv2.imread(f"{base}/masks/edges.png", cv2.IMREAD_UNCHANGED)
    if edges is None:
        edges = np.zeros((256, 256))
    masks.append(crop(edges, crop_index, aug_index))
    return image, np.stack(masks)


class Dataset(data.Dataset):
    def __init__(self, dataset, root, mode='train', classes=list(range(11))):
        self.root = root
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
        image, masks = get_data(path=self.root, index=self.dataset[index])
        if self.transform is not None:
            image = self.transform(image)
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
        masks = masks.astype('float32')
#         mask = mask.transpose(2, 0, 1)
        return image, masks, self.dataset[index]

    def __len__(self):
        return len(self.dataset)
