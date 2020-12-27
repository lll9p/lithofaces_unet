import json
import os
import random

import h5py
import numpy as np
from lithofaces_unet import get_neighbor_distance, get_shape_distance
from sklearn.model_selection import KFold
from torch.utils import data
from torchvision import transforms

# def semantic2onehot(mask, labels, ignore_labels):
#     def labels2num(labels, ignore_labels):
#         labelnums = []
#         for i, label in enumerate(labels):
#             if label in ignore_labels:
#                 continue
#             labelnums.append(i)
#         return labelnums

#     labelnums = labels2num(labels, ignore_labels)
#     mask_ = np.zeros((len(labelnums),) + mask.shape, dtype=np.uint8)
#     for index, label in enumerate(labelnums):
#         mask_[index][mask == label] = 1
#     return mask_


def normalize(image):
    # Calculate from whole lithofaces data
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.280313914506407,
                    0.41555059997248583,
                    0.3112942716287795,
                ],
                std=[
                    0.16130980680117304,
                    0.19598465956271507,
                    0.14531163979659875,
                ],
            ),
        ]
    )(image)


class Dataset(data.Dataset):
    full_idx = None

    def __init__(self, root, config=None, mode="train", idx=None):
        self.config = config
        self.root = root
        self.idx = idx
        labels = config.labels[:]
        ignore_labels = config.ignore_labels[:]
        for i_label in ignore_labels:
            if i_label in labels:
                labels.pop(labels.index(i_label))
        self.label_map = {label: i for i, label in enumerate(labels, start=1)}
        if config.train_on != 'edges':
            self.masks = "masks"
        else:
            self.masks = "edges"
        self.mode = mode
        self.dataset = None
        self.data_len = len(idx)
        if "KAGGLE_CONTAINER_NAME" in os.environ:
            h5file = h5py.File(
                self.config.path,
                "r",
                libver="latest",
                swmr=True)
            print("Detect in Kaggle, reading all data to memory.")
            self.dataset = dict()
            self.dataset["images"] = h5file["images"][()]
            self.dataset[self.masks] = h5file[self.masks][()]
            self.dataset["idx"] = h5file["idx"][()]
            self.dataset["labels"] = h5file["labels"][()]
        self.color_composed = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=0.1),
                        transforms.ColorJitter(contrast=0.1),
                        transforms.ColorJitter(saturation=0.15),
                        transforms.ColorJitter(hue=0.1),
                    ],
                    p=0.2,
                ),
            ]
        )

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(
                self.config.path,
                "r",
                libver="latest",
                swmr=True)
        dataset_index = Dataset.full_idx.index(self.idx[index])
        image, mask, idx, labels = (
            self.dataset["images"][dataset_index],
            self.dataset[self.masks][dataset_index],
            self.dataset["idx"][dataset_index],
            self.dataset["labels"][dataset_index],
        )
        if self.config.train_on == "masks":
            labels = json.loads(labels)
            mask_new = np.zeros_like(mask)
            for label, values in labels.items():
                if label in self.config.ignore_labels:
                    continue
                for value in values:
                    mask_new[mask == value] = self.label_map[label]
            mask = mask_new
        image, mask = self.transforms(image, mask)
        if self.config.train_on == "distance":
            shape_distance = get_shape_distance(mask)
            neighbor_distance = get_neighbor_distance(mask)
            return image, shape_distance, neighbor_distance, idx
        else:
            return image, mask, idx

    def __len__(self):
        return self.data_len

    def transforms(self, image, mask):
        image = transforms.functional.to_pil_image(image)
        mask = transforms.functional.to_pil_image(mask)
        # RandomCrop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        # Random rotate90
        if random.random() > 0.5:
            image = transforms.functional.rotate(image, 90)
            mask = transforms.functional.rotate(mask, 90)
        # Random gaussian blur
        if bool(np.random.choice([True, False], p=[0.2, 0.8])):
            kernel_size = int(np.random.choice([11, 21, 31]))
            sigma = int(np.random.choice([10, 15, 20]))
            image = transforms.functional.gaussian_blur(
                image, kernel_size, sigma)
        if self.mode == "train":
            image = self.color_composed(image)
        mask = np.array(mask)
        image = normalize(image)
        return image, mask.astype(np.int64)
