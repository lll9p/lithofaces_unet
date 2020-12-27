import json
import os
import random

import h5py
import numpy as np
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
    dataset = None

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
        self.mode = mode
        self.dataset = None
        self.data_len = len(idx)
        if Dataset.dataset is None:
            if "KAGGLE_CONTAINER_NAME" in os.environ:
                h5file = h5py.File(
                    self.config.path,
                    "r",
                    libver="latest",
                    swmr=True)
                print("Detect in Kaggle, reading all data to memory.")
                Dataset.dataset = dict()
                Dataset.dataset["images"] = h5file["images"][()]
                Dataset.dataset["idx"] = h5file["idx"][()]
                Dataset.dataset["labels"] = h5file["labels"][()]
                if self.config.train_on == "masks":
                    Dataset.dataset["masks"] = h5file["masks"][()]
                elif self.config.train_on == "edges":
                    Dataset.dataset["masks"] = h5file["edges"][()]
                elif self.config.train_on == "distance":
                    Dataset.dataset["shape_distance"] = \
                        h5file["shape_distance"][()]
                    Dataset.dataset["neighbor_distance"] = \
                        h5file["neighbor_distance"][()]
            else:
                Dataset.dataset = h5py.File(
                    self.config.path,
                    "r",
                    libver="latest",
                    swmr=True)
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
        dataset_index = Dataset.full_idx.index(self.idx[index])
        if self.config.train_on == "masks":
            image, mask, idx, labels = (
                Dataset.dataset["images"][dataset_index],
                Dataset.dataset["masks"][dataset_index],
                Dataset.dataset["idx"][dataset_index],
                Dataset.dataset["labels"][dataset_index],
            )
            labels = json.loads(labels)
            mask_new = np.zeros_like(mask)
            for label, values in labels.items():
                if label in self.config.ignore_labels:
                    continue
                for value in values:
                    mask_new[mask == value] = self.label_map[label]
            mask = mask_new
            image, mask, _ = self.transforms(image, mask)
            return image, mask, idx
        elif self.config.train_on == "edges":
            image, mask, idx, labels = (
                Dataset.dataset["images"][dataset_index],
                Dataset.dataset["edges"][dataset_index],
                Dataset.dataset["idx"][dataset_index],
                Dataset.dataset["labels"][dataset_index],
            )
            image, mask, _ = self.transforms(image, mask)
            return image, mask, idx
        elif self.config.train_on == "distance":
            image, shape_distance, neighbor_distance, idx, labels = (
                Dataset.dataset["images"][dataset_index],
                Dataset.dataset["shape_distance"][dataset_index],
                Dataset.dataset["neighbor_distance"][dataset_index],
                Dataset.dataset["idx"][dataset_index],
                Dataset.dataset["labels"][dataset_index],
            )
            image, shape_distance, neighbor_distance = self.transforms(
                image, shape_distance, neighbor_distance)
            return image, shape_distance, neighbor_distance, idx

    def __len__(self):
        return self.data_len

    def transforms(self, image, mask, mask2=None):
        image = transforms.functional.to_pil_image(image)
        # mask = transforms.functional.to_pil_image(mask)
        # RandomCrop
        # i, j, h, w = transforms.RandomCrop.get_params(
        # image, output_size=(224, 224))
        # image = transforms.functional.crop(image, i, j, h, w)
        # mask = transforms.functional.crop(mask, i, j, h, w)
        # weight_map = weight_map[..., i : i + h, j : j + w]
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = np.fliplr(mask)
            if mask2 is not None:
                mask2 = np.fliplr(mask2)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = np.flipud(mask)
            if mask2 is not None:
                mask2 = np.flipud(mask2)
        # Random rotate90
        if random.random() > 0.5:
            image = transforms.functional.rotate(image, 90)
            mask = np.rot90(mask)
            if mask2 is not None:
                mask2 = np.rot90(mask2)
        # Random gaussian blur
        if bool(np.random.choice([True, False], p=[0.2, 0.8])):
            kernel_size = int(np.random.choice([11, 21, 31]))
            sigma = int(np.random.choice([10, 15, 20]))
            image = transforms.functional.gaussian_blur(
                image, kernel_size, sigma)
        if self.mode == "train":
            image = self.color_composed(image)
        image = normalize(image)
        return image, mask, mask2
