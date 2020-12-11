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
    def __init__(self, root, config=None, mode="train"):
        self.config = config
        self.root = root
        self.labels = config.labels
        self.mode = mode
        self.dataset = None
        with h5py.File(self.config.path, "r") as file:
            self.data_len = len(file[f"{mode}/images"])
        if "KAGGLE_CONTAINER_NAME" in os.environ:
            h5file = h5py.File(self.config.path, "r", libver="latest", swmr=True)[
                self.mode
            ]
            print("Detect in Kaggle, reading all data to memory.")
            self.dataset = dict()
            self.dataset["images"] = h5file["images"][()]
            self.dataset["masks"] = h5file["masks"][()]
            self.dataset["weight_maps"] = h5file["weight_maps"][()]
            self.dataset["idx"] = h5file["idx"][()]
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
            self.dataset = h5py.File(self.config.path, "r", libver="latest", swmr=True)[
                self.mode
            ]
        image, mask, weight_map, idx = (
            self.dataset["images"][index],
            self.dataset["masks"][index],
            self.dataset["weight_maps"][index],
            self.dataset["idx"][index],
        )
        if self.transforms is not None:
            image, mask, weight_map = self.transforms(image, mask, weight_map)
        return image, mask, weight_map, idx.decode()

    def __len__(self):
        return self.data_len

    def transforms(self, image, mask, weight_map):
        image = transforms.functional.to_pil_image(image)
        mask = transforms.functional.to_pil_image(mask)
        # RandomCrop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(224, 224))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        weight_map = weight_map[..., i : i + h, j : j + w]
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            weight_map = np.fliplr(weight_map)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
            weight_map = np.flipud(weight_map)
        # Random rotate90
        if random.random() > 0.5:
            image = transforms.functional.rotate(image, 90)
            mask = transforms.functional.rotate(mask, 90)
            weight_map = np.rot90(weight_map)
        # Random gaussian blur
        if bool(np.random.choice([True, False], p=[0.2, 0.8])):
            kernel_size = int(np.random.choice([11, 21, 31]))
            sigma = int(np.random.choice([10, 15, 20]))
            image = transforms.functional.gaussian_blur(image, kernel_size, sigma)
        if self.mode == "train":
            image = self.color_composed(image)
        mask = np.array(mask)
        image = normalize(image)
        return image, mask.astype(np.int64), weight_map.copy()

