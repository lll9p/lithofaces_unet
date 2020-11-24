import random

import h5py
import numpy as np
from torch.utils import data
from torchvision import transforms


def get_datasets(file_path, modes=['train', 'val']):
    # Read datasets to memory
    print("Reading whole dataset to memory.")
    datasets = dict()
    with h5py.File(file_path, "r") as file:
        for mode in modes:
            datasets[f'{mode}'] = (file[f'{mode}/images'][()],
                                   file[f'{mode}/masks'][()],
                                   file[f'{mode}/weight_maps'][()],
                                   file[f'{mode}/idx'][()])
    return datasets


def get_data(dataset, index):
    # masks selector
    (image, mask, weight_map, idx) = (
        dataset[0][index], dataset[1][index],
        dataset[2][index], dataset[3][index])
    return image, mask, weight_map, idx


def semantic2onehot(mask, labels, ignore_labels):
    def labels2num(labels, ignore_labels):
        labelnums = []
        for i, label in enumerate(labels):
            if label in ignore_labels:
                continue
            labelnums.append(i)
        return labelnums
    labelnums = labels2num(labels, ignore_labels)
    mask_ = np.zeros((len(labelnums),) + mask.shape, dtype=np.uint8)
    for index, label in enumerate(labelnums):
        mask_[index][mask == label] = 1
    return mask_


class Dataset(data.Dataset):
    def __init__(self, dataset, root, config=None, mode='train'):
        self.config = config
        self.root = root
        self.labels = config.labels
        self.dataset = dataset[mode]
        self.mode = mode
        self.composed = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1),
                transforms.ColorJitter(contrast=0.1),
                transforms.ColorJitter(saturation=0.15),
                transforms.ColorJitter(hue=0.1)
            ], p=0.2),
        ])

    def transforms(self, image, mask, weight_map):
        def normalize(image):
            # Calculate from whole lithofaces data
            return transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(
                     mean=[0.280313914506407, 0.41555059997248583,
                           0.3112942716287795],
                     std=[0.16130980680117304, 0.19598465956271507,
                          0.14531163979659875]), ])(image)
        image = transforms.functional.to_pil_image(image)
        mask = transforms.functional.to_pil_image(mask)
        # RandomCrop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        weight_map = weight_map[..., i:i + h, j:j + w]
        if random.random() > .5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
            weight_map = np.fliplr(weight_map)
        if random.random() > .5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
            weight_map = np.flipud(weight_map)
        # Random rotate90
        if random.random() > .5:
            image = transforms.functional.rotate(image, 90)
            mask = transforms.functional.rotate(mask, 90)
            weight_map = np.rot90(weight_map)
        image = self.composed(image)
        mask = np.array(mask)
        image = normalize(image)
        mask = mask.astype(np.float32)
        # weight_map = torch.from_numpy(weight_map.copy()).float()
        return image, mask, weight_map.copy()

    def __getitem__(self, index):
        image, mask, weight_map, idx = get_data(
            dataset=self.dataset, index=index)
        if self.transforms is not None:
            image, mask, weight_map = self.transforms(image, mask, weight_map)
        return image, mask, weight_map, idx.decode()

    def __len__(self):
        return len(self.dataset[0])
