import copy
import gc
import multiprocessing
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from functools import partial
from itertools import zip_longest

import cv2
import h5py
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.autonotebook import tqdm


def fix_edges(masks, pbar=None):
    masks_edge_fixed = dict()
    masks_edge_fixed_inner = dict()
    masks_edge_fixed_edge = dict()  # 3~4pixel的边界
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    if pbar is not None:
        pbar.reset()
    for image_id, masks_dict in masks.items():
        # 处理后的masks
        masks_edge_fixed[image_id] = dict()
        masks_edge_fixed_inner[image_id] = dict()
        masks_edge_fixed_edge[image_id] = dict()
        # 带有重叠边缘的mask
        mask_with_overlays = sum(masks_dict.values())
        # 重叠的边缘
        overlays = (mask_with_overlays > 1).astype(np.uint8)
        # 不带有重叠边缘的mask
        mask_without_overlays = (mask_with_overlays == 1).astype(np.uint8)
        if pbar is not None:
            pbar.total = len(masks_dict)
        for shape_id, shape in masks_dict.items():
            # 不重叠的图形
            shape_without_overlays = ((shape-overlays) == 1).astype(np.uint8)
            # 不重叠的mask（且不含图形）
            mask_without_shape_without_overlays = mask_without_overlays-shape_without_overlays
            # 图形边界
            shape_conv = sp.ndimage.convolve(
                shape_without_overlays, kernel, mode='reflect')
            shape_edge = np.bitwise_and(
                shape_conv < 8, shape_without_overlays > 0)
            # mask边界
            mask_conv = sp.ndimage.convolve(
                mask_without_shape_without_overlays, kernel, mode='reflect')
            mask_edge_without_shape = np.bitwise_and(
                mask_conv < 8, mask_without_shape_without_overlays > 0)
            # 接触点
            mask_edge_without_shape_conv = sp.ndimage.convolve(
                mask_edge_without_shape, kernel, mode='reflect') > 0
            shape_touched = np.bitwise_and(
                mask_edge_without_shape_conv, shape_edge)
            # 消除接触点（接触的地方向内1px）
            shape_edge_fixed = shape_without_overlays-shape_touched
            shape_inner_1 = (sp.ndimage.convolve(
                shape_edge_fixed, kernel, mode='reflect') == 9).astype(np.uint8)
            shape_inner_2 = (sp.ndimage.convolve(
                shape_inner_1, kernel, mode='reflect') == 9).astype(np.uint8)
            shape_fat_edge = shape_edge_fixed - shape_inner_2
            masks_edge_fixed[image_id][shape_id] = shape_edge_fixed
            masks_edge_fixed_inner[image_id][shape_id] = shape_inner_2
            masks_edge_fixed_edge[image_id][shape_id] = shape_fat_edge
            if pbar is not None:
                pbar.update(1)
    return masks_edge_fixed, masks_edge_fixed_inner, masks_edge_fixed_edge


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def split(shape, window=256):
    """
    split shape(y,x) to list of blocks
    """
    y_size, x_size = shape
    for y in range(0, y_size, window):
        for x in range(0, x_size, window):
            if y+window > y_size or x+window > x_size:
                block = None
            else:
                block = [y, y+window], [x, x+window]
            yield block


def save(path, image_id, contents, labels, stack_mask=False):
    """
    contents : [image,{'masks':{id:mask,...}, 'inners':inner, 'edges':edges}]
    """
    image = contents.pop(0)
    masks_ = contents[0]
    image_path = path/image_id
    os.makedirs(image_path/"images", exist_ok=True)
    os.makedirs(image_path/"masks", exist_ok=True)
    cv2.imwrite(f"{image_path}/images/{image_id}.png", image)
    if stack_mask is False:
        for masks_name, masks in masks_.items():
            flag = masks_name[0]  # m,i,e
            for id_, mask in masks.items():
                cv2.imwrite(f"{image_path}/masks/{flag}{id_}.png", mask)
    if stack_mask is True:
        edges = np.zeros((256, 256), dtype=np.uint8)
        for masks_name, masks in masks_.items():
            flag = masks_name[0]
            for label in labels:
                masks_temp = np.zeros((256, 256), dtype=np.uint8)
                for id_, mask in masks.items():
                    if not id_.startswith(label):
                        continue
                    if masks_name == 'edges':
                        edges = np.bitwise_or(edges, mask)
                    masks_temp = np.bitwise_or(masks_temp, mask)
                cv2.imwrite(
                    f"{image_path}/masks/{flag}{label}.png", masks_temp)
        cv2.imwrite(
            f"{image_path}/masks/edges.png", edges)


def process_original_dataset(image_node,
                             path=None,
                             path_256=None,
                             input_path="/kaggle/input/lithofaces",
                             translations=None):
    """
    处理一张图片节点，生成contour,并把图片及countor分割为256x256->256x256,512x512->256x256
    """
    pbar = None
    path = pathlib.Path(path)
    path_256 = pathlib.Path(path_256)
    labels = tuple(translations.values())
    # get image id
    image_id = image_node.attrib['id']
    # get image name
    image_name = image_node.attrib['name'].split("/")[1]
    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)
    masks_ = dict()
    label_count = dict()
    for polygon in image_node:
        label = polygon.attrib['label']
        label = translations[label]
        if label not in labels:
            continue
        label_count.setdefault(label, 0)
        label_count[label] += 1
        shape_points = np.array(
            [eval(_) for _ in polygon.attrib['points'].split(";")]).astype(np.int)
        shape = cv2.drawContours(np.zeros(image.shape[:2]),  # blank image
                                 [shape_points],  # contours
                                 -1,  # contour id
                                 True,  # contour color [255,255,255]
                                 -1  # contour thickness -1 means fill
                                 )
        masks_[f"{label}{label_count[label]}"] = shape.astype(np.uint8)
    # process masks touched
    masks, masks_inner, masks_edge = fix_edges({image_id: masks_}, pbar=pbar)
    contents = [image, dict(masks=masks[image_id], inners=masks_inner[image_id],
                            edges=masks_edge[image_id])]
    save(path=path, image_id=image_id, contents=contents,
         labels=labels, stack_mask=False)
    del contents
    blocks_256 = list(split(image.shape[:2], window=256))
    resize_index = len(blocks_256)
    blocks_512 = list(split(image.shape[:2], window=512))
    blocks = blocks_256+blocks_512
    for i in range(2):
        # add 2 random 384x384 crop
        y_ = np.random.randint(0, image.shape[0]-384)
        x_ = np.random.randint(0, image.shape[1]-384)
        blocks.append(([y_, y_+384], [x_, x_+384]))
    is_resize = False
    for index, block in enumerate(blocks):
        if block is None:
            continue
        if index >= resize_index:
            is_resize = True
        else:
            is_resize = False
        image_256_id = f"{image_id}-{index}"
        y, x = block
        image_256 = image[y[0]:y[1], x[0]:x[1], ...]
        masks_256_ = dict()
        if is_resize:
            image_256 = cv2.resize(image_256, (256, 256))
        for shape_id, shape in masks_.items():
            shape_256 = shape[y[0]:y[1], x[0]:x[1], ...]
            if is_resize:
                shape_256 = cv2.resize(shape_256, (256, 256))
            masks_256_[f"{shape_id}"] = shape_256
        masks_256, masks_inner_256, masks_edge_256 = fix_edges(
            {image_256_id: masks_256_}, pbar=pbar)
        contents = [image_256, dict(masks=masks_256[image_256_id],
                                    inners=masks_inner_256[image_256_id],
                                    edges=masks_edge_256[image_256_id])]
        save(path=path_256, image_id=image_256_id,
             contents=contents, labels=labels, stack_mask=True)
        del contents
        gc.collect()
    return True


def dataset_perform(image_ranges, translations={'A矿': 'Alite', 'B矿': 'Blite', 'C3A': 'C3A', '游离钙': 'fCaO', '孔洞': 'Pore'}):
    path = pathlib.Path("/kaggle/working/data")
    path_256 = pathlib.Path("/kaggle/working/data_256")
    input_path = '/kaggle/input/lithofaces'
    func = partial(process_original_dataset,
                   input_path=input_path, path=path, path_256=path_256,
                   translations=translations)
    tree = ET.parse("/kaggle/working/data/annotations.xml")
    root = tree.getroot()
    images = []
    for image_ in root.findall(f".//image"):
        if int(image_.attrib['id']) in image_ranges:
            images.append(image_)
    # Reduce from 27s/image to 14s/image
    CPU_NUM = multiprocessing.cpu_count()
    with multiprocessing.Pool(CPU_NUM) as pool:
        result = list(tqdm(pool.imap(func, images),
                           desc="Images", position=0, total=len(images)))


def get_datasets(path=None):
    image_folders = sorted(
        path.iterdir(), key=lambda path_: path_.name.split("-")[0])
    train_folders, test_folders = train_test_split(
        image_folders, test_size=0.1, random_state=42)
    train_folders, val_folders = train_test_split(
        train_folders, test_size=0.1, random_state=42)
    # folder expand
    return dict(train=shuffle(expand_idx(train_folders), random_state=42),
                val=shuffle(expand_idx(val_folders), random_state=42),
                test=shuffle(expand_idx(test_folders), random_state=42))


def expand_idx(folders):
    idxes = []
    for folder in folders:
        for crop_index in range(4):
            idxes.append(f"{folder.name}_{crop_index}")
    return idxes


def crop(image, crop_index):
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
    """aug_funcs = {
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
    }"""

    crop_image = crop(image, crop_index)
    return crop_image


def get_data(path=None, index='23-44_0', labels=None):
    image_path, crop_index = index.split("_")
    base = str(path/f"{image_path}")
    image = cv2.imread(f"{base}/images/{image_path}.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop(image, crop_index)
    masks = []
    for mask_name in labels:
        temp = cv2.imread(f"{base}/masks/{mask_name}.png",
                          cv2.IMREAD_UNCHANGED)
        if temp is None:
            temp = np.zeros((256, 256))
        masks.append(crop(temp, crop_index))
    return image, np.stack(masks)


#(len(train)*32,224,224, 3)


def dataset_file_init(path="lithofaces.h5",
                      datasets=None,
                      images_shape=None,
                      masks_shape=None,
                      dtype=np.dtype('uint8')):
    with h5py.File(path, "w") as file:
        for dataset_name, dataset in datasets.items():
            file.create_dataset(f"{dataset_name}/images", shape=tuple([len(dataset)])+images_shape,
                                dtype=dtype, compression="gzip", compression_opts=4, chunks=True)
            file.create_dataset(f"{dataset_name}/masks", shape=tuple([len(dataset)])+masks_shape,
                                dtype=dtype, compression="gzip", compression_opts=4, chunks=True)
            idxes = [s.encode('ascii') for s in dataset]
            file.create_dataset(f"{dataset_name}/idx", shape=(len(idxes),), data=idxes,
                                dtype='S10', compression="gzip", compression_opts=4, chunks=True)


def dataset_preprocess(datasets, dataset_path="lithofaces.h5", data_path=None, chunk_size=100, labels=None):
    dataset_file_init(path=dataset_path,
                      datasets=datasets,
                      images_shape=(224, 224, 3),
                      masks_shape=(len(labels), 224, 224),
                      dtype=np.dtype('uint8'))
    with h5py.File(dataset_path, "a") as file:
        for dataset_name, dataset in datasets.items():
            images = []
            masks = []
            chunks = tuple(grouper(dataset, chunk_size))
            for chunk_num, chunk in tqdm(enumerate(chunks), desc=dataset_name, total=len(chunks)):
                for index in chunk:
                    if index is None:
                        continue
                    _image, _mask = get_data(
                        path=data_path, index=index, labels=labels)
                    images.append(_image)
                    masks.append(_mask)
                chunk_len = len(images)
                _images = np.stack(images)
                file[f"{dataset_name}/images"][chunk_num *
                                               chunk_size:chunk_num*chunk_size+chunk_len, :, :, :] = _images
                _masks = np.stack(masks)
                file[f"{dataset_name}/masks"][chunk_num *
                                              chunk_size:chunk_num*chunk_size+chunk_len, :, :, :] = _masks
                images.clear()
                masks.clear()


def prepare_dataset_224(hdf5_file, path='/kaggle/working/data_256', labels=["mAlite", "mBlite", "mPore", "iAlite", "iBlite", "iPore", "edges"]):
    # labels就是读取masks的文件名
    path = pathlib.Path(path)
    datasets = get_datasets(path)
    dataset_preprocess(datasets, dataset_path=hdf5_file,
                       data_path=pathlib.Path(path), chunk_size=500,
                       labels=labels)
