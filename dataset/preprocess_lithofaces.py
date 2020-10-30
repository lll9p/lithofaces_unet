import copy
import multiprocessing
import os
import pathlib
import shutil
import xml.etree.ElementTree as ET
from functools import parial
from itertools import zip_longest

import cv2
import numpy as np
import scipy as sp
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


def prepare_dir(path):
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path)


def save_shape(masks, path):
    for index, contour in masks.items():
        cv2.imwrite(str((path/index).with_suffix(".png")), contour)


def split(shape, window=256):
    """
    split shape(y,x) to list of blocks
    """
    y_size, x_size = shape
    for y in range(0, y_size, window):
        for x in range(0, x_size, window):
            yield [y, y+window], [x, x+window]


def mask_256(masks, masks_, block, minerals):
    def _mask_(label, mineral, contour):
        if label.startswith(mineral):
            masks_[mineral] = np.bitwise_or(masks_[mineral], contour)
    y, x = block
    for label, contour in masks.items():
        contour_256 = contour[y[0]:y[1], x[0]:x[1]]
        for mineral in minerals:
            _mask_(label, mineral, contour_256)

# images_group = list(grouper(images,CPU_NUM))
# 256 11 classes
# Contours:Alite Blite C3A fCaO Pore
# Inners: Alite Blite C3A fCaO Pore
# Edges
# data
#    |
#    images
#    masks
#        |
#        Alite Blite C3A fCaO Pore iAlite iBlite iC3A ifCaO iPore edges


def process_original_dataset(image_node, minerals, input_path, translation, path, path_256, image_path):
    pbar = None
    translations = {x: y for x, y in zip(translation, minerals)}
    image_id = image_node.attrib['id']
    image_name = image_node.attrib['name'].split("/")[1]
    prepare_dir(path/image_id/"images")
    prepare_dir(path/image_id/"masks")
    prepare_dir(path/image_id/"inners")
    prepare_dir(path/image_id/"edges")
    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)
    masks = dict()
    label_count = dict()
    for polygon in image_node:
        label = polygon.attrib['label']
        label = translations[label]
        label_count.setdefault(label, 0)
        label_count[label] += 1
        contour_ = np.array(
            [eval(_) for _ in polygon.attrib['points'].split(";")]).astype(np.int)
        contour = cv2.drawContours(np.zeros(image.shape[:2]),  # blank image
                                   [contour_],  # contours
                                   -1,  # contour id
                                   True,  # contour color [255,255,255]
                                   -1  # contour thickness -1 means fill
                                   )
        masks[f"{label}{label_count[label]}"] = contour.astype(np.uint8)

    masks, masks_inner, masks_edge = fix_edges({image_id: masks}, pbar=pbar)

    cv2.imwrite(
        str((path/image_id/"images"/image_id).with_suffix(".png")), image)
    save_shape(masks[image_id], path/image_id/"masks")
    save_shape(masks_inner[image_id], path/image_id/"inners")
    save_shape(masks_edge[image_id], path/image_id/"edges")
    blocks = list(split(image.shape[:2]))
    masks = masks[image_id]
    inners = masks_inner[image_id]
    edges = masks_edge[image_id]
    for index, block in enumerate(blocks):
        image_name = f"{image_id}-{index}"
        y, x = block
        image_256 = image[y[0]:y[1], x[0]:x[1], :]
        prepare_dir(path_256/image_name/"images")
        cv2.imwrite(str((path_256/image_name/"images" /
                         image_name).with_suffix(".png")), image_256)
        # "minerals = ["Alite","Blite","C3A","fCaO","Pore"]"
        prepare_dir(path_256/image_name/"masks")
        masks_ = {mineral: np.zeros((256, 256), np.uint8)
                  for mineral in minerals}
        mask_256(masks, masks_, block, minerals)
        inners_ = {mineral: np.zeros((256, 256), np.uint8)
                   for mineral in minerals}
        mask_256(inners, inners_, block, minerals)
        edges_ = {mineral: np.zeros((256, 256), np.uint8)
                  for mineral in minerals}
        mask_256(edges, edges_, block, minerals)
        for k, v in masks_.items():
            cv2.imwrite(
                str((path_256/image_name/"masks"/k).with_suffix(".png")), v)
        for k, v in inners_.items():
            cv2.imwrite(
                str((path_256/image_name/"masks"/("i"+k)).with_suffix(".png")), v)
        edges_addup = np.zeros((256, 256), np.uint8)
        for k, v in edges_.items():
            edges_addup = np.bitwise_or(v, edges_addup)
        cv2.imwrite(
            str((path_256/image_name/"masks"/"edges").with_suffix(".png")), edges_addup)
    return None


def dataset_split_256(image_ranges):
    path = pathlib.Path("/kaggle/working/data")
    path_256 = pathlib.Path("/kaggle/working/data_256")
    minerals = ["Alite", "Blite", "C3A", "fCaO", "Pore"]
    translation = ["A矿", "B矿", "C3A", "游离钙", "孔洞"]
    input_path = '/kaggle/input/lithofaces'
    func = partial(process_original_dataset, input=input_path, path=path, path_256=path_256,
                   minerals=minerals, translation=translation, image_path=image_path)
    tree = ET.parse('data/annotations.xml')
    root = tree.getroot()
    images = []
    for image_ in root.findall(f".//image"):
        if int(image_.attrib['id']) in image_ranges:
            images.append(image_)
    # Reduce from 27s/image to 14s/image
    CPU_NUM = multiprocessing.cpu_count()
    with multiprocessing.Pool(CPU_NUM) as pool:
        result = list(tqdm(pool.imap(process_original_dataset, images),
                           desc="Images", position=0, total=len(images)))
