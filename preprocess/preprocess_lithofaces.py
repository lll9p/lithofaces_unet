import copy
import gc
import multiprocessing
import os
import pathlib
import shutil
import time
import xml.etree.ElementTree as ET
from functools import partial
from itertools import zip_longest

import cv2
import h5py
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import morphology
from tqdm.autonotebook import tqdm

kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
def get_val_images(image_ranges):
    four_blocks = list(range(4))
    val_images=dict()
    from sklearn.model_selection import train_test_split
    image_blocks = list()
    for block in four_blocks:
        for image in image_ranges:
            image_blocks.append(f"{image}-{block}")
    train,val = train_test_split(image_blocks,test_size=0.2,random_state=42)
    for val_id in val:
        image,i = val_id.split("-")
        val_images.setdefault(image,[])
        val_images[image].append(int(i))
    return val_images

def mask_instance_to_semantic(mask, label, label_map):
    mask_temp = np.zeros_like(mask, dtype=mask.dtype)
    for label_name, label_values in label.items():
        for label_value in label_values:
            mask_temp[mask == label_value] = label_map[label_name]
    return mask_temp


def split(shape, ywindow=256, xwindow=256):
    """
    split shape(y,x) to list of blocks
    """
    y_size, x_size = shape
    for y in range(0, y_size, ywindow):
        for x in range(0, x_size, xwindow):
            ystop, xstop = y+ywindow, x+xwindow
            if ystop > y_size or xstop > x_size:
                if ystop > y_size:
                    block = True, [y, y_size], [x, xstop]
                if xstop > x_size:
                    block = True, [y, ystop], [x, x_size]
                if ystop > y_size and xstop > x_size:
                    block = True, [y, y_size], [x, x_size]
            else:
                block = False, [y, ystop], [x, xstop]
            yield block


def split_four(shape):
    y_size, x_size = shape
    return split(shape, ywindow=y_size//2, xwindow=x_size//2)


def fix_edge(mask, kernel=kernel):
    shape_classes = np.unique(mask)
    mask_new = np.zeros(mask.shape, dtype=np.uint16)
    touched = np.zeros(mask.shape, dtype=np.uint16)
    for shape_id in shape_classes:
        if shape_id == 0:
            # background
            continue
        # 图形边界
        shape_ = (mask == shape_id).astype(np.uint16)

        shape_inner = (scipy.ndimage.convolve(
            shape_, kernel, mode='reflect') == 8).astype(np.uint16)

        shape_edge = shape_ - shape_inner
        # 不含图形的mask
        mask_ = (mask > 0).astype(np.uint16)-shape_
        mask_pad = np.pad(mask_, 1, mode='reflect')
        for (y, x) in np.argwhere(shape_edge):
            if mask_pad[y:y+3, x:x+3].sum() != 0:
                shape_[y, x] = 0
                touched[y, x] = 1
        shape_ *= shape_id
        mask_new += shape_
    return mask_new, touched


def process_original_dataset(image_node,
                             input_path="/kaggle/input/lithofaces",
                             translations=None,
                             label_map=None):
    """
    处理一张图片节点，生成contour,并把图片及countor分割为256x256->256x256,512x512->256x256
    """
    pbar = None
    #path = pathlib.Path(path)
    labels = label_map.keys()
    # get image id
    image_id = image_node.attrib['id']
    # get image name
    image_name = image_node.attrib['name'].split("/")[1]
    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], dtype=np.uint16)
    edges = np.zeros(image.shape[:2], dtype=np.uint16)
    label_dict = {label: [] for label in labels}
    label_num = 1
    # 0 is background
    for polygon in image_node:
        label = polygon.attrib['label']
        label = translations[label]
        if label not in labels:
            continue
        shape_points = np.array(
            [eval(_) for _ in polygon.attrib['points'].split(";")]).astype(np.int)
        shape = cv2.drawContours(np.zeros(image.shape[:2], dtype=np.uint16),  # blank image
                                 [shape_points],  # contours
                                 -1,  # contour id
                                 label_num,  # contour color OR [255,255,255]
                                 -1  # contour thickness -1 means fill
                                 )
        # 保存shape的类别指示，由于是uint16，可容纳65536个类
        label_dict[label].append(label_num)
        label_num += 1
        # Process shapes overlapping
        overlapping = ((mask > 0).astype(np.uint16) +
                       (shape > 0).astype(np.uint16)) > 1
        mask[overlapping] = 0
        shape[overlapping] = 0
        edges[overlapping] = 1
        mask = mask + shape
    mask, touched = fix_edge(mask)
    edges[touched.astype(bool)] = 1
    label_dict['edges'] = [max(np.unique(mask))+1]
    mask[edges.astype(bool)] = label_dict['edges'][0]
    return image_id, image, mask, label_dict


# class balance weight map


def balancewm(mask):
    # from 0-1
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [1.0 / np.sum(mask == i) for i in classes]
    freq /= max(freq)
    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]
    return wc
#wc = balancewm(annotation)


def get_unet_border_weight_map(annotation, w0=5.0, sigma=13.54591536778324, eps=1e-32):
    # https://github.com/czbiohub/microDL/blob/master/micro_dl/utils/masks.py
    """
    Return weight map for borders as specified in UNet paper
    :param annotation A 2D array of shape (image_height, image_width)
     contains annotation with each class labeled as an integer.
    :param w0 multiplier to the exponential distance loss
     default 10 as mentioned in UNet paper
    :param sigma standard deviation in the exponential distance term
     e^(-d1 + d2) ** 2 / 2 (sigma ^ 2)
     default 5 as mentioned in UNet paper
    :return weight mapt for borders as specified in UNet
    TODO: Calculate boundaries directly and calculate distance
    from boundary of cells to another
    Note: The below method only works for UNet Segmentation only
    """
    # if there is only one label, zero return the array as is
    if np.sum(annotation) == 0:
        return annotation

    # Masks could be saved as .npy bools, if so convert to uint8 and generate
    # labels from binary
    if annotation.dtype == bool:
        annotation = annotation.astype(np.uint8)
    assert annotation.dtype in [np.uint8, np.uint16], (
        "Expected data type uint, it is {}".format(annotation.dtype))

    # cells instances for distance computation
    # 4 connected i.e default (cross-shaped)
    # structuring element to measure connectivy
    # If cells are 8 connected/touching they are labeled as one single object
    # Loss metric on such borders is not useful
    # Not NEED to find labels
    #labeled_array, _ = scipy.ndimage.measurements.label(annotation)
    labeled_array = annotation.copy()
    inner = scipy.ndimage.distance_transform_edt(annotation)
    inner = (inner.max()-inner)/inner.max()
    inner[annotation == 0] = 0
    # if there is only one label or only background
    if len(np.unique(labeled_array)) == 1:
        return inner
    # if there is only one label and background
    if len(np.unique(labeled_array)) == 2:
        if 0 in np.unique(labeled_array):
            return inner
#     # class balance weights w_c(x)
#     unique_values = np.unique(labeled_array).tolist()
#     weight_map = [0] * len(unique_values)
#     for index, unique_value in enumerate(unique_values):
#         mask = np.zeros(
#             (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
#         mask[annotation == unique_value] = 1
#         weight_map[index] = 1 / (mask.sum()+eps)

#     # this normalization is important - foreground pixels must have weight 1
#     weight_map = [i / max(weight_map) for i in weight_map]

#     wc = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
#     for index, unique_value in enumerate(unique_values):
#         wc[annotation == unique_value] = weight_map[index]

    # cells distance map
    border_loss_map = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    distance_maps = np.zeros(
        (annotation.shape[0], annotation.shape[1], np.max(labeled_array)),
        dtype=np.float64)

    if np.max(labeled_array) >= 2:
        for index in range(np.max(labeled_array)):
            mask = np.ones_like(labeled_array)
            mask[labeled_array == index + 1] = 0
            distance_maps[:, :, index] = \
                scipy.ndimage.distance_transform_edt(mask)
    distance_maps = np.sort(distance_maps, 2)
    d1 = distance_maps[:, :, 0]
    d2 = distance_maps[:, :, 1]
    border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

    zero_label = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    zero_label[labeled_array == 0] = 1
    border_loss_map = np.multiply(border_loss_map, zero_label)
    return border_loss_map+inner


def split_to_256(image, mask, label):
    # image or mask
    y_size, x_size = image.shape[:2]
    resize_factors = {"768": [1, 2**0.5, 2],
                      "1536": [1, 2**0.5, 2, 3, 4], "384": [1, 2**0.5]}
    images, masks, weight_maps = [], [], []
    for i in resize_factors[f"{y_size}"]:
        new_shape = (int(y_size/i), int(x_size/i))
        if new_shape[0] == y_size:
            # no need to resize
            image_new = image
            mask_new = mask
        else:
            image_new = cv2.resize(image, (new_shape[1], new_shape[0]))
            mask_resized = cv2.resize(
                mask, (new_shape[1], new_shape[0]), cv2.INTER_NEAREST)
            edges = mask_resized == label['edges'][0]
            mask_resized[edges] = 0
            mask_new, touched = fix_edge(mask_resized)
            mask_new[edges] = label['edges'][0]
            mask_new[touched.astype(bool)] = label['edges'][0]
        blocks = list(split(new_shape, 256))
        for block in blocks:
            pad_flag, [y, y_stop], [x, x_stop] = block
            image_block = image_new[y:y_stop, x:x_stop, ...]
            mask_block = mask_new[y:y_stop, x:x_stop]
            yy, xx = image_block.shape[:2]
            if yy == 0 or xx == 0:
                continue
            if 0.5 <= (yy/xx) and (yy/xx) <= 2.1:
                try:
                    image_block = np.pad(
                        image_block, ((256-yy, 0), (256-xx, 0), (0, 0)), mode='reflect')
                    mask_block = np.pad(
                        mask_block, ((256-yy, 0), (256-xx, 0)), mode='reflect')
                except:
                    print(image_block.shape, mask_block.shape)
                    raise f"{image_block.shape}/{mask_block.shape}"
            else:
                continue
            images.append(image_block)
            masks.append(mask_block)
            weight_map_block = mask_block.copy()
            # 去掉边界再计算weightmap
            edges = weight_map_block == label['edges'][0]
            weight_map_block[edges] = 0
            # 腐蚀3pixel
            weight_map_block = morphology.erosion(
                weight_map_block, morphology.disk(3))
            weight_map = get_unet_border_weight_map(
                weight_map_block)
            weight_maps.append(weight_map)
            # convert mask to semantic
            assert image_block.shape == (
                256, 256, 3), f"{image_block.shape} Wrong!"
            assert mask_block.shape == (256, 256), f"{mask_block.shape} Wrong!"

    return images, masks, weight_maps


# val_images = {"3": [0],
#              "5": [0, 2],
#              "9": [1],
#              "20": [3],
#              "22": [3],
#              "26": [0, 3],
#              }

def _image_deal(result, val_images, label_map):
    _dataset = dict(images=[], masks=[], weight_maps=[], idx=[])
    image_id, image, mask, label = result
    if image_id in val_images:
        # 若是在预选的val图片
        split_idx = val_images[image_id]
        for index, split_block in enumerate(split_four(image.shape[:2])):
            _, [y, y_stop], [x, x_stop] = split_block
            images, masks, weight_maps = split_to_256(
                image[y:y_stop, x:x_stop, ...], mask[y:y_stop, x:x_stop], label)
            masks = [mask_instance_to_semantic(
                mask, label, label_map) for mask in masks]
            if index in split_idx:
                _dataset["images"] += images
                _dataset["masks"] += masks
                _dataset["weight_maps"] += weight_maps
                _dataset["idx"] += [
                    f"{image_id}-V{index}-{i}" for i in range(len(images))]
            else:
                _dataset["images"] += images
                _dataset["masks"] += masks
                _dataset["weight_maps"] += weight_maps
                _dataset["idx"] += [
                    f"{image_id}-T{index}-{i}" for i in range(len(images))]
    else:
        images, masks, weight_maps = split_to_256(
            image, mask, label)
        masks = [mask_instance_to_semantic(
            mask, label, label_map) for mask in masks]
        _dataset["images"] += images
        _dataset["masks"] += masks
        _dataset["weight_maps"] += weight_maps
        _dataset["idx"] += [
            f"{image_id}-T0-{i}" for i in range(len(images))]
    return _dataset


def form_datasets(results, val_images, label_map):
    val_dataset = dict(images=[], masks=[], weight_maps=[], idx=[])
    train_dataset = dict(images=[], masks=[], weight_maps=[], idx=[])
    func = partial(_image_deal, val_images=val_images, label_map=label_map)
    CPU_NUM = multiprocessing.cpu_count()
    with multiprocessing.Pool(CPU_NUM) as pool:
        dataset_results = list(tqdm(pool.imap(func, results),
                                    desc="Datasets", position=0, total=len(results)))
    for dataset_result in dataset_results:
        for i, idx in enumerate(dataset_result['idx']):
            if "V" in idx:
                val_dataset["images"].append(dataset_result["images"][i])
                val_dataset["masks"].append(dataset_result["masks"][i])
                val_dataset["weight_maps"].append(
                    dataset_result["weight_maps"][i])
                val_dataset["idx"].append(idx)
            if "T" in idx:
                train_dataset["images"].append(dataset_result["images"][i])
                train_dataset["masks"].append(dataset_result["masks"][i])
                train_dataset["weight_maps"].append(
                    dataset_result["weight_maps"][i])
                train_dataset["idx"].append(idx)
    datasets = dict(train=train_dataset, val=val_dataset)
    return datasets


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def dataset_file_init(path="lithofaces.h5",
                      datasets=None,
                      images_shape=None,
                      masks_shape=None,
                      dtype=np.dtype('uint8')):
    with h5py.File(path, "w") as file:
        for dataset_name, dataset in datasets.items():
            file.create_dataset(f"{dataset_name}/images", shape=tuple([len(dataset['images'])])+images_shape,
                                dtype=dtype, compression="gzip", compression_opts=4, chunks=True)
            file.create_dataset(f"{dataset_name}/masks", shape=tuple([len(dataset['masks'])])+masks_shape,
                                dtype=dtype, compression="gzip", compression_opts=4, chunks=True)
            file.create_dataset(f"{dataset_name}/weight_maps", shape=tuple([len(dataset['weight_maps'])])+masks_shape,
                                dtype=np.float, compression="gzip", compression_opts=4, chunks=True)
            #idxes = [s.encode('ascii') for s in dataset['idx']]
            file.create_dataset(f"{dataset_name}/idx", shape=(len(dataset['idx']),),
                                dtype='S10', compression="gzip", compression_opts=4, chunks=True)


def dataset_to_h5(datasets, dataset_path="lithofaces.h5"):
    dataset_file_init(path=dataset_path,
                      datasets=datasets,
                      images_shape=(256, 256, 3),
                      masks_shape=(256, 256),
                      dtype=np.dtype('uint8'))
    with h5py.File(dataset_path, "a") as file:
        for dataset_name, dataset in datasets.items():
            dataset_len = len(dataset["images"])
            assert len(dataset['masks']) == dataset_len
            assert len(dataset['weight_maps']) == dataset_len
            assert len(dataset['idx']) == dataset_len
            file[f"{dataset_name}/images"][:] = np.stack(dataset["images"])
            #_masks = np.stack(masks)
            file[f"{dataset_name}/masks"][:] = np.stack(dataset["masks"])
            file[f"{dataset_name}/weight_maps"][:
                                                ] = np.stack(dataset["weight_maps"])
            file[f"{dataset_name}/idx"][:] = [s.encode('ascii')
                                              for s in dataset['idx']]
#translations={'A矿': 'Alite', 'B矿': 'Blite', 'C3A': 'C3A', '游离钙': 'fCaO', '孔洞': 'Pore'}
#select_classes=['Alite', 'Blite', 'C3A', 'Pore']
#image_ranges = list(range(31))+[38,39]
#input_path = '/kaggle/input/lithofaces'
#annotations = "/kaggle/working/data/annotations.xml"


def process_original(annotations, translations, label_map, image_ranges, input_path):
    tree = ET.parse(annotations)
    root = tree.getroot()
    images = []
    for image_ in root.findall(f".//image"):
        if int(image_.attrib['id']) in image_ranges:
            images.append(image_)
    func = partial(process_original_dataset, input_path=input_path,
                   translations=translations, label_map=label_map)

    CPU_NUM = multiprocessing.cpu_count()
    with multiprocessing.Pool(CPU_NUM) as pool:
        results = list(tqdm(pool.imap(func, images),
                            desc="Images", position=0, total=len(images)))
    return results


def class_weight(select_classes, results, label_map):
    class_weight = {label: 0 for label in select_classes}
    pixels = 0
    for result in results:
        image_id, image, mask, label = result
        mask_class = mask_instance_to_semantic(mask, label, label_map)
        classes = np.unique(mask)
        for label_name, label_value in label_map.items():
            class_weight[label_name] += np.sum(mask_class == label_value)
        pixels += np.product(mask.shape)
    for label, value in class_weight.items():
        class_weight[label] = pixels/value
    max_weight = max(class_weight.values())
    for label, value in class_weight.items():
        class_weight[label] = value/max_weight
    return class_weight


if __name__ == "__main__":
    translations = {'A矿': 'Alite', 'B矿': 'Blite',
                    'C3A': 'C3A', '游离钙': 'fCaO', '孔洞': 'Pore'}
    select_classes = ['Alite', 'Blite', 'C3A', 'Pore', 'edges']
    image_ranges = list(range(31))+[38, 39, 94, 121, 138]

    input_path = '/home/lao/Notebook/Research/Clinker Lithofacies Automation/data/segmentation/images'
    annotations = "/home/lao/Data/annotations.xml"
    label_map = {label: i for i, label in enumerate(select_classes, start=1)}

    print("Original dataset generating from annotations.xml.")
    results = process_original(
        annotations, translations, label_map, image_ranges, input_path)

    print("Caculating class_weight.")
    class_weights = class_weight(select_classes, results, label_map)
    with open("/home/lao/Data/class_weights.txt", mode="w") as file:
        file.write(class_weights.__repr__())
    val_images = get_val_images(image_ranges)
    #val_images = {"3": [0],
    #              "5": [0, 2],
    #              "9": [1],
    #              "20": [3],
    #              "22": [3],
    #              "26": [0, 3],
    #              }
    print("Generating Datasets.")
    datasets = form_datasets(results, val_images, label_map)
    print("Generating hdf5 file from Datasets.")
    dataset_to_h5(datasets, dataset_path="/home/lao/Data/lithofaces.h5")
"""

https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
def weight_map(mask, w0=10, sigma=5):
"
Create a UNet weight map from a boolean `mask` where `True`
marks the interior pixels of an instance.
"
import scipy.ndimage as ndi

# if the mask only has one contiguous class,
# then there isn't much to do.
if len(np.unique(mask)) == 1:
return np.ones(mask.shape, dtype=np.float32) * 0.5

# calculate the class-balanced weight map w_c
w_c = np.zeros(mask.shape, dtype=np.float32)
w_1 = 1 - float(np.count_nonzero(mask)) / w_c.size
w_0 = 1 - w_1
w_c[mask > 0.5] = w_1
w_c[mask < 0.5] = w_0

# calculate the distance-weighted emphases w_e
segs, _ = ndi.label(mask)
if segs.max() == 1:
# if there is only 1 instance plus background,
# then there are no separations
return w_c
ilabels = range(1, segs.max()+1)
distmaps = np.stack([ndi.distance_transform_edt(segs != l) for l in ilabels])
distmaps = np.sort(distmaps, axis=0)[:2]

w_e = w0 * np.exp((-1 * (distmaps[0] + distmaps[1]) ** 2) / (2 * (sigma ** 2)))
w_e[mask] = 0.

return w_c + w_e
"""
