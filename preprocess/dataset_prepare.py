import itertools
import json
import math
import multiprocessing
import os
import xml.etree.ElementTree as ET
from functools import partial
from itertools import zip_longest

import cv2
import h5py
import numpy as np
import scipy
from skimage import morphology
# from sklearn.model_selection import KFold, train_test_split
from tqdm.autonotebook import tqdm


def split(shape, ywindow=256, xwindow=256):
    """
    split shape(y,x) to list of blocks
    """
    y_size, x_size = shape
    for y in range(0, y_size, ywindow):
        for x in range(0, x_size, xwindow):
            ystop, xstop = y + ywindow, x + xwindow
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


def get_edge(mask, kernel=None):
    if kernel is None:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    shape_classes = np.unique(mask)
    touched = np.zeros(mask.shape, dtype=np.uint16)
    edges = np.zeros(mask.shape, dtype=np.uint16)
    for shape_id in shape_classes:
        if shape_id == 0:
            # background
            continue
        # 图形边界
        shape_ = (mask == shape_id).astype(np.uint16)

        shape_inner = (
            scipy.ndimage.convolve(shape_, kernel, mode="reflect") == 8
        ).astype(np.uint16)

        shape_edge = shape_ - shape_inner
        # 不含图形的mask
        mask_ = (mask > 0).astype(np.uint16) - shape_
        mask_pad = np.pad(mask_, 1, mode="reflect")
        for (y, x) in np.argwhere(shape_edge):
            if mask_pad[y - 1: y + 2, x - 1: x + 2].sum() != 0:
                shape_[y, x] = 0
                touched[y, x] = 1
        edges[shape_edge.astype(bool)] = 1
    edges[touched.astype(bool)] = 1
    return edges


def dataset_file_init(path="lithofaces.h5"):
    # create dataset for every id and every image
    def create_dataset(file, name, shape, dtype):
        file.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            compression="gzip",
            compression_opts=4,
            chunks=True,
            maxshape=(None,) + shape[1:]
        )
    with h5py.File(path, "w", libver="latest", swmr=True) as file:
        create_dataset(file, "images", (0, 256, 256, 3), np.dtype("uint8"))
        create_dataset(file, "masks", (0, 256, 256), np.dtype("uint8"))
        create_dataset(file, "edges", (0, 256, 256), np.dtype("uint8"))
        # create_dataset(file, "weight_maps", (0, 256, 256), np.float)
        create_dataset(file, "shape_distance", (0, 256, 256), np.float)
        create_dataset(file, "neighbor_distance", (0, 256, 256), np.float)
        create_dataset(file, "idx", (0,), h5py.special_dtype(vlen=str))
        create_dataset(file, "labels", (0,), h5py.special_dtype(vlen=str))


def dataset_file_append(path, data, name):
    with h5py.File(path, 'a', libver="latest", swmr=True) as file:
        shape = len(data)
        file[name].resize((file[name].shape[0] + shape), axis=0)
        file[name][-shape:] = data


def process_image(
        image_node,
        input_path="/kaggle/input/lithofaces",
        translations=None,
        label_map=None):
    """
    处理一张图片节点，生成contour
    """
    labels = label_map.keys()
    # get image id
    image_id = image_node.attrib["id"]
    # get image name
    # image_name = image_node.attrib["name"].split("/")[1]
    image_name = image_node.attrib["name"]
    image_path = os.path.join(input_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = image.shape[:2]
    mask = np.zeros(size, dtype=np.uint16)
    edges = np.zeros(size, dtype=np.uint16)
    label_dict = {label: [] for label in labels}
    label_num = 1
    # 0 is background
    for polygon in image_node:
        label = polygon.attrib["label"]
        label = translations[label]
        if label not in labels:
            continue
        shape_points = np.array(
            [eval(_) for _ in polygon.attrib["points"].split(";")]
        ).astype(np.int)
        shape = cv2.drawContours(
            np.zeros(size, dtype=np.uint16),
            # blank image
            [shape_points],  # contours
            -1,  # contour id
            1,  # contour color OR [255,255,255]
            -1,  # contour thickness -1 means fill
        )
        # get shapes overlapping
        overlapping = ((mask > 0).astype(np.uint16) +
                       (shape > 0).astype(np.uint16)) > 1
        edges[overlapping] = 1
        mask[shape.astype(bool)] = label_num
        # 保存shape的类别指示，由于是uint16，可容纳65536个类
        label_dict[label].append(label_num)
        label_num += 1
    edges_ = get_edge(mask)
    edges[edges_.astype(bool)] = 1
    # if you want to remove edges:
    # mask[edges.astype(bool)] = 0
    assert mask.shape == image.shape[:2]
    assert edges.shape == image.shape[:2]

    # label_dict["edges"] = [max(np.unique(mask)) + 1]
    # mask[edges.astype(bool)] = label_dict["edges"][0]
    return image_id, image, mask, edges, label_dict


def process_images(
        annotations,
        translations,
        label_map,
        image_ranges,
        input_path):

    tree = ET.parse(annotations)
    root = tree.getroot()
    images = []
    for image_ in root.findall(".//image"):
        if int(image_.attrib["id"]) in image_ranges:
            images.append(image_)
    func = partial(
        process_image,
        input_path=input_path,
        translations=translations,
        label_map=label_map,
    )

    CPU_NUM = multiprocessing.cpu_count()
    with multiprocessing.Pool(CPU_NUM) as pool:
        results = list(
            tqdm(
                pool.imap(
                    func,
                    images),
                desc="Images",
                position=0,
                total=len(images)))
    results_dict = dict()
    for result in results:
        image_id = result[0]
        results_dict[image_id] = dict()
        results_dict[image_id]["image"] = result[1]
        results_dict[image_id]["masks"] = result[2]
        results_dict[image_id]["edges"] = result[3]
        results_dict[image_id]["dict"] = result[4]
    return results_dict


def trim(label_dict, masks):
    """remove label not in masks"""
    mask_labels = np.unique(masks)
    label_dict_new = dict()
    for label_name, labels in label_dict.items():
        label_dict_new.setdefault(label_name, [])
        for label in labels:
            if label in mask_labels:
                label_dict_new[label_name].append(label)
    return label_dict_new


def balancewm(mask):
    # class balance weight map
    # from 0-1
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [1.0 / np.sum(mask == i) for i in classes]
    freq /= max(freq)
    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]
    return wc


def get_shape_distance(mask):
    # calculate shape distance as described in <Cell segmentation and tracking
    # using CNN-based distancepredictions and a graph-based matching strategy>
    shape_distance = np.zeros_like(mask, dtype=np.float64)
    for shape_id in np.unique(mask):
        if shape_id == 0:
            # background
            continue
        indice = mask == shape_id
        shape_distance_ = scipy.ndimage.distance_transform_edt(indice)
        shape_distance_ = shape_distance_ / shape_distance_.max()
        shape_distance += shape_distance_
    return shape_distance


def get_neighbor_distance(mask):
    # calculate shape distance as described in <Cell segmentation and tracking
    # using CNN-based distancepredictions and a graph-based matching strategy>
    neighbor_distance = np.zeros_like(mask, dtype=np.float64)
    mask_binary = mask.astype(bool).astype(mask.dtype)
    for shape_id in np.unique(mask):
        if shape_id == 0:
            # background
            continue
        indice = mask == shape_id
        indice_ = mask != shape_id
        mask_without_this_shape = mask_binary.copy()
        mask_without_this_shape[indice] = 0
        invert_dt = scipy.ndimage.distance_transform_edt(
            1 - mask_without_this_shape)
        # cutting
        invert_dt[indice_] = 0.0
        # normalize
        maxi = invert_dt.max()
        if maxi != 0.0:
            invert_dt = invert_dt / maxi
        # invert
        invert_dt = 1.0 - invert_dt
        # cutting again
        invert_dt[indice_] = 0.0
        # normalize again
        maxi = invert_dt.max()
        if maxi != 0.0:
            invert_dt = invert_dt / maxi
        neighbor_distance += invert_dt
    neighbor_distance = morphology.closing(neighbor_distance,
                                           morphology.disk(3))
    return np.power(neighbor_distance, 3)


def get_unet_border_weight_map(
        annotation,
        w0=5.0,
        sigma=13.54591536778324,
        eps=1e-32):
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
    assert annotation.dtype in [
        np.uint8,
        np.uint16,
    ], "Expected data type uint, it is {}".format(annotation.dtype)
    labeled_array = annotation.copy()
    inner = scipy.ndimage.distance_transform_edt(annotation)
    inner = (inner.max() - inner) / inner.max()
    inner[annotation == 0] = 0
    # if there is only one label or only background
    if len(np.unique(labeled_array)) == 1:
        return inner
    # if there is only one label and background
    if len(np.unique(labeled_array)) == 2:
        if 0 in np.unique(labeled_array):
            return inner
    # cells instances for distance computation
    # 4 connected i.e default (cross-shaped)
    # structuring element to measure connectivy
    # If cells are 8 connected/touching they are labeled as one single object
    # Loss metric on such borders is not useful
    # class balance weights w_c(x)
    unique_values = np.unique(labeled_array).tolist()
    weight_map = [0] * len(unique_values)
    for index, unique_value in enumerate(unique_values):
        mask = np.zeros(
            (annotation.shape[0],
             annotation.shape[1]),
            dtype=np.float64)
        mask[annotation == unique_value] = 1
        weight_map[index] = 1 / mask.sum()

    # this normalization is important - foreground pixels must have weight 1
    weight_map = [i / max(weight_map) for i in weight_map]

    wc = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.float64)
    for index, unique_value in enumerate(unique_values):
        wc[annotation == unique_value] = weight_map[index]
    # cells instances for distance computation
    # 4 connected i.e default (cross-shaped)
    # structuring element to measure connectivy
    # If cells are 8 connected/touching they are labeled as one single object
    # Loss metric on such borders is not useful
    # Not NEED to find labels
    # labeled_array, _ = scipy.ndimage.measurements.label(annotation)
    # cells distance map
    border_loss_map = np.zeros(
        (annotation.shape[0], annotation.shape[1]), dtype=np.float64
    )
    distance_maps = np.zeros(
        (annotation.shape[0], annotation.shape[1], np.max(labeled_array)),
        dtype=np.float64,
    )

    if np.max(labeled_array) >= 2:
        for index in range(np.max(labeled_array)):
            mask = np.ones_like(labeled_array)
            mask[labeled_array == index + 1] = 0
            distance_maps[:, :,
                          index] = scipy.ndimage.distance_transform_edt(mask)
    distance_maps = np.sort(distance_maps, 2)
    d1 = distance_maps[:, :, 0]
    d2 = distance_maps[:, :, 1]
    border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

    zero_label = np.zeros(
        (annotation.shape[0],
         annotation.shape[1]),
        dtype=np.float64)
    zero_label[labeled_array == 0] = 1
    border_loss_map = np.multiply(border_loss_map, zero_label)
    return border_loss_map + inner + wc


def split_image(result, resize_factors=[
        i**0.5 for i in [1, 2, 3, 4]], window=256):
    def pad(block, pady, padx):
        # to center image or mask
        pady1 = pady // 2
        pady2 = pady - pady1
        padx1 = padx // 2
        padx2 = padx - padx1
        #
        if len(block.shape) == 3:
            shape = ((pady1, pady2), (padx1, padx2), (0, 0))
        else:
            shape = ((pady1, pady2), (padx1, padx2))
        return np.pad(
            block, shape, mode="constant", constant_values=0)
    if result is None:
        return
    image_id, content = result
    # idx format: image_id-resize_id-block_id
    image = content["image"]
    masks = content["masks"]
    edges = content["edges"]
    labels_dict = content["dict"]
    size = content["image"].shape[:2]
    idx_data = []
    images_data = []
    masks_data = []
    edges_data = []
    # weight_maps_data = []
    shape_distance_data = []
    neighbor_distance_data = []
    labels_data = []
    window = window

    for resize_id, factor in enumerate(resize_factors):
        size_new = (math.ceil(size[0] / factor), math.ceil(size[1] / factor))
        if factor == 1:
            image_new = image
            masks_new = masks
            edges_new = edges
        else:
            image_new = cv2.resize(image, (size_new[1], size_new[0]))
            masks_new = np.zeros((size_new[0], size_new[1]), dtype=np.uint16)
            # 分离进行缩放masks，并避免出现边界出现不同数值的bug
            for i in np.unique(masks):
                shape_tmp = masks == i
                shape = cv2.resize(shape_tmp.astype(np.uint16),
                                   (size_new[1], size_new[0]),
                                   cv2.INTER_NEAREST)
                masks_new[shape.astype(bool)] = i
            # after resize mask,get edges of touched
            touched = get_edge(masks_new)
            edges_new = cv2.resize(edges.astype(np.uint16),
                                   (size_new[1], size_new[0]),
                                   cv2.INTER_NEAREST)
            edges_new = ((edges_new + touched) > 0).astype(np.uint16)
        blocks = split(size_new, ywindow=window, xwindow=window)
        for idx, (should_pad, [y, y_stop], [x, x_stop]) in enumerate(blocks):
            image_block = image_new[y:y_stop, x:x_stop, ...]
            masks_block = masks_new[y:y_stop, x:x_stop]
            edges_block = edges_new[y:y_stop, x:x_stop]
            # # weight map should calculate before padding
            # weight_map_block = masks_block.copy()
            # weight_map_block[edges_block.astype(bool)] = 0
            # border_wm = get_unet_border_weight_map(weight_map_block)
            # distances should calculate before padding
            shape_distance_block = get_shape_distance(masks_block)
            neighbor_distance_block = get_neighbor_distance(masks_block)
            if should_pad:
                pady = window - (y_stop - y)
                padx = window - (x_stop - x)
                image_block = pad(image_block, pady, padx)
                masks_block = pad(masks_block, pady, padx)
                edges_block = pad(edges_block, pady, padx)
                # border_wm = pad(border_wm, pady, padx)
                shape_distance_block = pad(shape_distance_block, pady, padx)
                neighbor_distance_block = pad(
                    neighbor_distance_block, pady, padx)
            idx_data.append(f"{image_id}-{resize_id}-{idx}".encode("ascii"))
            images_data.append(image_block)
            masks_data.append(masks_block)
            edges_data.append(edges_block)
            # weight_maps_data.append(border_wm)
            shape_distance_data.append(shape_distance_block)
            neighbor_distance_data.append(neighbor_distance_block)
            labels_data.append(
                json.dumps(
                    trim(
                        labels_dict,
                        masks_block)).encode("ascii"))
    data_len = len(idx_data)
    assert len(labels_data) == data_len
    for data in (
            images_data,
            masks_data,
            edges_data,
            # weight_maps_data,
            shape_distance_data,
            neighbor_distance_data
    ):
        assert len(data) == data_len
        for element in data:
            assert element.shape[0] == window
            assert element.shape[1] == window
    return idx_data, images_data, masks_data, \
        edges_data, shape_distance_data, neighbor_distance_data, labels_data


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def create_dataset(results, path):
    dataset_file_init(path)
    CPU_NUM = multiprocessing.cpu_count()
    func = split_image
    size = 10
    grouped = grouper(results.items(), size)
    for group in grouped:
        pool = multiprocessing.Pool(CPU_NUM)
        with pool:
            results = tuple(tqdm(
                pool.imap_unordered(func, group),
                desc="adding to h5file",
                position=0,
                total=size))
        results = tuple(filter(None, results))
        (idx_data,
         images_data,
         masks_data,
         edges_data,
         labels_data,
         weight_maps_data,
         shape_distance_data,
         neighbor_distance_data) = zip(*results)
        idx_data = tuple(itertools.chain.from_iterable(idx_data))
        images_data = tuple(itertools.chain.from_iterable(images_data))
        masks_data = tuple(itertools.chain.from_iterable(masks_data))
        edges_data = tuple(itertools.chain.from_iterable(edges_data))
        # weight_maps_data = tuple(
        # itertools.chain.from_iterable(weight_maps_data))
        shape_distance_data = tuple(
            itertools.chain.from_iterable(shape_distance_data))
        neighbor_distance_data = tuple(
            itertools.chain.from_iterable(neighbor_distance_data))
        labels_data = tuple(itertools.chain.from_iterable(labels_data))
        dataset_file_append(path, idx_data, "idx")
        dataset_file_append(path, labels_data, "labels")
        dataset_file_append(path, images_data, "images")
        dataset_file_append(path, masks_data, "masks")
        dataset_file_append(path, edges_data, "edges")
        # dataset_file_append(path, weight_maps_data, "weight_maps")
        dataset_file_append(path, shape_distance_data, "shape_distance")
        dataset_file_append(path, neighbor_distance_data, "neighbor_distance")
        del idx_data
        del images_data
        del masks_data
        del edges_data
        # del weight_maps_data
        del shape_distance_data
        del neighbor_distance_data
        del labels_data
        del results


if __name__ == "__main__":
    translations = {
        "A矿": "Alite",
        "B矿": "Blite",
        "C3A": "C3A",
        "游离钙": "fCaO",
        "孔洞": "Pore",
    }
    select_classes = ["Alite", "Blite", "C3A", "Pore"]
    image_ranges = list(range(40)) + [84, 94, 121, 138]

    if "KAGGLE_CONTAINER_NAME" in os.environ:
        input_path = '/kaggle/input/lithofaces'
        annotations = "/kaggle/working/data/annotations.xml"
        cwpath = "/kaggle/working/class_weights.txt"
        dataset_path = "/kaggle/working/lithofaces.h5"
    else:
        input_path = (
            "/home/lao/Notebook/Research/"
            + "Clinker Lithofacies Automation/data/segmentation/images"
        )
        annotations = "/home/lao/Data/annotations.xml"
        cwpath = "/home/lao/Data/class_weights.txt"
        dataset_path = "/home/lao/Data/lithofaces.h5"
    label_map = {label: i for i, label in enumerate(select_classes, start=1)}

    print("Original dataset generating from annotations.xml.")
    results = process_images(
        annotations=annotations,
        translations=translations,
        label_map=label_map,
        image_ranges=image_ranges,
        input_path=input_path)
    print("Splitting to train val.")

    print("Generating hdf5 file from Datasets.")
    create_dataset(results, path=dataset_path)
