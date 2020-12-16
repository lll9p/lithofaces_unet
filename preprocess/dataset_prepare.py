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
from sklearn.model_selection import train_test_split
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


kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def fix_edge(mask, kernel=kernel):
    shape_classes = np.unique(mask)
    mask_new = np.zeros(mask.shape, dtype=np.uint16)
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
        mask_new[shape_.astype(bool)] = shape_id
        edges[shape_edge.astype(bool)] = 1
    edges[touched.astype(bool)] = 1
    return mask_new, edges


def get_val_blocks(images):
    image_blocks = list()
    for image in images:
        image_id = image.attrib["id"]
        y, x = int(image.attrib["height"]), int(image.attrib["width"])
        blocks = split((y, x), 512, 512)
        for idx, block in enumerate(blocks):
            image_blocks.append(f"{image_id}-{idx}")
    train, val = train_test_split(image_blocks, test_size=0.3, random_state=42)
    val_ = dict()
    for val_block in val:
        image_id, block_id = val_block.split("-")
        val_.setdefault(image_id, [])
        val_[image_id].append(int(block_id))
    return val_


def dataset_file_init(path="lithofaces.h5"):
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
        for dataset_name in ["train", "val"]:
            create_dataset(file, f"{dataset_name}/images",
                           (0, 256, 256, 3), np.dtype("uint8"))
            create_dataset(file, f"{dataset_name}/masks",
                           (0, 256, 256), np.dtype("uint8"))
            create_dataset(file, f"{dataset_name}/edges",
                           (0, 256, 256), np.dtype("uint8"))
            create_dataset(
                file, f"{dataset_name}/weight_maps", (0, 256, 256), np.float)
            create_dataset(file, f"{dataset_name}/idx", (0,), "S15")
            create_dataset(file, f"{dataset_name}/labels", (0,), "S10000")


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
    处理一张图片节点，生成contour,并把图片padding成256的偶数倍
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
    should_pad = False
    if size[0] == 768:
        should_pad = True
        image = np.pad(image, ((256, 0), (0, 0), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((256, 0), (0, 0)), mode="reflect")
        edges = np.pad(edges, ((256, 0), (0, 0)), mode="reflect")

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
            label_num,  # contour color OR [255,255,255]
            -1,  # contour thickness -1 means fill
        )
        if should_pad:
            # pad 1024*768 image to 1024*1024
            # easy to split train test sets
            shape = np.pad(shape, ((256, 0), (0, 0)), mode="reflect")
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
    mask, edges_ = fix_edge(mask)
    edges[edges_.astype(bool)] = 1
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
    val_blocks = get_val_blocks(images)
    return results_dict, val_blocks


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


def dataset_split(results, val_blocks):
    result_train = dict()
    result_val = dict()
    for image_id, result in results.items():
        if image_id not in val_blocks:
            result_train[f"{image_id}F"] = result
    for image_id, block_ids in val_blocks.items():
        result = results[image_id]
        size = result["image"].shape[:2]
        blocks = split(size, 512, 512)
        for idx, (pad_flag, [y, y_stop], [x, x_stop]) in enumerate(blocks):
            if idx not in block_ids:
                key = f"{image_id}S-{idx}"
                result_train.setdefault(key, dict())
                result_train[key]["image"] = result["image"][
                    y: y_stop, x: x_stop, ...]
                result_train[key]["masks"] = result["masks"][
                    y: y_stop, x: x_stop, ...]
                result_train[key]["edges"] = result["edges"][
                    y: y_stop, x: x_stop, ...]
                result_train[key]["dict"] = trim(
                    result["dict"], result_train[key]["masks"])
            else:
                key = f"{image_id}V-{idx}"
                result_val.setdefault(key, dict())
                result_val[key]["image"] = result["image"][
                    y: y_stop, x: x_stop, ...]
                result_val[key]["masks"] = result["masks"][
                    y: y_stop, x: x_stop, ...]
                result_val[key]["edges"] = result["edges"][
                    y: y_stop, x: x_stop, ...]
                result_val[key]["dict"] = trim(
                    result["dict"], result_val[key]["masks"])
    return result_train, result_val
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


def split_data(result, lock, path, resize_factors=[
               i**0.5 for i in [1, 2, 3, 4, 9, 16]]):
    name, content = result
    idx_data = []
    images_data = []
    masks_data = []
    edges_data = []
    labels_data = []
    weight_maps_data = []
    window = 256

    def pad(size, window=256):
        return math.ceil(size / window) * window
    for factor in resize_factors:
        image = content["image"]
        masks = content["masks"]
        edges = content["edges"]
        labels_dict = content["dict"]
        size = content["image"].shape[:2]
        size_new = (int(size[0] / factor), int(size[1] / factor))
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
                shape = shape > 0
                masks_new[shape] = i
            masks_new, touched = fix_edge(masks_new)
            edges_new = cv2.resize(edges.astype(np.uint16),
                                   (size_new[1], size_new[0]),
                                   cv2.INTER_NEAREST)
            edges_new = ((edges_new + touched) > 0).astype(np.uint16)
            pady = pad(size_new[0], window) - size_new[0]
            padx = pad(size_new[1], window) - size_new[1]
            image_new = np.pad(
                image_new,
                ((pady, 0), (padx, 0), (0, 0)),
                mode="reflect")
            masks_new = np.pad(
                masks_new,
                ((pady, 0), (padx, 0)),
                mode="reflect")
            edges_new = np.pad(
                edges_new,
                ((pady, 0), (padx, 0)),
                mode="reflect")
        size_new_pad = image_new.shape[:2]
        blocks = split(size_new_pad, ywindow=window, xwindow=window)
        for idx, (_, [y, y_stop], [x, x_stop]) in enumerate(blocks):
            image_block = image_new[y:y_stop, x:x_stop, ...]
            masks_block = masks_new[y:y_stop, x:x_stop]
            edges_block = edges_new[y:y_stop, x:x_stop]
            border_wm = get_unet_border_weight_map(masks_block)
            idx_data.append(f"{name}-{factor:.1f}-{idx}".encode("ascii"))
            images_data.append(image_block)
            masks_data.append(masks_block)
            edges_data.append(edges_block)
            weight_maps_data.append(border_wm)
            labels_data.append(
                json.dumps(
                    trim(
                        labels_dict,
                        masks_block)).encode("ascii"))
    with lock:
        dataset_file_append(path, idx_data, f"{name}/idx")
        dataset_file_append(path, labels_data, f"{name}/labels")
        dataset_file_append(path, images_data, f"{name}/images")
        dataset_file_append(path, masks_data, f"{name}/masks")
        dataset_file_append(path, edges_data, f"{name}/edges")
        dataset_file_append(path, weight_maps_data, f"{name}/weight_maps")
    # return idx_data, images_data, masks_data, \
        # edges_data, weight_maps_data, labels_data


def create_dataset(train, val, path):
    dataset_file_init(path)
    CPU_NUM = multiprocessing.cpu_count()
    LOCK = multiprocessing.Lock()
    func = partial(
        lock=LOCK, path=path
    )
    for name, dataset in {"val": val, "train": train}.items():
        with multiprocessing.Pool(CPU_NUM) as pool:
            tqdm(
                pool.imap_unordered(
                    func,
                    dataset.items()),
                desc=name,
                position=0,
                total=len(dataset))
        # for [
            # idx_data,
            # images_data,
            # masks_data,
            # edges_data,
            # weight_maps_data,
            # labels_data] in results:
            # dataset_file_append(path, idx_data, f"{name}/idx")
            # dataset_file_append(path, labels_data, f"{name}/labels")
            # dataset_file_append(path, images_data, f"{name}/images")
            # dataset_file_append(path, masks_data, f"{name}/masks")
            # dataset_file_append(path, edges_data, f"{name}/edges")
            # dataset_file_append(path, weight_maps_data, f"{name}/weight_maps")


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
    results, val_blocks = process_images(
        annotations=annotations,
        translations=translations,
        label_map=label_map,
        image_ranges=image_ranges,
        input_path=input_path)
    print("Splitting to train val.")
    result_train, result_val = dataset_split(results, val_blocks)
    del results
    del val_blocks

    print("Generating hdf5 file from Datasets.")
    create_dataset(result_train, result_val, path=dataset_path)
