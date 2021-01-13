import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torch.backends.cudnn as cudnn
from scipy import ndimage
from skimage import measure
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.segmentation import watershed
from torchvision import transforms

from config import Config
from models import get_model

ALITE, BLITE, PORE, EDGE = 1, 2, 3, 4


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


def get_type(rigion, labels, mask):
    values = mask[labels == rigion.label]
    type_sum = np.array([(values == 0).sum(),
                         (values == ALITE).sum(),
                         (values == BLITE).sum(),
                         (values == PORE).sum(),
                         (values == EDGE).sum()])
    type_ = np.argmax(type_sum)
    return type_


def predict(image_path, imid, model):
    image = cv2.imread(str(image_path))
    image_name = image_path.stem
    x,y = image.shape[:2]
    r = 1
    xx,yy=int(x/r),int(y/r)
    if xx%2==1:
        xx+=1
    if yy%2==1:
        yy+=1
    image = cv2.resize(image,(yy,xx))
    input_ = normalize(image).unsqueeze(0)
    output = model(input_.to("cuda"))
    output = torch.sigmoid(output).cpu()
    print(f"{time.time()-t0:.1f}-model eval complete")
    mask = torch.argmax(output[0], 0).numpy()
    mask_edge = (mask == EDGE).astype(np.int)
    mask_pore = (mask == PORE).astype(np.int)
    mask_pore_edge = (sobel(mask_pore) > 0).astype(np.int)
    mask_binary = (mask > 0).astype(np.int) - mask_edge
    mask_binary[mask_pore.astype(bool)] = 0
    distance = ndimage.distance_transform_edt(mask_binary)
    distance = distance / distance.max()
    distance = ndimage.gaussian_filter(distance, 3)
    coor = peak_local_max(
        distance,
        exclude_border=False,
        min_distance=45)
    mask_ = np.zeros(distance.shape, dtype=bool)
    mask_[tuple(coor.T)] = True
    markers, _ = ndimage.label(mask_)
    labels = watershed(-distance, markers, mask=mask_binary,
                       watershed_line=False)
    props = measure.regionprops(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    types = ["Back", "Alite", "Blite", "Pore", "Edge"]
    stats = dict()
    image[sobel(labels) > 0] = 0
    np.save(f"./app/{image_name}.npy",labels)
    for rigion in props:
        if rigion.area < 300:
            continue
        rigion_type = get_type(rigion, labels, mask)
        type_ = types[rigion_type]
        stats.setdefault(type_, dict())
        stats[type_][str(rigion.label)] = int(rigion.major_axis_length)
        # draw label
        y, x = rigion.centroid
        if rigion_type == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.putText(image, f"{rigion.label}",
                    (int(x), int(y)), font, 0.5, color, 2)
    cv2.imwrite(f"./app/{image_name}.png", image)
    # for k, v in stats.items():
    #     with open(f"./app/result-{k}.csv", "w") as f:
    #         for label_id, data in v.items():
    #             f.write(f"{label_id},{data}\n")


def show(mask, fs=(16, 12)):
    plt.figure(figsize=fs)
    plt.axis("off")
    plt.imshow(mask, cmap="gray")


def showlabel(label, fs=(16, 12)):
    rand_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
    plt.figure(figsize=fs)
    plt.axis("off")
    plt.imshow(mask, cmap=rand_cmap)


if __name__ == "__main__":
    # with open("./app/fn.txt", "r") as file:
    # image_path = file.readline()
    p = r"C:\Users\lao\Documents\Sync\Notebook\Research\Clinker Lithofacies Automation\data\segmentation\images\\"
    p = r"/home/lao/Notebook/Research/Clinker Lithofacies Automation/data/segmentation/images/"
    config = Config.from_yml("./app/config.yml")
    print(config.model.name)
    # no GPU
    # cudnn.benchmark =True
    t0 = time.time()
    print("0.0-model constructing")
    model = get_model(config).cuda()
    print(f"{time.time()-t0:.1f}-model parameters loading")
    t0 = time.time()
    model.load_state_dict(
        torch.load(
            "./app/model.pth",
            map_location=torch.device('cuda')))
    print(f"{time.time()-t0:.1f}-model evaling")
    t0 = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    import pathlib
    ims = list(pathlib.Path(p).glob("*.JPG"))
    with torch.no_grad():
        for i, ii in enumerate(ims):
            predict(ii, i, model)
