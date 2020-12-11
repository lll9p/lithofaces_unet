import numpy as np
import torch
def expand_as_one_hot(input, labels, ignore_labels=None):
    """
    Converts NxHxW label image to NxCxHxW, where each label gets
    converted to its corresponding one-hot vector
    :param input: 3D input image (NxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 4D output image (NxCxHxW)
    """
    assert input.dim() == 3
    input = input.clone()
    num_classes = len(labels) - len(ignore_labels)
    labels_index = list(range(1, len(labels) + 1))
    ignore_labels_index = sorted(
        map(lambda i: labels.index(i) + 1, ignore_labels))
    for i in ignore_labels_index:
        input[input == i] = 0
        labels_index.pop(labels_index.index(i))
    for i, l in enumerate(labels_index, start=1):
        input[input == l] = i
    # expand the input tensor to Nx1xHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxHxW)
    shape = list(input.size())
    shape[1] = num_classes + 1

    # scatter to get the one-hot tensor
    result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
    # 1 means except background
    # return result[:, 1:, ...]
    # including background
    return result

def iou_score(output, target, *args):
    smooth = 1e-5
    target = expand_as_one_hot(target, *args)
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output[:10, :, :] = output[:10, :, :] * 3.0
    #target[:10, :, :] = target[:10, :, :] * 3.0
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target, *args):
    smooth = 1e-5
    target = expand_as_one_hot(target, *args)
    output = touch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
