import torch
import torch.nn as nn
import torch.nn.functional as F

# COPY FROM https://github.com/wolny/pytorch-3dunet


class _DiceLoss(nn.Module):
    def __init__(self, activation="sigmoid", weight=None):
        super(_DiceLoss, self).__init__()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax2d()
        else:
            self.activation = lambda x: x
        self.weight = weight

    def forward(self, input, target, *args):
        input = self.activation(input)
        per_channel_dice = compute_per_channel_dice(
            input, target, weight=self.weight)
        return 1.0 - torch.mean(per_channel_dice)


class DiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, activation, config):
        super(DiceLoss, self).__init__()
        self.dice = _DiceLoss(activation)
        self.ignore_labels = config.ignore_labels
        self.labels = config.labels
        self.weight = config.weight

    def forward(self, input, target, *args):
        target = expand_as_one_hot(
            target, labels=self.labels, ignore_labels=self.ignore_labels
        )
        return self.dice(input, target)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta, activation, config):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta
        self.dice = _DiceLoss(activation)
        self.ignore_labels = config.ignore_labels
        self.labels = config.labels
        self.weight = config.weight

    def forward(self, input, target, *args):
        target = expand_as_one_hot(
            target, labels=self.labels, ignore_labels=self.ignore_labels
        )
        bce = self.bce(input, target)
        dice = self.dice(input, target)
        return self.alpha * bce + self.beta * dice


class BCEPixelWiseDiceLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        beta=None,
        gamma=None,
        activation="sigmoid",
        weight=None,
        config=None,
    ):
        super(BCEPixelWiseDiceLoss, self).__init__()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax2d()
        else:
            self.activation = lambda x: x
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = weight
        self.labels = config.labels
        self.ignore_labels = config.ignore_labels
        self.dice = _DiceLoss(activation, weight=self.weight)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, input, target, weights=None):
        target = expand_as_one_hot(target, self.labels, self.ignore_labels)
        # expand weights
        weights = weights.unsqueeze(1)  # 0->1
        weights = weights.expand_as(input).to(input.device)
        input_wmse = self.activation(input)
        wmse = (
            self.mse(
                input_wmse,
                target) *
            weights /
            target.shape[1] /
            target.shape[0]).mean()
        bce = self.bce(input, target)
        dice = self.dice(input, target)
        # print(wmse.item(),bce.item(),dice.item())
        return (
            self.alpha * bce
            + self.beta * dice
            + self.gamma * wmse
        )


class PixelWiseDiceLoss(nn.Module):
    def __init__(
            self,
            beta=None,
            gamma=None,
            weight=None,
            activation="sigmoid",
            config=None):
        super(PixelWiseDiceLoss, self).__init__()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax2d()
        else:
            self.activation = lambda x: x
        self.beta = beta
        self.gamma = gamma
        self.weight = weight
        self.labels = config.labels
        self.ignore_labels = config.ignore_labels
        self.dice = _DiceLoss(activation, weight=self.weight)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, input, target, weights=None):
        target = expand_as_one_hot(target, self.labels, self.ignore_labels)
        # expand weights
        weights = weights.unsqueeze(1)  # 0->1
        weights = weights.expand_as(input).to(input.device)
        input_wmse = self.activation(input)
        wmse = (
            self.mse(
                input_wmse,
                target) *
            weights /
            target.shape[1] /
            target.shape[0]).mean()
        dice = self.dice(input, target)
        return self.beta * dice + self.gamma * wmse


class DistanceLoss(nn.Module):
    def __init__(self, config):
        super(DistanceLoss, self).__init__()
        self.alpha = config.loss_alpha
        self.beta = config.loss_beta
        self.shape_distance_criterion = nn.SmoothL1Loss()
        self.neighbor_distance_criterion = nn.SmoothL1Loss()

    def forward(
            self,
            shape_distance_true,
            neighbor_distance_true,
            shape_distance,
            neighbor_distance):
        return self.alpha * self.shape_distance_criterion(
            shape_distance_true,
            shape_distance) + \
            self.beta * self.neighbor_distance_criterion(
            neighbor_distance_true,
            neighbor_distance)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, config, class_weights=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_labels = config.ignore_labels
        self.labels = config.labels
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW),
        # so we need to expand it to (NxCxDxHxW)
        # expand to background,class1,class2,class3... noneed background
        target = expand_as_one_hot(
            target, labels=self.labels, ignore_labels=self.ignore_labels
        )
        # expand weights
        weights = weights.unsqueeze(1)  # 0->1
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        # class_weights = class_weights.view(1, -1, 1, 1, 1)
        # resize class_weights to be broadcastable into the weights
        # for 2d just NxCxHxW
        class_weights = class_weights.view(1, -1, 1, 1)

        # add weights tensor by class weights
        weights = class_weights + weights.to(input.device)

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797
    given  a multi channel input and target.
    Assumes the input is a normalized probability,
    e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert (
        input.size() == target.size()
    ), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see
    # V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    # return 2 * (intersect / denominator.clamp(min=epsilon))
    # Fix when one of class target is zero.
    return (2.0 * intersect).clamp(min=epsilon) / denominator.clamp(min=epsilon)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, H, W) -> (C, N, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, H, W) -> (C, N *  H * W)
    return transposed.contiguous().view(C, -1)


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
    if num_classes == 0:
        num_classes = 1
    # labels_index = list(range(1, len(labels) + 1))
    # ignore_labels_index = sorted(map(lambda i: labels.index(i) + 1, ignore_labels))
    # for i in ignore_labels_index:
    #     input[input == i] = 0
    #     labels_index.pop(labels_index.index(i))
    # for i, l in enumerate(labels_index, start=1):
    #     input[input == l] = i
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


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing
    the 'loss' key
    :return: an instance of the loss function
    """
    name = config.loss
    labels = config.labels
    ignore_labels = config.ignore_labels
    weight = config.weight
    if weight is not None:
        weight_ = []
        for label in labels:
            if label in ignore_labels:
                continue
            weight_.append(weight[label])
        weight_ = torch.Tensor(weight_)
        weight_ /= weight_.max()
        # weight need ignore
        weight = weight_.to(config.device)
    loss = _create_loss(name, config, weight)
    return loss


SUPPORTED_LOSSES = [
    "DiceLoss",
    "BCEDiceLoss",
    "BCEPixelWiseDiceLoss",
    "PixelWiseCrossEntropyLoss",
    "PixelWiseDiceLoss",
    "DistanceLoss"
]


def _create_loss(name, config, weight):
    if name == "DiceLoss":
        return DiceLoss(activation=config.dice_activation, config=config)
    elif name == "BCEDiceLoss":
        return BCEDiceLoss(
            config.loss_alpha, config.loss_beta, config.dice_activation, config
        )
    elif name == "BCEPixelWiseDiceLoss":
        return BCEPixelWiseDiceLoss(
            alpha=config.loss_alpha,
            beta=config.loss_beta,
            gamma=config.loss_gamma,
            activation=config.dice_activation,
            config=config
        )
    elif name == "PixelWiseCrossEntropyLoss":
        return PixelWiseCrossEntropyLoss(config, class_weights=weight)
    elif name == "PixelWiseDiceLoss":
        return PixelWiseDiceLoss(
            config.loss_beta,
            config.loss_gamma,
            config.dice_activation,
            weight=weight,
            config=config,
        )
    elif name == "DistanceLoss":
        return DistanceLoss(config)
    else:
        raise RuntimeError(
            f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}"
        )
