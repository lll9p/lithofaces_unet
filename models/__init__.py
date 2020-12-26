from .Nested_UNet import NestedUNet, UNet
from .u2net import U2NET, U2NETP


def get_model(config):
    model = None
    if config.model == "NestedUNet":
        model = NestedUNet(
            config.num_classes, config.input_channels, config.deep_supervision
        )
    if config.model == "UNet":
        model = UNet(
            config.num_classes, config.input_channels
        )
    if config.model == "U2NETP":
        model = U2NETP(config.input_channels, config.num_classes,)
    return model
